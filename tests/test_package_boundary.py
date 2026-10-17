# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the standalone package dependency boundary."""

import ast
import base64
import csv
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import unittest
import zipfile
from email.parser import Parser
from io import StringIO
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "cuda",
        "cutlass",
        "dist_moe",
        "packaging",
        "torch",
        "triton",
        "tvm_ffi",
    }
)
_PROJECT_ROOT = Path(
    os.environ.get("DIST_MOE_SOURCE_ROOT", Path(__file__).parents[1])
).resolve()
_ARTIFACT_DIR = os.environ.get("DIST_MOE_ARTIFACT_DIR")
_EXPECTED_ROOT_EXPORTS = frozenset(
    {
        "Bf16GroupedGemmPreset",
        "BlockScaledConfig",
        "BlockScaledFormat",
        "BlockScaledKernelConfig",
        "Config",
        "Context",
        "create_context",
        "ExecutionOptions",
        "MemoryPlan",
        "PreparedWeight",
        "RMSNormPostprocess",
        "VmmConfig",
        "plan_memory",
        "prepare_block_scaled_weight",
        "relayout_to_block_major",
        "routed_experts",
        "supports_fused_post_expert_rmsnorm",
    }
)
_PUBLIC_PROSE_DENYLIST = (
    "fb.workplace.com",
    "fbcode",
    "hierarchical rs",
    "internal routing kernel",
    "frozen internal",
    "fused internal",
    "ops.interfaces",
    "ops.kernels",
    "ops/",
    "gb200_moe_sol",
    "gb4026",
    "msl bench",
    "sweep2-",
    "tier5",
    "tier6",
    "v13e",
    "triton.language.extra.tlx",
)


class StaticPackageBoundaryTest(unittest.TestCase):
    """Validate package metadata and source without importing the runtime."""

    def test_production_imports_are_standalone(self) -> None:
        """Reject absolute imports outside the package dependency boundary."""
        package = _PROJECT_ROOT / "dist_moe"
        violations: list[str] = []

        for source in sorted(package.rglob("*.py")):
            for node in ast.walk(ast.parse(source.read_text())):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules.append(node.module or "")
                for module in modules:
                    root = module.partition(".")[0]
                    if (
                        root not in sys.stdlib_module_names
                        and root not in _ALLOWED_IMPORT_ROOTS
                    ):
                        violations.append(f"{source.relative_to(package)}: {module}")

        self.assertEqual(
            violations,
            [],
            "imports outside the package boundary: " + ", ".join(violations),
        )

    def test_every_root_export_is_documented(self) -> None:
        """Require every intentional root export to link to a local guide."""

        def exported_names(module: str) -> set[str]:
            """Read one module's literal ``__all__`` declaration."""
            tree = ast.parse((_PROJECT_ROOT / "dist_moe" / module).read_text())
            assignment = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
            )
            return set(ast.literal_eval(assignment.value))

        exports = exported_names("__init__.py")
        api_exports = exported_names("api.py")
        self.assertEqual(exports, _EXPECTED_ROOT_EXPORTS)
        self.assertEqual(exports, api_exports)
        readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        api_table = readme.split("## Public API", 1)[1].split("\n## ", 1)[0]
        linked_exports = {
            name.removeprefix("dist_moe.")
            for name, _target in re.findall(r"\[`([^`]+)`\]\(([^)]+)\)", api_table)
        }
        self.assertEqual(linked_exports, exports)

    def test_readme_links_and_snippets_are_structurally_valid(self) -> None:
        """Validate local links, fences, and Python snippets in every guide."""
        documents = [
            _PROJECT_ROOT / "README.md",
            _PROJECT_ROOT / "CONTRIBUTING.md",
            _PROJECT_ROOT / "AGENTS.md",
            _PROJECT_ROOT / "tests" / "AGENTS.md",
            *sorted((_PROJECT_ROOT / "docs").glob("*.md")),
        ]
        for document in documents:
            text = document.read_text(encoding="utf-8")
            self.assertEqual(
                text.count("```") % 2,
                0,
                f"unbalanced Markdown fences in {document.name}",
            )
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                path = target.split("#", 1)[0]
                self.assertTrue((document.parent / path).exists(), target)
            for index, program in enumerate(
                text.split("```python")[1:],
                start=1,
            ):
                compile(
                    program.split("```", 1)[0],
                    f"{document.name}:python-block-{index}",
                    "exec",
                )

    def test_public_tests_explain_their_invariants(self) -> None:
        """Require every public test to explain the invariant it protects."""
        missing: list[str] = []
        for path in sorted((_PROJECT_ROOT / "tests").glob("test_*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name.startswith("test")
                    and ast.get_docstring(node) is None
                ):
                    missing.append(f"{path.name}:{node.lineno}:{node.name}")
        self.assertEqual(
            missing, [], "undocumented public tests: " + ", ".join(missing)
        )

    def test_documented_callables_exist(self) -> None:
        """Require every callable named in public prose to resolve."""
        callable_prefixes = (
            "broadcast_",
            "conditional_",
            "copy_",
            "create_",
            "dist_",
            "get_",
            "grouped_",
            "plan_",
            "prefetch_",
            "prepare_",
            "reduce_",
            "scale_",
            "supports_",
        )
        documented: set[str] = set()
        for path in (
            _PROJECT_ROOT / "README.md",
            *sorted((_PROJECT_ROOT / "docs").glob("*.md")),
        ):
            documented.update(
                name
                for name in re.findall(r"`([A-Za-z_]\w*)\([^`]*\)`", path.read_text())
                if name.startswith(callable_prefixes)
            )

        defined: set[str] = set()
        for path in (_PROJECT_ROOT / "dist_moe").rglob("*.py"):
            defined.update(
                node.name
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        self.assertEqual(documented - defined, set())

    def test_notice_covers_modified_apache_sources(self) -> None:
        """Keep shipped modified-Apache source and NOTICE paths identical."""
        package = _PROJECT_ROOT / "dist_moe"
        expected = {
            str(Path("dist_moe") / source.relative_to(package))
            for source in package.rglob("*.py")
            if "Licensed under the Apache License, Version 2.0."
            in source.read_text(encoding="utf-8")
        }
        notice = (_PROJECT_ROOT / "NOTICE").read_text(encoding="utf-8")
        recorded = {
            line.strip()
            for line in notice.splitlines()
            if line.startswith("  dist_moe/")
        }
        self.assertEqual(recorded, expected)

    def test_public_artifact_prose_is_self_contained(self) -> None:
        """Reject private names and links from files shipped publicly."""
        roots = (
            _PROJECT_ROOT / "README.md",
            _PROJECT_ROOT / "CONTRIBUTING.md",
            _PROJECT_ROOT / "AGENTS.md",
            _PROJECT_ROOT / "tests" / "AGENTS.md",
            _PROJECT_ROOT / "dist_moe",
            _PROJECT_ROOT / "docs",
            _PROJECT_ROOT / "examples",
        )
        violations: list[str] = []
        for root in roots:
            candidates = (root,) if root.is_file() else root.rglob("*")
            for path in candidates:
                if not path.is_file() or path.suffix not in {".md", ".py"}:
                    continue
                prose = path.read_text(encoding="utf-8").lower()
                for forbidden in _PUBLIC_PROSE_DENYLIST:
                    if forbidden in prose:
                        violations.append(
                            f"{path.relative_to(_PROJECT_ROOT)}: {forbidden}"
                        )
        self.assertEqual(
            violations, [], "private public prose: " + ", ".join(violations)
        )

    def test_release_workflow_defaults_to_validation_only(self) -> None:
        """Require least-privilege CPU-only CI and manual release publication."""
        workflow = (
            _PROJECT_ROOT / ".github" / "workflows" / "publish_release.yml"
        ).read_text(encoding="utf-8")
        ci_workflow = (_PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        publish_input = workflow.split("publish_release:", 1)[1].split(
            "concurrency:", 1
        )[0]
        self.assertIn("default: false", publish_input)
        self.assertEqual(workflow.count("if: inputs.publish_release"), 3)
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertEqual(workflow.count("persist-credentials: false"), 1)
        self.assertEqual(ci_workflow.count("persist-credentials: false"), 1)
        self.assertNotIn("self-hosted", workflow + ci_workflow)
        self.assertNotIn("ci-blackwell", workflow + ci_workflow)
        self.assertIn("needs: build", workflow)
        self.assertIn("needs: [build, stage-release]", workflow)
        self.assertIn("environment: pypi", workflow)
        for selected_class in (
            "test_package_boundary.py::StaticPackageBoundaryTest",
            "test_package_boundary.py::DistributionArtifactTest",
        ):
            self.assertIn(selected_class, workflow)
            self.assertIn(selected_class, ci_workflow)


class RuntimeDocumentationTest(unittest.TestCase):
    """Validate documentation that imports or executes the selected runtime."""

    def test_package_import_does_not_require_tlx(self) -> None:
        """Import the package while making the optional TLX module unavailable."""
        program = """
import importlib.abc
import sys

class BlockTlx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton.language.extra.tlx":
            raise ImportError("TLX intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockTlx())
import dist_moe
"""
        subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.distribution_artifact
@unittest.skipUnless(
    _ARTIFACT_DIR,
    "DIST_MOE_ARTIFACT_DIR must name the directory containing built artifacts",
)
class DistributionArtifactTest(unittest.TestCase):
    """Validate the exact wheel and sdist selected for release."""

    @classmethod
    def setUpClass(cls) -> None:
        """Resolve exactly one wheel and one source distribution."""
        assert _ARTIFACT_DIR is not None
        artifact_dir = Path(_ARTIFACT_DIR)
        wheels = sorted(artifact_dir.glob("dist_moe-*.whl"))
        sdists = sorted(artifact_dir.glob("dist_moe-*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise AssertionError(
                "expected exactly one dist_moe wheel and sdist, got "
                f"{wheels!r} and {sdists!r}"
            )
        cls.wheel = wheels[0]
        cls.sdist = sdists[0]

    @staticmethod
    def _assert_no_private_members(members: list[str]) -> None:
        """Reject internal, generated, or local-only archive members."""
        forbidden_names = {
            ".codex",
            ".pytest_cache",
            ".ruff_cache",
            "BUCK",
            "PACKAGE",
            "__pycache__",
            "fb",
        }
        violations = [
            member
            for member in members
            if Path(member).is_absolute()
            or any(part in forbidden_names for part in Path(member).parts)
            or member.endswith((".pyc", ".pyo"))
        ]
        if violations:
            raise AssertionError(f"private archive members: {violations}")

    @staticmethod
    def _source_files(*roots: str) -> dict[str, bytes]:
        """Return release source bytes below the requested project roots.

        Args:
            *roots: Project-relative file or directory names.

        Returns:
            POSIX paths mapped to their exact source bytes.
        """
        files: dict[str, bytes] = {}
        for root_name in roots:
            root = _PROJECT_ROOT / root_name
            candidates = (root,) if root.is_file() else root.rglob("*")
            for path in candidates:
                if path.is_file() and "__pycache__" not in path.parts:
                    files[path.relative_to(_PROJECT_ROOT).as_posix()] = (
                        path.read_bytes()
                    )
        return files

    def test_wheel_contains_only_public_package_and_metadata(self) -> None:
        """The wheel exactly mirrors public source and valid RECORD metadata."""
        with zipfile.ZipFile(self.wheel) as archive:
            members = archive.namelist()
            self._assert_no_private_members(members)
            source = self._source_files("dist_moe")
            dist_info = {
                member.partition("/")[0]
                for member in members
                if ".dist-info/" in member
            }
            self.assertEqual(len(dist_info), 1)
            metadata_root = next(iter(dist_info))
            metadata_members = {
                f"{metadata_root}/METADATA",
                f"{metadata_root}/RECORD",
                f"{metadata_root}/WHEEL",
                f"{metadata_root}/top_level.txt",
                f"{metadata_root}/licenses/LICENSE",
                f"{metadata_root}/licenses/NOTICE",
                f"{metadata_root}/licenses/LICENSES/Apache-2.0.txt",
            }
            self.assertEqual(set(members), set(source) | metadata_members)
            for name, expected in source.items():
                self.assertEqual(archive.read(name), expected, name)
            for name in ("LICENSE", "NOTICE", "LICENSES/Apache-2.0.txt"):
                self.assertEqual(
                    archive.read(f"{metadata_root}/licenses/{name}"),
                    (_PROJECT_ROOT / name).read_bytes(),
                    name,
                )

            record_name = f"{metadata_root}/RECORD"
            records = {
                row[0]: (row[1], row[2])
                for row in csv.reader(
                    StringIO(archive.read(record_name).decode("utf-8"))
                )
            }
            self.assertEqual(set(records), set(members))
            for name in members:
                digest, size = records[name]
                if name == record_name:
                    self.assertEqual((digest, size), ("", ""))
                    continue
                payload = archive.read(name)
                encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
                self.assertEqual(digest, "sha256=" + encoded.rstrip(b"=").decode())
                self.assertEqual(size, str(len(payload)))

            project = tomllib.loads(
                (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
            )["project"]
            metadata = Parser().parsestr(
                archive.read(f"{metadata_root}/METADATA").decode("utf-8")
            )
            self.assertEqual(metadata["Name"], project["name"])
            self.assertEqual(metadata["Version"], project["version"])
            self.assertEqual(
                SpecifierSet(metadata["Requires-Python"]),
                SpecifierSet(project["requires-python"]),
            )
            self.assertEqual(
                {Requirement(value) for value in metadata.get_all("Requires-Dist", [])},
                {Requirement(value) for value in project["dependencies"]},
            )

    def test_sdist_contains_only_public_source_inputs(self) -> None:
        """The sdist exactly contains public sources and generated metadata."""
        with tarfile.open(self.sdist, mode="r:gz") as archive:
            entries = archive.getmembers()
            members = [entry.name for entry in entries]
            self._assert_no_private_members(members)
            symlinks = [
                entry.name for entry in entries if entry.issym() or entry.islnk()
            ]
            self.assertEqual(symlinks, [], f"sdist links are not allowed: {symlinks}")
            roots = {Path(member).parts[0] for member in members}
            self.assertEqual(len(roots), 1)
            files = {
                Path(*Path(entry.name).parts[1:]).as_posix(): entry
                for entry in entries
                if entry.isfile()
            }
            source = self._source_files(
                ".github",
                ".pre-commit-config.yaml",
                "AGENTS.md",
                "CONTRIBUTING.md",
                "examples",
                "LICENSE",
                "LICENSES",
                "MANIFEST.in",
                "NOTICE",
                "README.md",
                "dist_moe",
                "docs",
                "pyproject.toml",
                "tests",
            )
            generated = {
                "PKG-INFO",
                "setup.cfg",
                "dist_moe.egg-info/PKG-INFO",
                "dist_moe.egg-info/SOURCES.txt",
                "dist_moe.egg-info/dependency_links.txt",
                "dist_moe.egg-info/requires.txt",
                "dist_moe.egg-info/top_level.txt",
            }
            self.assertEqual(set(files), set(source) | generated)
            for name, expected in source.items():
                extracted = archive.extractfile(files[name])
                assert extracted is not None
                self.assertEqual(extracted.read(), expected, name)

            project = tomllib.loads(
                (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
            )["project"]
            pkg_info = archive.extractfile(files["PKG-INFO"])
            assert pkg_info is not None
            metadata = Parser().parsestr(pkg_info.read().decode("utf-8"))
            self.assertEqual(metadata["Name"], project["name"])
            self.assertEqual(metadata["Version"], project["version"])
            self.assertEqual(
                SpecifierSet(metadata["Requires-Python"]),
                SpecifierSet(project["requires-python"]),
            )
