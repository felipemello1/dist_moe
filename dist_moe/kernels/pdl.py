# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Programmatic dependent launch (PDL) for the BF16 inference kernel chain.

With PDL a kernel is scheduled while its predecessor finishes. Every PDL-capable
kernel (Triton ``PDL``, CuTe ``use_pdl``) waits (``griddepcontrol.wait``) before
its first global read or write, so it still sees all of the predecessor's
results, and then lets its own successor launch. Only the launch latency and the
predecessor's tail overlap:

    without PDL:  [kernel A ......][gap][kernel B ......]
    with PDL:     [kernel A ......]
                         [B scheduled, waits][kernel B ......]

A launch sets ``launch_pdl`` only for a kernel compiled with its wait;
``pdl_launch_kwargs`` returns both together.
"""

import contextlib
import threading
from collections.abc import Iterator

import triton
import triton.language as tl

_STATE = threading.local()


def pdl_enabled() -> bool:
    """Return whether launches on this thread use programmatic dependent launch."""
    return getattr(_STATE, "enabled", False)


def pdl_launch_kwargs() -> dict[str, bool]:
    """Return the Triton launch kwargs for this thread: the kernel's ``PDL`` and ``launch_pdl``."""
    enabled = pdl_enabled()
    return {"PDL": enabled, "launch_pdl": enabled}


@contextlib.contextmanager
def pdl_launches(enabled: bool) -> Iterator[None]:
    """Make PDL-capable launches inside the block use PDL when ``enabled``.

    Example:
        >>> with pdl_launches(config.inference):
        ...     swiglu_fwd(...)  # launched with PDL in inference
    """
    previous = pdl_enabled()
    _STATE.enabled = enabled
    try:
        yield
    finally:
        _STATE.enabled = previous


@triton.jit
def pdl_wait_then_release(PDL: tl.constexpr):
    """Wait for the previous kernel, then let the next one be scheduled.

    Call before the kernel's first global read or write. The next kernel waits
    for this one to finish before its own first access, so releasing it right
    after the wait is safe and overlaps its launch with this kernel's body.
    """
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
