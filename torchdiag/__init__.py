"""
torchdiag — PyTorch Model Health Diagnostics
=============================================

A lightweight diagnostic toolkit for PyTorch models.
Built from an SRE perspective: observe, measure, debug.

Usage:
    import torchdiag

    torchdiag.summary(model)
    torchdiag.check_gradients(model)
    torchdiag.check_dead_neurons(model, sample_input)
    torchdiag.verify_step(model, optimizer, loss_fn, x, y)
    torchdiag.memory_report()
"""

__version__ = "0.1.0"

from torchdiag.diagnostics import (
    summary,
    check_gradients,
    check_dead_neurons,
    verify_step,
    memory_report,
)

__all__ = [
    "summary",
    "check_gradients",
    "check_dead_neurons",
    "verify_step",
    "memory_report",
]
