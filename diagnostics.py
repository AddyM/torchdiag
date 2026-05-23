"""Core diagnostic functions for PyTorch models."""

import os
import torch
import torch.nn as nn
from typing import Optional, Callable


def summary(model: nn.Module) -> None:
    """Print a health summary of the model: layers, parameters, memory, devices.

    Args:
        model: Any ``torch.nn.Module``.

    Example::

        >>> model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        >>> torchdiag.summary(model)
    """
    print("=" * 70)
    print("MODEL HEALTH SUMMARY")
    print("=" * 70)

    total_params = 0
    trainable_params = 0
    devices = set()
    dtypes = {}

    print(f"\n{'Layer':<40} {'Shape':<20} {'Params':>10}")
    print("-" * 70)

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        devices.add(str(param.device))
        dtype_name = str(param.dtype)
        dtypes[dtype_name] = dtypes.get(dtype_name, 0) + num_params

        print(f"{name:<40} {str(list(param.shape)):<20} {num_params:>10,}")

    frozen_params = total_params - trainable_params
    param_memory_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    print("-" * 70)
    print(f"{'Total parameters':<40} {'':20} {total_params:>10,}")
    print(f"{'Trainable':<40} {'':20} {trainable_params:>10,}")
    print(f"{'Frozen':<40} {'':20} {frozen_params:>10,}")
    print(f"\nParameter memory:  {param_memory_mb:.2f} MB")
    print(f"Devices:           {', '.join(sorted(devices))}")
    print(f"Dtypes:            {', '.join(f'{k} ({v:,})' for k, v in dtypes.items())}")

    # Check for potential issues
    issues = []
    if frozen_params == total_params:
        issues.append("ALL parameters are frozen — model will not learn")
    if len(devices) > 1:
        issues.append(f"Parameters split across devices: {devices}")

    if issues:
        print(f"\n⚠ WARNINGS:")
        for issue in issues:
            print(f"  • {issue}")

    print("=" * 70)


def check_gradients(model: nn.Module) -> dict:
    """Check gradient health after a backward pass.

    Call this after ``loss.backward()`` to diagnose vanishing or
    exploding gradients, or broken computational graphs.

    Args:
        model: Model after ``loss.backward()`` has been called.

    Returns:
        Dictionary with gradient statistics per layer.

    Example::

        >>> loss.backward()
        >>> torchdiag.check_gradients(model)
    """
    print("=" * 70)
    print("GRADIENT HEALTH CHECK")
    print("=" * 70)

    stats = {}
    issues = []

    print(f"\n{'Layer':<35} {'Mean':>12} {'Max':>12} {'Min':>12} {'Status':>10}")
    print("-" * 70)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            print(f"{name:<35} {'—':>12} {'—':>12} {'—':>12} {'NO GRAD':>10}")
            issues.append(f"{name}: gradient is None — disconnected from graph")
            stats[name] = {"status": "no_grad"}
            continue

        grad = param.grad.data
        grad_mean = grad.mean().item()
        grad_max = grad.abs().max().item()
        grad_min = grad.abs().min().item()

        # Determine status
        if grad_max > 100:
            status = "EXPLODE"
            issues.append(f"{name}: gradient max {grad_max:.2f} — exploding")
        elif grad_max < 1e-7:
            status = "VANISH"
            issues.append(f"{name}: gradient max {grad_max:.2e} — vanishing")
        else:
            status = "OK"

        print(f"{name:<35} {grad_mean:>12.6f} {grad_max:>12.6f} {grad_min:>12.6f} {status:>10}")
        stats[name] = {
            "mean": grad_mean,
            "max": grad_max,
            "min": grad_min,
            "status": status,
        }

    if issues:
        print(f"\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print(f"\n✓ All gradients healthy")

    print("=" * 70)
    return stats


def check_dead_neurons(
    model: nn.Module,
    sample_input: torch.Tensor,
    threshold: float = 0.0,
) -> dict:
    """Detect dead ReLU neurons by running a forward pass.

    A dead neuron outputs zero for every sample in the input batch,
    meaning its gradient is permanently zero and it will never learn.

    Args:
        model: The model to check.
        sample_input: A batch of inputs (larger batch = more reliable check).
        threshold: Activation threshold below which a neuron is considered dead.

    Returns:
        Dictionary mapping layer names to dead neuron counts.

    Example::

        >>> x = torch.randn(100, 784)  # 100 samples
        >>> torchdiag.check_dead_neurons(model, x)
    """
    print("=" * 70)
    print("DEAD NEURON CHECK")
    print("=" * 70)

    dead_report = {}
    hooks = []
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks on activation layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        model(sample_input)

    # Analyze activations
    print(f"\n{'Layer':<40} {'Total':>8} {'Dead':>8} {'Dead %':>8} {'Status':>10}")
    print("-" * 70)

    issues = []
    for name, activation in activations.items():
        if activation.dim() < 2:
            continue

        # Check which neurons are dead across all samples
        # Reshape to (batch, neurons) for FC layers, or (batch, channels, ...) for conv
        if activation.dim() == 2:
            neuron_activity = activation
        else:
            # For conv layers, check per channel
            neuron_activity = activation.flatten(2).mean(dim=2)

        total_neurons = neuron_activity.shape[1]
        dead_mask = (neuron_activity.abs() <= threshold).all(dim=0)
        dead_count = dead_mask.sum().item()
        dead_pct = (dead_count / total_neurons) * 100

        if dead_pct > 50:
            status = "CRITICAL"
            issues.append(f"{name}: {dead_pct:.0f}% neurons dead")
        elif dead_pct > 20:
            status = "WARNING"
            issues.append(f"{name}: {dead_pct:.0f}% neurons dead")
        else:
            status = "OK"

        print(f"{name:<40} {total_neurons:>8} {dead_count:>8} {dead_pct:>7.1f}% {status:>10}")
        dead_report[name] = {
            "total": total_neurons,
            "dead": dead_count,
            "dead_pct": dead_pct,
        }

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    if not activations:
        print("\n  No activation layers (ReLU, LeakyReLU, ELU, GELU) found.")
        print("  Tip: check_dead_neurons only inspects nn.ReLU etc., not")
        print("  functional calls like F.relu(). Use nn.ReLU() in your model.")

    if issues:
        print(f"\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\n  Possible causes: learning rate too high, poor weight init,")
        print(f"  or input distribution shifted heavily negative.")
    elif activations:
        print(f"\n✓ All neurons active")

    print("=" * 70)
    return dead_report


def verify_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
) -> dict:
    """Run one training step and verify everything works.

    Checks: forward pass produces output, loss is finite, gradients
    flow, optimizer updates parameters.

    Args:
        model: The model to verify.
        optimizer: Configured optimizer.
        loss_fn: Loss function (e.g., ``nn.CrossEntropyLoss()``).
        sample_input: One batch of input data.
        sample_target: Corresponding targets.

    Returns:
        Dictionary with verification results.

    Example::

        >>> torchdiag.verify_step(model, optimizer, nn.CrossEntropyLoss(),
        ...                       torch.randn(8, 784), torch.randint(0, 10, (8,)))
    """
    print("=" * 70)
    print("TRAINING STEP VERIFICATION")
    print("=" * 70)

    results = {}
    issues = []

    # Snapshot parameters before step
    param_snapshots = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Forward pass
    try:
        output = model(sample_input)
        print(f"\n✓ Forward pass:    output shape {list(output.shape)}")
        results["forward"] = "OK"
    except Exception as e:
        print(f"\n✗ Forward pass:    FAILED — {e}")
        results["forward"] = f"FAILED: {e}"
        issues.append(f"Forward pass failed: {e}")
        print("=" * 70)
        return results

    # Loss computation
    try:
        loss = loss_fn(output, sample_target)
        loss_val = loss.item()
        print(f"✓ Loss:            {loss_val:.6f}")
        results["loss"] = loss_val

        if not torch.isfinite(loss):
            issues.append(f"Loss is {loss_val} — not finite")
            print(f"  ⚠ Loss is not finite!")
    except Exception as e:
        print(f"✗ Loss:            FAILED — {e}")
        results["loss"] = f"FAILED: {e}"
        issues.append(f"Loss computation failed: {e}")
        print("=" * 70)
        return results

    # Backward pass
    try:
        optimizer.zero_grad()
        loss.backward()
        print(f"✓ Backward pass:   gradients computed")
        results["backward"] = "OK"

        # Check for None gradients
        none_grads = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        if none_grads:
            issues.append(f"None gradients: {', '.join(none_grads)}")
            print(f"  ⚠ {len(none_grads)} parameter(s) have None gradients")

    except Exception as e:
        print(f"✗ Backward pass:   FAILED — {e}")
        results["backward"] = f"FAILED: {e}"
        issues.append(f"Backward pass failed: {e}")
        print("=" * 70)
        return results

    # Optimizer step
    optimizer.step()

    # Verify parameters changed
    changed = 0
    unchanged = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in param_snapshots:
            diff = (param.data - param_snapshots[name]).abs().mean().item()
            if diff > 0:
                changed += 1
            else:
                unchanged += 1

    print(f"✓ Optimizer step:  {changed} params updated, {unchanged} unchanged")
    results["params_updated"] = changed
    results["params_unchanged"] = unchanged

    if unchanged > 0 and changed == 0:
        issues.append("No parameters changed — model is not learning")

    if issues:
        print(f"\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print(f"\n✓ Training step verified — all systems go")

    print("=" * 70)
    return results


def memory_report() -> dict:
    """Print GPU and CPU memory usage.

    Returns:
        Dictionary with memory statistics.

    Example::

        >>> torchdiag.memory_report()
    """
    print("=" * 70)
    print("MEMORY REPORT")
    print("=" * 70)

    stats = {}

    # CPU memory
    try:
        import resource
        cpu_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports in bytes, Linux in KB
        import platform
        if platform.system() == "Darwin":
            cpu_rss_mb = cpu_rss_mb / (1024 * 1024)
        else:
            cpu_rss_mb = cpu_rss_mb / 1024

        print(f"\nCPU:")
        print(f"  Peak RSS:        {cpu_rss_mb:.1f} MB")
        stats["cpu_peak_rss_mb"] = cpu_rss_mb
    except ImportError:
        print(f"\nCPU:  (resource module not available)")

    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(i) / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
            total = torch.cuda.get_device_properties(i).total_mem / (1024 * 1024)

            print(f"\nGPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"  Allocated:       {allocated:.1f} MB")
            print(f"  Cached:          {cached:.1f} MB")
            print(f"  Peak allocated:  {peak:.1f} MB")
            print(f"  Total:           {total:.1f} MB")
            print(f"  Utilization:     {(allocated / total) * 100:.1f}%")

            stats[f"gpu_{i}"] = {
                "allocated_mb": allocated,
                "cached_mb": cached,
                "peak_mb": peak,
                "total_mb": total,
            }

            if allocated / total > 0.9:
                print(f"  ⚠ GPU memory >90% — risk of OOM")
    elif torch.backends.mps.is_available():
        print(f"\nMPS (Apple Silicon):")
        allocated = torch.mps.driver_allocated_memory() / (1024 * 1024)
        print(f"  Driver allocated: {allocated:.1f} MB")
        stats["mps_allocated_mb"] = allocated
    else:
        print(f"\nGPU:  No CUDA or MPS device available")

    print("=" * 70)
    return stats
