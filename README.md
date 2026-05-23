# torchdiag

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**PyTorch model health diagnostics — built from an SRE perspective.**

Stop guessing why your model isn't learning. `torchdiag` gives you five diagnostic commands that answer the questions that matter: Are gradients flowing? Are neurons alive? Did the optimizer actually update weights?

## Installation

```bash
pip install torchdiag
```

## Quick Start

```python
import torch
import torch.nn as nn
import torchdiag

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# 1. Model overview
torchdiag.summary(model)

# 2. Check for dead neurons
x = torch.randn(100, 784)
torchdiag.check_dead_neurons(model, x)

# 3. Verify a full training step works
torchdiag.verify_step(
    model,
    torch.optim.Adam(model.parameters()),
    nn.CrossEntropyLoss(),
    torch.randn(32, 784),
    torch.randint(0, 10, (32,)),
)

# 4. Check gradient health (after backward)
x = torch.randn(32, 784)
loss = nn.CrossEntropyLoss()(model(x), torch.randint(0, 10, (32,)))
loss.backward()
torchdiag.check_gradients(model)

# 5. Memory usage
torchdiag.memory_report()
```

## What Each Command Does

### `torchdiag.summary(model)`

Prints parameter count per layer, total/trainable/frozen breakdown, memory footprint, device placement, and dtype distribution. Flags issues like all-frozen parameters or split-device models.

### `torchdiag.check_gradients(model)`

Call after `loss.backward()`. Reports gradient mean, max, and min per layer. Flags vanishing gradients (max < 1e-7), exploding gradients (max > 100), and disconnected parameters (None gradients).

### `torchdiag.check_dead_neurons(model, sample_input)`

Runs a forward pass and checks activation layers for neurons that output zero for every sample. Reports dead neuron count and percentage per layer. Flags critical layers (>50% dead) and warnings (>20% dead).

### `torchdiag.verify_step(model, optimizer, loss_fn, x, y)`

Runs one complete training step (forward → loss → backward → step) and verifies each stage works: output shape is correct, loss is finite, gradients are computed, and parameters actually change.

### `torchdiag.memory_report()`

Reports CPU peak RSS, GPU memory (allocated, cached, peak, total) per device, and MPS memory on Apple Silicon. Flags when GPU utilization exceeds 90%.

## Why This Exists

Most PyTorch debugging happens by staring at loss curves. That's like monitoring a distributed system by watching a single dashboard number.

`torchdiag` brings SRE observability practices to model training:

- **Measure, don't guess** — print the actual gradient values, don't assume they're fine
- **Check preconditions** — verify the training step works before running 100 epochs
- **Detect silent failures** — dead neurons and None gradients don't raise errors

## Requirements

- Python 3.8+
- PyTorch 2.0+

## License

MIT

## Author

[Aditya Mehra](https://github.com/AddyM) — Staff Engineer, IEEE Senior Member, PyTorch ecosystem contributor.
