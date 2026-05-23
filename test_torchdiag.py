"""Test script for torchdiag — run to verify all diagnostics work."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, ".")
import torchdiag


def main():
    print("\n" + "=" * 70)
    print("TESTING TORCHDIAG v" + torchdiag.__version__)
    print("=" * 70)

    # Create a test model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    # Test 1: Summary
    print("\n\n>>> torchdiag.summary(model)\n")
    torchdiag.summary(model)

    # Test 2: Dead neuron check
    print("\n\n>>> torchdiag.check_dead_neurons(model, x)\n")
    x = torch.randn(100, 784)
    torchdiag.check_dead_neurons(model, x)

    # Test 3: Verify training step
    print("\n\n>>> torchdiag.verify_step(...)\n")
    optimizer = torch.optim.Adam(model.parameters())
    torchdiag.verify_step(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.randn(32, 784),
        torch.randint(0, 10, (32,)),
    )

    # Test 4: Gradient check (need a fresh backward)
    print("\n\n>>> torchdiag.check_gradients(model)\n")
    optimizer.zero_grad()
    x = torch.randn(32, 784)
    loss = nn.CrossEntropyLoss()(model(x), torch.randint(0, 10, (32,)))
    loss.backward()
    torchdiag.check_gradients(model)

    # Test 5: Memory report
    print("\n\n>>> torchdiag.memory_report()\n")
    torchdiag.memory_report()

    print("\n\n✓ All tests passed!\n")


if __name__ == "__main__":
    main()
