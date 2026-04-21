"""
Part 1 (Lab 9): FeedForward and single AttentionHead (self-attention).
Task: FeedForward (linear–ReLU–linear–dropout), AttentionHead (Q,K,V + causal mask + softmax).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


def _test_config():
    """Minimal config for testing modules (block_size, n_embd, n_head, dropout)."""
    return SimpleNamespace(block_size=8, n_embd=32, n_head=4, dropout=0.1)


class FeedForward(nn.Module):
    """Two linear layers with ReLU and dropout: n_embd -> 4*n_embd -> n_embd."""

    def __init__(self, config):
        super().__init__()
        """
        TODO: Implement the FeedForward module with two linear layers, ReLU and dropout.
        Order: Linear(n_embd, 4*n_embd) → ReLU → Linear(4*n_embd, n_embd) → Dropout.
        """
        raise NotImplementedError("TODO: Implement the FeedForward module.")


    def forward(self, x):
        """
        TODO: Implement the forward pass: pass x through the network and return the result.
        """
        raise NotImplementedError("TODO: Implement the forward pass of the FeedForward module.")


class AttentionHead(nn.Module):
    """Single self-attention head: Q, K, V linear projections; causal mask; softmax; output = attention @ V."""

    def __init__(self, config, head_size):
        super().__init__()
        """
        TODO: Implement the AttentionHead module: key, query, value (nn.Linear to head_size);
        register_buffer('tril', torch.tril(...)) for causal mask; dropout.
        """
        raise NotImplementedError("TODO: Implement the AttentionHead module.")

    def forward(self, x):
        """
        TODO: Implement the forward pass: Q,K,V from x; wei = scaled Q@K^T; mask; softmax; dropout; out = wei @ V.
        """
        raise NotImplementedError("TODO: Implement the forward pass of the AttentionHead module.")


# ------ Tests ------

def test_feedforward_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run FeedForward, then output shape is (B,T,C) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    ff = FeedForward(config)
    out = ff(x)
    assert out.shape == (B, T, C), "FeedForward shape"
    assert not torch.isnan(out).any(), "FeedForward output should not be NaN"
    print("  test_feedforward_shape_and_finite [OK]")


def test_head_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run AttentionHead, then output shape is (B,T,head_size) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    head_size = config.n_embd // config.n_head
    head = AttentionHead(config, head_size)
    out = head(x)
    assert out.shape == (B, T, head_size), "AttentionHead shape"
    assert not torch.isnan(out).any(), "AttentionHead output should not be NaN"
    print("  test_head_shape_and_finite [OK]")


def main():
    print("=== Part 1: FeedForward and single AttentionHead ===\n")
    test_feedforward_shape_and_finite()
    test_head_shape_and_finite()
    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    main()
