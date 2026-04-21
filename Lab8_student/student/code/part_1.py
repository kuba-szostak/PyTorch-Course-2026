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
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class AttentionHead(nn.Module):
    """Single self-attention head: Q, K, V linear projections; causal mask; softmax; output = attention @ V."""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores with scaling
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Apply softmax
        wei = F.softmax(wei, dim=-1)

        # Apply dropout
        wei = self.dropout(wei)

        # Weighted aggregation of values
        out = wei @ v  # (B, T, head_size)
        return out


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
