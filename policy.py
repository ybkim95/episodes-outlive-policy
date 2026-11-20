"""Tiny causal transformer policy. Each agent turn is two tokens: a verb, then an argument.

act() samples and records behaviour log-probs; score() recomputes log-probs of stored actions with grad.
Both share masked_logp so sampling and training use the same action masks.
"""
import torch
from torch import nn

from env import ARG_IDS, VERB_IDS, VOCAB


def _mask(ids):
    m = torch.zeros(len(VOCAB), dtype=torch.bool)
    m[torch.as_tensor(ids)] = True
    return m


class Policy(nn.Module):
    def __init__(self, max_len, d=64, layers=2, heads=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(len(VOCAB), d), nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, len(VOCAB))
        self.register_buffer("verb_ok", _mask(VERB_IDS))
        self.register_buffer("arg_ok", _mask(ARG_IDS))

    def forward(self, x):  # (N, L) -> logits (N, L, V)
        L = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        return self.head(self.ln(self.body(h, mask=causal, is_causal=True)))


def masked_logp(logits, ok):
    return torch.log_softmax(logits.masked_fill(~ok, float("-inf")), -1)


@torch.no_grad()
def act(model, ctx, gen):
    """ctx (B, L) -> actions (B, 2), behaviour log-probs (B, 2)."""
    lp_v = masked_logp(model(ctx)[:, -1], model.verb_ok)
    v = torch.multinomial(lp_v.exp(), 1, generator=gen)
    lp_a = masked_logp(model(torch.cat([ctx, v], 1))[:, -1], model.arg_ok)
    a = torch.multinomial(lp_a.exp(), 1, generator=gen)
    return torch.cat([v, a], 1), torch.cat([lp_v.gather(1, v), lp_a.gather(1, a)], 1)


def score(model, ctx, act):
    """Log-probs (N, 2) of stored actions under model, in one forward over ctx + verb."""
    logits = model(torch.cat([ctx, act[:, :1]], 1))
    lp_v = masked_logp(logits[:, -2], model.verb_ok).gather(1, act[:, :1])
    lp_a = masked_logp(logits[:, -1], model.arg_ok).gather(1, act[:, 1:])
    return torch.cat([lp_v, lp_a], 1)
