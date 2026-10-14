"""omm.py -- shared plumbing for the Options Market Making lab (Tissot-Daguette & Fathi, slides 6-24).

Conventions (all three notebooks)
  * time is measured in trading days: tau = 1 is one session, so lambda, kappa, alpha, beta, sigma
    are per-day quantities and lambda = 100, kappa = 1.5 are exactly the numbers on slide 9;
  * fills in a step are Poisson(lambda e^{-kappa delta} dt): the exact law of the slide-8 point
    process over a step with the quote held fixed;
  * Black-Scholes with r = 0, vols and maturities in daily units.
"""
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAST = os.environ.get("OMM_FAST") == "1"          # OMM_FAST=1 -> reduced-scale smoke run
_DIV = int(os.environ.get("OMM_SCALE", "1"))      # OMM_SCALE=k -> every budget divided by k

def scale(n, f=20):
    """Training budget (iterations / batch): n at full scale, n/f under OMM_FAST=1, n/k under OMM_SCALE=k."""
    return max(n // (f if FAST else _DIV), 8)

# ----------------------------------------------------------------------------- networks
class MLP(nn.Module):
    def __init__(self, din, dout, width=64, depth=2, zero_last=False):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]          # smooth: DGM needs second derivatives
            d = width
        last = nn.Linear(d, dout)
        if zero_last:                                           # policy nets start at the myopic quote 1/kappa
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)

    def forward(self, x):
        return self.net(x)

def ckpt(net, x):
    """Policy evaluation with activation checkpointing: keeps the DSO rollout graph O(#steps x state)."""
    return checkpoint(net, x, use_reentrant=False)

# ----------------------------------------------------------------------------- Black-Scholes (r = 0)
_SQRT2, _SQRT2PI = math.sqrt(2.0), math.sqrt(2.0 * math.pi)
def Phi(x): return 0.5 * (1.0 + torch.erf(x / _SQRT2))
def phi(x): return torch.exp(-0.5 * x * x) / _SQRT2PI

def bs_call(S, K, T, sig):
    """Call price, delta, gamma and vega (dC/dsig) for dS/S = sig dW, r = 0. Broadcasts."""
    sT = sig * torch.sqrt(T)
    d1 = (torch.log(S / K) + 0.5 * sT ** 2) / sT
    d2 = d1 - sT
    return S * Phi(d1) - K * Phi(d2), Phi(d1), phi(d1) / (S * sT), S * phi(d1) * torch.sqrt(T)

# ----------------------------------------------------------------------------- order flow (slide 8)
RATE_CAP = 10.0   # numerical guard: at most 10 expected fills per step per quote (never binding near the optimum)

def fills(lam_dt):
    """dN ~ Poisson(lambda e^{-kappa delta} dt). Returns (dN, log P(dN | quote)) with the quote dependence kept
    in the log-likelihood (up to a theta-independent constant) for the score-function gradient."""
    lam_dt = lam_dt.clamp(max=RATE_CAP)
    dN = torch.poisson(lam_dt.detach())
    return dN, dN * torch.log(lam_dt) - lam_dt

# ----------------------------------------------------------------------------- DSO (Algorithm 2, slide 12)
def dso_surrogate(rbar, logp, cost, term):
    """Score-function (likelihood-ratio) estimator of grad_theta J(delta^theta).

    The fills' law depends on theta, so plain autograd through a rollout misses the dependence of the inventory
    path on the quotes.  With  f_k = rbar_k - cost_k  (depends on the path up to the start of step k):
        grad J = E[ sum_k grad f_k ]  +  E[ sum_k grad log P(dN_k | quote_k) * G_k ],
        G_k = sum_{j>k} f_j - term   (reward after step k; a batch-mean baseline is subtracted).
    Autograd of the returned surrogate gives exactly this estimate.

    rbar (B,N): compensated spread capture  sum_i delta_i lambda_i e^{-kappa_i delta_i} dt   (differentiable)
    logp (B,N): log-likelihood of the drawn fills given the quotes                           (differentiable)
    cost (B,N): running cost alpha|Q_k|^2 dt (net of any state-dependent P&L drift);  term (B,): beta|Q_tau|^2
    Returns (surrogate, J_hat); J_hat is the sample objective of slide 11."""
    r = rbar.detach() - cost
    G = r.flip(1).cumsum(1).flip(1) - r - term[:, None]
    adv = G - G.mean(0, keepdim=True)
    J_hat = (rbar - cost).sum(1) - term
    return (J_hat + (logp * adv).sum(1)).mean(), J_hat.mean().item()

# ----------------------------------------------------------------------------- DGM helper (Algorithm 3, slide 16)
def grad(y, x):
    """d(sum y)/dx with graph kept, for the automatic differentiation step of Algorithm 3."""
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]

# ----------------------------------------------------------------------------- training / reporting
def train(step, params, iters, lr=1e-3, clip=None, log_every=None, name=""):
    """Adam loop; step() -> (loss_to_minimize, metric_to_print). Returns the metric history."""
    params = list(params)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    log_every = log_every or max(iters // 10, 1)
    hist = []
    for it in range(1, iters + 1):
        opt.zero_grad()
        loss, metric = step()
        loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step(); sched.step()
        hist.append(metric)
        if it % log_every == 0 or it == iters:
            print(f"{name:>4s}  it {it:5d}   {metric:+.4f}")
    return np.array(hist)

def report(name, pnl, J, Q_T, fills_per_day=None, header=False):
    """One line of backtest statistics (means over paths, with the standard error of E[J])."""
    if header:
        print(f"{'policy':>12s} {'E[Pi_tau]':>10s} {'sd[Pi]':>8s} {'E[J]':>10s} {'se':>6s} {'E|Q_tau|':>9s}"
              + ("" if fills_per_day is None else f" {'fills/day':>9s}"))
    n = len(J)
    line = (f"{name:>12s} {pnl.mean():10.3f} {pnl.std():8.3f} {J.mean():10.3f} {J.std() / math.sqrt(n):6.3f} "
            f"{Q_T.abs().mean():9.2f}")
    if fills_per_day is not None: line += f" {fills_per_day:9.1f}"
    print(line)

COL = dict(bid="#d62728", ask="#2ca02c", mid="#7f7f7f", exact="black", dso="#1f77b4", dgm="#ff7f0e", myo="#9467bd")
