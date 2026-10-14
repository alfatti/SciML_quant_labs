# Options Market Making — hands-on lab (slide 25)

Reference implementations of the three lab items, one notebook each, on top of a small shared module.

| notebook | problem | methods |
|---|---|---|
| `lab1_spot_bachelier.ipynb` | spot MM, Bachelier (slides 6–19) | DSO, DGM, exact HJB solution, backtest |
| `lab2_options_localvol.ipynb` | single-expiry options MM, local vol, 10 options (slides 22–24) | DSO, DGM on v(t,S,Q), hedged-P&L identity check |
| `lab3_options_heston.ipynb` | multi-expiry options MM, Heston, 100 options | DSO (shared per-option quoting net), DGM via the separable HJB |
| `omm.py` | MLP, Black-Scholes greeks, Poisson fills, DSO score-function surrogate, DGM autograd helper, training loop | |

## Running on the Brev box

```bash
pip install -r requirements.txt      # PyTorch with CUDA is normally preinstalled on the image
jupyter lab                          # run the notebooks top to bottom; omm.py must sit next to them
```

Everything runs on one GPU in minutes (lab 1 ≈ 3 min, lab 2 ≈ 5 min, lab 3 ≈ 15 min on an A10/L4-class card).
`OMM_FAST=1 jupyter lab` runs every notebook at 1/20 of the training budget for a quick end-to-end check.

## Conventions

* Time in trading days: τ = 1 is one session, so λ = 100, κ = 1.5 (slide 9) and α, β, σ are per-day quantities.
* Fills per 1-minute step are Poisson(λ e^{−κδ} dt), the exact law of the slide-8 point process with the quote held fixed over the step.
* DSO (Algorithm 2): the fills' law depends on θ, so ∇θ Ĵ is the score-function estimator (`omm.dso_surrogate`, reward-to-go with a batch baseline); verified against finite differences.
* DGM (Algorithm 3): residual of the reduced HJB (sup solved in closed form) + weighted terminal condition; spreads recovered by the finite differences of slide 17.
* Options: Black-Scholes with r = 0; implied vols constant over the session; hedged book value Π_τ = spread capture + ½∫(σ_t² − σ²(K,T)) Q Γ^$ dt (realized minus implied — a delta-hedged long option earns when realized > implied; slide 23 has the sign the other way round). |Q|² = Σ_i (Q^i)², as on slide 24.
* Lab 3: with |Q|² = Σ_i (Q^i)² the HJB for v(t,S,σ²,Q) separates into 100 four-dimensional problems v_i(t,S,σ²,Q^i); one network conditioned on (K_i, T_i, σ_i) solves all of them.
