#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS = REPO_ROOT / "figs"
HIST = FIGS / "history"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def wiener_paths(n_paths: int, n_steps: int, T: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    t = np.linspace(0.0, T, n_steps + 1)
    return t, W

def ou_paths(n_paths: int, n_steps: int, T: float, x0: float,
             kappa: float, theta: float, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = x0
    for k in range(n_steps):
        dW = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        X[:, k + 1] = X[:, k] + kappa * (theta - X[:, k]) * dt + sigma * dW
    return t, X

def gbm_paths(n_paths: int, n_steps: int, T: float, S0: float,
              mu: float, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for k in range(n_steps):
        dW = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        S[:, k + 1] = S[:, k] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    return t, S

def make_01_brownian_paths_static() -> None:
    t, W = wiener_paths(n_paths=7, n_steps=900, T=1.0, seed=1)
    fig = plt.figure(figsize=(6.2, 3.8))
    ax = plt.gca()
    for i in range(W.shape[0]):
        ax.plot(t, W[i], linewidth=1.6, alpha=0.9)
    ax.set_title(r"Wiener process $W_t$: sample paths")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$W_t$")
    save(fig, FIGS / "01_brownian_paths.png")

def make_01_brownian_paths_anim(n_frames: int) -> None:
    out = HIST / "01_brownian_paths_anim"
    ensure_dir(out)
    n_steps = 900
    t, W = wiener_paths(n_paths=7, n_steps=n_steps, T=1.0, seed=1)
    for i in range(n_frames):
        frac = i / max(1, n_frames - 1)
        idx = max(2, int(frac * n_steps))
        fig = plt.figure(figsize=(6.2, 3.8))
        ax = plt.gca()
        for p in range(W.shape[0]):
            ax.plot(t[:idx], W[p, :idx], linewidth=1.6, alpha=0.9)
        ax.set_title(r"Wiener process $W_t$: paths unfold in time")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$W_t$")
        ax.set_xlim(0, 1.0)
        ymin = np.min(W[:, :idx])
        ymax = np.max(W[:, :idx])
        pad = 0.15 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin - pad, ymax + pad)
        save(fig, out / f"frame_{i}.png")

def make_02_ou_paths_static() -> None:
    t, X = ou_paths(
        n_paths=7, n_steps=900, T=2.0, x0=2.0,
        kappa=2.2, theta=0.0, sigma=0.75, seed=2
    )
    fig = plt.figure(figsize=(6.2, 3.8))
    ax = plt.gca()
    for i in range(X.shape[0]):
        ax.plot(t, X[i], linewidth=1.6, alpha=0.9)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title(r"Ornstein–Uhlenbeck (mean-reverting) paths")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$X_t$")
    save(fig, FIGS / "02_ou_paths.png")

def make_02_ou_paths_anim(n_frames: int) -> None:
    out = HIST / "02_ou_paths_anim"
    ensure_dir(out)
    n_steps = 900
    t, X = ou_paths(
        n_paths=7, n_steps=n_steps, T=2.0, x0=2.0,
        kappa=2.2, theta=0.0, sigma=0.75, seed=2
    )
    for i in range(n_frames):
        frac = i / max(1, n_frames - 1)
        idx = max(2, int(frac * n_steps))
        fig = plt.figure(figsize=(6.2, 3.8))
        ax = plt.gca()
        for p in range(X.shape[0]):
            ax.plot(t[:idx], X[p, :idx], linewidth=1.6, alpha=0.9)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(r"OU process: paths unfold (mean reversion)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$X_t$")
        ax.set_xlim(0, 2.0)
        ymin = np.min(X[:, :idx])
        ymax = np.max(X[:, :idx])
        pad = 0.15 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin - pad, ymax + pad)
        save(fig, out / f"frame_{i}.png")

def make_03_gbm_paths_static() -> None:
    t, S = gbm_paths(
        n_paths=7, n_steps=700, T=1.0, S0=100.0,
        mu=0.08, sigma=0.25, seed=3
    )
    fig = plt.figure(figsize=(6.2, 3.8))
    ax = plt.gca()
    for i in range(S.shape[0]):
        ax.plot(t, S[i], linewidth=1.6, alpha=0.9)
    ax.set_title(r"Geometric Brownian Motion (GBM): sample price paths")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$S_t$")
    save(fig, FIGS / "03_gbm_paths.png")

def make_03_gbm_paths_anim(n_frames: int) -> None:
    out = HIST / "03_gbm_paths_anim"
    ensure_dir(out)
    n_steps = 700
    t, S = gbm_paths(
        n_paths=7, n_steps=n_steps, T=1.0, S0=100.0,
        mu=0.08, sigma=0.25, seed=3
    )
    for i in range(n_frames):
        frac = i / max(1, n_frames - 1)
        idx = max(2, int(frac * n_steps))
        fig = plt.figure(figsize=(6.2, 3.8))
        ax = plt.gca()
        for p in range(S.shape[0]):
            ax.plot(t[:idx], S[p, :idx], linewidth=1.6, alpha=0.9)
        ax.set_title(r"GBM: paths unfold (multiplicative noise)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$S_t$")
        ax.set_xlim(0, 1.0)
        ymin = np.min(S[:, :idx])
        ymax = np.max(S[:, :idx])
        pad = 0.10 * (ymax - ymin + 1e-9)
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
        save(fig, out / f"frame_{i}.png")

def make_04_gbm_terminal_hist_static() -> None:
    t, S = gbm_paths(n_paths=7000, n_steps=400, T=1.0, S0=100.0, mu=0.08, sigma=0.25, seed=4)
    ST = S[:, -1]
    fig = plt.figure(figsize=(6.0, 3.8))
    ax = plt.gca()
    ax.hist(ST, bins=45, density=True, alpha=0.9)
    ax.set_title(r"GBM terminal distribution: $S_T$ is (approximately) lognormal")
    ax.set_xlabel(r"$S_T$")
    ax.set_ylabel("density")
    save(fig, FIGS / "04_gbm_ST_hist.png")

def make_04_gbm_terminal_hist_anim(n_frames: int) -> None:
    out = HIST / "04_gbm_hist_anim"
    ensure_dir(out)
    t, S = gbm_paths(n_paths=7000, n_steps=400, T=1.0, S0=100.0, mu=0.08, sigma=0.25, seed=4)
    ST = S[:, -1]
    bins = np.linspace(np.min(ST), np.max(ST), 45)
    xlim = (max(0.0, np.percentile(ST, 0.5)), np.percentile(ST, 99.5))
    for i in range(n_frames):
        frac = i / max(1, n_frames - 1)
        k = max(50, int(50 + frac * (len(ST) - 50)))
        fig = plt.figure(figsize=(6.0, 3.8))
        ax = plt.gca()
        ax.hist(ST[:k], bins=bins, density=True, alpha=0.9)
        ax.set_title(rf"GBM $S_T$ histogram: using {k} samples")
        ax.set_xlabel(r"$S_T$")
        ax.set_ylabel("density")
        ax.set_xlim(*xlim)
        save(fig, out / f"frame_{i}.png")

def make_05_noisy_integrate_fire_static() -> None:
    rng = np.random.default_rng(0)
    T = 1.0
    n = 2000
    dt = T / n
    t = np.linspace(0, T, n + 1)
    V = np.zeros(n + 1)
    V[0] = 0.2
    tau = 0.02
    I = 1.1
    sigma = 0.35
    thr = 1.0
    reset = 0.1
    spikes = []
    for k in range(n):
        dW = rng.normal(0.0, math.sqrt(dt))
        dV = (-(V[k]) / tau + I) * dt + sigma * dW
        V[k + 1] = V[k] + dV
        if V[k + 1] >= thr:
            spikes.append(t[k + 1])
            V[k + 1] = reset
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = plt.gca()
    ax.plot(t, V, linewidth=1.2)
    for s in spikes[:50]:
        ax.axvline(s, alpha=0.15)
    ax.set_title("Noisy Integrate-and-Fire: membrane potential with spikes")
    ax.set_xlabel("t")
    ax.set_ylabel("V(t)")
    save(fig, FIGS / "05_noisy_if.png")

def make_06_forward_diffusion_static() -> None:
    rng = np.random.default_rng(123)
    n = 12000
    mix = rng.uniform(size=n) < 0.5
    x0 = np.where(mix, rng.normal(-2.0, 0.6, size=n), rng.normal(2.0, 0.8, size=n))
    sigmas = [0.0, 0.5, 1.0, 2.0]
    fig = plt.figure(figsize=(7.4, 3.8))
    for j, s in enumerate(sigmas, 1):
        ax = plt.subplot(1, len(sigmas), j)
        xt = x0 + s * rng.normal(size=n)
        ax.hist(xt, bins=40, density=True, alpha=0.9)
        ax.set_title(rf"$\sigma={s}$")
        ax.set_xlabel("x")
        if j == 1:
            ax.set_ylabel("density")
        ax.grid(alpha=0.18)
    plt.suptitle("Forward diffusion 1D: add noise → distribution becomes simpler")
    save(fig, FIGS / "06_forward_diffusion_1d.png")

def make_06_forward_diffusion_anim(n_frames: int) -> None:
    out = HIST / "forward1d_anim"
    ensure_dir(out)
    rng = np.random.default_rng(123)
    n = 14000
    mix = rng.uniform(size=n) < 0.5
    x0 = np.where(mix, rng.normal(-2.0, 0.6, size=n), rng.normal(2.0, 0.8, size=n))
    z = rng.normal(size=n)
    sigmas = np.linspace(0.0, 3.0, n_frames)
    x_min = np.percentile(x0 + sigmas[-1] * z, 0.5)
    x_max = np.percentile(x0 + sigmas[-1] * z, 99.5)
    bins = np.linspace(x_min, x_max, 50)
    for i, s in enumerate(sigmas):
        xt = x0 + s * z
        fig = plt.figure(figsize=(6.0, 3.8))
        ax = plt.gca()
        ax.hist(xt, bins=bins, density=True, alpha=0.9)
        ax.set_title(rf"Forward diffusion: $\sigma$ increases → simpler $p_t(x)$   (σ={s:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_xlim(x_min, x_max)
        save(fig, out / f"frame_{i}.png")

def gaussian2d_pdf(X: np.ndarray, mean: np.ndarray, var: float) -> np.ndarray:
    d = X - mean
    norm = 1.0 / (2.0 * math.pi * var)
    return norm * np.exp(-0.5 * (d[..., 0]**2 + d[..., 1]**2) / var)

def mixture_density_and_score(grid_x: np.ndarray, grid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.array([[-1.6, -0.7], [1.3, 1.1]], dtype=float)
    vars_ = np.array([0.55**2, 0.70**2], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)
    X = np.stack([grid_x, grid_y], axis=-1)
    p0 = weights[0] * gaussian2d_pdf(X, means[0], vars_[0])
    p1 = weights[1] * gaussian2d_pdf(X, means[1], vars_[1])
    p = p0 + p1 + 1e-12
    grad0 = p0[..., None] * (-(X - means[0]) / vars_[0])
    grad1 = p1[..., None] * (-(X - means[1]) / vars_[1])
    gradp = grad0 + grad1
    score = gradp / p[..., None]
    return p, score, means

def score_at_points(P: np.ndarray) -> np.ndarray:
    means = np.array([[-1.6, -0.7], [1.3, 1.1]], dtype=float)
    vars_ = np.array([0.55**2, 0.70**2], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)
    def pdf_point(x: np.ndarray, m: np.ndarray, v: float) -> np.ndarray:
        d = x - m
        norm = 1.0 / (2.0 * math.pi * v)
        return norm * np.exp(-0.5 * (d[:, 0]**2 + d[:, 1]**2) / v)
    p0 = weights[0] * pdf_point(P, means[0], vars_[0])
    p1 = weights[1] * pdf_point(P, means[1], vars_[1])
    p = p0 + p1 + 1e-12
    grad0 = (p0[:, None]) * (-(P - means[0]) / vars_[0])
    grad1 = (p1[:, None]) * (-(P - means[1]) / vars_[1])
    return (grad0 + grad1) / p[:, None]

def make_07_score_field_static() -> None:
    xs = np.linspace(-4, 4, 41)
    ys = np.linspace(-4, 4, 41)
    Xg, Yg = np.meshgrid(xs, ys)
    p, score, _ = mixture_density_and_score(Xg, Yg)
    fig = plt.figure(figsize=(6.6, 6.2))
    ax = plt.gca()
    ax.contourf(Xg, Yg, p, levels=18)
    U = score[..., 0]
    V = score[..., 1]
    mag = np.sqrt(U**2 + V**2) + 1e-9
    U2, V2 = U / mag, V / mag
    ax.quiver(Xg, Yg, U2, V2, angles="xy", scale_units="xy", scale=22, width=0.003)
    ax.set_title("Score field: direction to higher density (2D mixture)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, FIGS / "07_score_field_2d.png")

def make_07_score_particles_anim(n_frames: int) -> None:
    out = HIST / "score_particles_anim"
    ensure_dir(out)
    xs = np.linspace(-4, 4, 61)
    ys = np.linspace(-4, 4, 61)
    Xg, Yg = np.meshgrid(xs, ys)
    p, score, means = mixture_density_and_score(Xg, Yg)
    rng = np.random.default_rng(7)
    P = rng.normal(0.0, 2.0, size=(140, 2))
    step = 0.08
    for i in range(n_frames):
        fig = plt.figure(figsize=(6.6, 6.2))
        ax = plt.gca()
        ax.contourf(Xg, Yg, p, levels=18)
        U = score[..., 0]
        V = score[..., 1]
        mag = np.sqrt(U**2 + V**2) + 1e-9
        ax.quiver(Xg, Yg, U / mag, V / mag, angles="xy", scale_units="xy", scale=24, width=0.003)
        ax.scatter(P[:, 0], P[:, 1], s=10, alpha=0.9)
        ax.scatter(means[:, 0], means[:, 1], s=120, marker="x")
        ax.set_title("Reverse intuition: particles move along score to high-density regions")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        save(fig, out / f"frame_{i}.png")
        P = P + step * score_at_points(P)
        P = np.clip(P, -4.2, 4.2)

def simulate_ou_single_from_dW(dW: np.ndarray, dt: float, x0: float,
                               kappa: float, theta: float, sigma: float) -> np.ndarray:
    x = np.zeros(len(dW) + 1)
    x[0] = x0
    for i in range(len(dW)):
        x[i + 1] = x[i] + kappa * (theta - x[i]) * dt + sigma * dW[i]
    return x

def make_08_em_stepsize_static() -> None:
    rng = np.random.default_rng(42)
    T = 2.0
    kappa, theta, sigma, x0 = 2.5, 0.0, 0.8, 2.0
    N1 = 200
    dt1 = T / N1
    dW1 = rng.normal(0.0, math.sqrt(dt1), size=N1)
    t1 = np.linspace(0, T, N1 + 1)
    x1 = simulate_ou_single_from_dW(dW1, dt1, x0, kappa, theta, sigma)
    rng = np.random.default_rng(42)
    N2 = 2000
    dt2 = T / N2
    dW2 = rng.normal(0.0, math.sqrt(dt2), size=N2)
    t2 = np.linspace(0, T, N2 + 1)
    x2 = simulate_ou_single_from_dW(dW2, dt2, x0, kappa, theta, sigma)
    fig = plt.figure(figsize=(8.0, 4.0))
    ax = plt.gca()
    ax.plot(t1, x1, label="coarse step (N=200)")
    ax.plot(t2, x2, label="fine step (N=2000)", alpha=0.9)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Euler–Maruyama: effect of step size on OU simulation")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$X_t$")
    ax.legend()
    save(fig, FIGS / "08_em_stepsize.png")

def make_08_em_refine_anim(n_frames: int) -> None:
    out = HIST / "em_refine_anim"
    ensure_dir(out)
    rng = np.random.default_rng(2025)
    T = 2.0
    Nfine = 4000
    dt_f = T / Nfine
    dW_f = rng.normal(0.0, math.sqrt(dt_f), size=Nfine)
    kappa, theta, sigma, x0 = 2.5, 0.0, 0.8, 2.0
    tf = np.linspace(0, T, Nfine + 1)
    xf = simulate_ou_single_from_dW(dW_f, dt_f, x0, kappa, theta, sigma)
    Ns = [50, 100, 200, 400, 800]
    blocks = [Nfine // N for N in Ns]
    coarse = []
    for N, b in zip(Ns, blocks):
        dt = T / N
        dWc = dW_f.reshape(N, b).sum(axis=1)
        tc = np.linspace(0, T, N + 1)
        xc = simulate_ou_single_from_dW(dWc, dt, x0, kappa, theta, sigma)
        coarse.append((tc, xc, N))
    chunk = max(1, n_frames // len(coarse))
    for i in range(n_frames):
        j = min(len(coarse) - 1, i // chunk)
        tc, xc, N = coarse[j]
        fig = plt.figure(figsize=(8.0, 4.0))
        ax = plt.gca()
        ax.plot(tf, xf, alpha=0.65, label=f"reference (fine, N={Nfine})")
        ax.plot(tc, xc, linewidth=2.0, label=f"Euler–Maruyama (N={N})")
        ax.axhline(0.0, linestyle="--")
        ax.set_title("Step size matters: refining Δt reduces discretization error")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$X_t$")
        ax.legend(loc="upper right")
        save(fig, out / f"frame_{i}.png")

def make_history_brownian_particle_anim(n_frames: int) -> None:
    out = HIST / "bm_anim"
    ensure_dir(out)
    rng = np.random.default_rng(11)
    T = 1.0
    n_steps = n_frames
    dt = T / n_steps
    dW = rng.normal(0.0, math.sqrt(dt), size=(n_steps, 2))
    X = np.vstack([np.zeros((1, 2)), np.cumsum(dW, axis=0)])
    pad = 0.6
    xmin, xmax = X[:, 0].min() - pad, X[:, 0].max() + pad
    ymin, ymax = X[:, 1].min() - pad, X[:, 1].max() + pad
    for i in range(n_frames):
        fig = plt.figure(figsize=(5.2, 4.2))
        ax = plt.gca()
        ax.plot(X[: i + 2, 0], X[: i + 2, 1], linewidth=2.0, alpha=0.9)
        ax.scatter([X[i + 1, 0]], [X[i + 1, 1]], s=90)
        ax.set_title("Brownian motion: many micro-kicks → random-looking path")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        save(fig, out / f"frame_{i}.png")

def make_history_wiener_paths_static() -> None:
    t, W = wiener_paths(n_paths=8, n_steps=700, T=1.0, seed=12)
    fig = plt.figure(figsize=(5.4, 4.1))
    ax = plt.gca()
    for i in range(W.shape[0]):
        ax.plot(t, W[i], linewidth=1.5, alpha=0.9)
    ax.set_title(r"Wiener process $W_t$: realizations")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$W_t$")
    save(fig, HIST / "wiener_paths.png")

def make_history_wiener_paths_anim(n_frames: int) -> None:
    out = HIST / "wiener_anim"
    ensure_dir(out)
    n_steps = 700
    t, W = wiener_paths(n_paths=8, n_steps=n_steps, T=1.0, seed=12)
    for i in range(n_frames):
        frac = i / max(1, n_frames - 1)
        idx = max(2, int(frac * n_steps))
        fig = plt.figure(figsize=(5.4, 4.1))
        ax = plt.gca()
        for p in range(W.shape[0]):
            ax.plot(t[:idx], W[p, :idx], linewidth=1.5, alpha=0.9)
        ax.set_title(r"Wiener process $W_t$ (unfolding)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$W_t$")
        ax.set_xlim(0, 1.0)
        ymin = np.min(W[:, :idx])
        ymax = np.max(W[:, :idx])
        pad = 0.15 * (ymax - ymin + 1e-9)
        ax.set_ylim(ymin - pad, ymax + pad)
        save(fig, out / f"frame_{i}.png")

def make_history_ito_qv_static() -> None:
    rng = np.random.default_rng(21)
    T = 1.0
    n = 2000
    dt = T / n
    dW = rng.normal(0.0, math.sqrt(dt), size=n)
    qv = np.cumsum(dW**2)
    t = np.linspace(dt, T, n)
    fig = plt.figure(figsize=(5.8, 3.8))
    ax = plt.gca()
    ax.plot(t, qv, linewidth=2.0)
    ax.plot([0, T], [0, T], linestyle="--", linewidth=1.2)
    ax.set_title(r"Quadratic variation: $\sum (\Delta W)^2 \approx T$")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\sum (\Delta W)^2$")
    save(fig, HIST / "ito_quadratic_variation.png")

def make_history_ito_qv_anim(n_frames: int) -> None:
    out = HIST / "qv_anim"
    ensure_dir(out)
    rng = np.random.default_rng(21)
    T = 1.0
    n = 2000
    dt = T / n
    dW = rng.normal(0.0, math.sqrt(dt), size=n)
    qv = np.cumsum(dW**2)
    t = np.linspace(dt, T, n)
    for i in range(n_frames):
        frac = i / max(1, n_frames - 1)
        idx = max(2, int(frac * n))
        fig = plt.figure(figsize=(5.8, 3.8))
        ax = plt.gca()
        ax.plot(t[:idx], qv[:idx], linewidth=2.0)
        ax.plot([0, T], [0, T], linestyle="--", linewidth=1.2)
        ax.set_title(r"Quadratic variation accumulates (Itô intuition)")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\sum (\Delta W)^2$")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, max(1.1, qv[:idx].max() * 1.1))
        save(fig, out / f"frame_{i}.png")

def make_history_black_scholes_static() -> None:
    rng = np.random.default_rng(99)
    T = 1.0
    n_steps = 240
    n_paths = 120
    S0, mu, sig = 100.0, 0.08, 0.25
    K = 110.0
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for k in range(n_steps):
        dW = rng.normal(0.0, math.sqrt(dt), size=n_paths)
        S[:, k + 1] = S[:, k] * np.exp((mu - 0.5 * sig**2) * dt + sig * dW)
    fig = plt.figure(figsize=(6.4, 4.0))
    ax = plt.gca()
    for i in range(n_paths):
        ax.plot(t, S[i], linewidth=0.9, alpha=0.30)
    ax.axhline(K, linestyle="--", linewidth=1.2)
    ax.text(0.02, K + 2.0, "strike K", fontsize=10)
    ax.set_title("Black–Scholes intuition: lognormal fan + payoff threshold")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$S_t$")
    save(fig, HIST / "black_scholes_intuition.png")

def make_history_black_scholes_anim(n_frames: int) -> None:
    out = HIST / "bs_anim"
    ensure_dir(out)
    rng = np.random.default_rng(99)
    T = 1.0
    n_steps = 240
    n_paths = 120
    S0, mu, sig = 100.0, 0.08, 0.25
    K = 110.0
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for k in range(n_steps):
        dW = rng.normal(0.0, math.sqrt(dt), size=n_paths)
        S[:, k + 1] = S[:, k] * np.exp((mu - 0.5 * sig**2) * dt + sig * dW)
    for i in range(n_frames):
        frac = i / max(1, n_frames - 1)
        idx = max(2, int(frac * n_steps))
        fig = plt.figure(figsize=(6.4, 4.0))
        ax = plt.gca()
        for p in range(n_paths):
            ax.plot(t[:idx], S[p, :idx], linewidth=0.9, alpha=0.30)
        ax.axhline(K, linestyle="--", linewidth=1.2)
        ax.set_title("Black–Scholes fan unfolding in time")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$S_t$")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(60, np.percentile(S[:, :idx], 99.5) + 15)
        save(fig, out / f"frame_{i}.png")

def make_history_diffusion_timeline_static() -> None:
    fig = plt.figure(figsize=(6.6, 2.8))
    ax = plt.gca()
    ax.axis("off")
    ax.text(0.05, 0.60, "data\n(complex)", ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"))
    ax.text(0.50, 0.60, "add noise\n(forward)", ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"))
    ax.text(0.95, 0.60, "noise\n(Gaussian)", ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"))
    ax.annotate("", xy=(0.40, 0.60), xytext=(0.18, 0.60), arrowprops=dict(arrowstyle="->", lw=2.0))
    ax.annotate("", xy=(0.82, 0.60), xytext=(0.60, 0.60), arrowprops=dict(arrowstyle="->", lw=2.0))
    ax.text(0.50, 0.18, "learn reverse: score / denoiser\n(reverse SDE / probability flow ODE)",
            ha="center", va="center", fontsize=11)
    save(fig, HIST / "diffusion_toy_timeline.png")

def make_history_diffusion_anim(n_frames: int) -> None:
    out = HIST / "diffusion_anim"
    ensure_dir(out)
    rng = np.random.default_rng(123)
    n = 14000
    mix = rng.uniform(size=n) < 0.5
    x0 = np.where(mix, rng.normal(-2.0, 0.6, size=n), rng.normal(2.0, 0.8, size=n))
    z = rng.normal(size=n)
    sigmas = np.linspace(0.0, 3.0, n_frames)
    x_min = np.percentile(x0 + sigmas[-1] * z, 0.5)
    x_max = np.percentile(x0 + sigmas[-1] * z, 99.5)
    bins = np.linspace(x_min, x_max, 50)
    for i, s in enumerate(sigmas):
        xt = x0 + s * z
        fig = plt.figure(figsize=(6.0, 3.8))
        ax = plt.gca()
        ax.hist(xt, bins=bins, density=True, alpha=0.9)
        ax.set_title(rf"Forward diffusion (toy): σ={s:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_xlim(x_min, x_max)
        save(fig, out / f"frame_{i}.png")

def make_history_wiener_definition() -> None:
    fig = plt.figure(figsize=(6.4, 3.8))
    ax = plt.gca()
    ax.axis("off")
    txt = (
        "Wiener process $W_t$:\n\n"
        "1) $W_0 = 0$\n"
        "2) independent increments\n"
        "3) $W_{t+\\Delta}-W_t \\sim \\mathcal{N}(0,\\Delta)$\n"
        "4) paths are continuous, but nowhere differentiable\n\n"
        "This is the noise source in SDE: $dW_t$."
    )
    ax.text(0.02, 0.95, txt, va="top", fontsize=12)
    save(fig, HIST / "wiener_definition.png")

def make_history_ito_formula() -> None:
    fig = plt.figure(figsize=(7.0, 3.8))
    ax = plt.gca()
    ax.axis("off")
    txt = (
        "Itô formula (idea):\n\n"
        "If $dX_t = a(X_t,t)\\,dt + b(X_t,t)\\,dW_t$,\n"
        "then for smooth $f$:\n\n"
        "$df(X_t) = f_x\\,dX_t + \\frac{1}{2} f_{xx}\\, b^2\\,dt$  (plus $f_t dt$).\n\n"
        "Key surprise vs ODE: the $\\frac{1}{2} f_{xx} b^2 dt$ term does NOT vanish\n"
        "because $(dW_t)^2 \\sim dt$."
    )
    ax.text(0.02, 0.95, txt, va="top", fontsize=12)
    save(fig, HIST / "ito_formula.png")

def make_history_einstein_equation() -> None:
    fig = plt.figure(figsize=(7.0, 3.4))
    ax = plt.gca()
    ax.axis("off")
    txt = (
        "Einstein (1905): Brownian motion relates microscopic randomness to macroscopic diffusion.\n\n"
        "Mean squared displacement:\n"
        "$\\mathbb{E}[|X_t - X_0|^2] = 2 d D t$  (in $d$ dimensions)\n\n"
        "Diffusion coefficient:\n"
        "$D = \\frac{k_B T}{6\\pi \\eta r}$  (Stokes–Einstein)\n\n"
        "Message: randomness is physics, not measurement error."
    )
    ax.text(0.02, 0.95, txt, va="top", fontsize=12)
    save(fig, HIST / "einstein_equation.png")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200, help="frames per animation sequence (default: 200)")
    args = parser.parse_args()
    n_frames = int(args.frames)
    ensure_dir(FIGS)
    ensure_dir(HIST)
    make_01_brownian_paths_static()
    make_02_ou_paths_static()
    make_03_gbm_paths_static()
    make_04_gbm_terminal_hist_static()
    make_05_noisy_integrate_fire_static()
    make_06_forward_diffusion_static()
    make_07_score_field_static()
    make_08_em_stepsize_static()
    make_01_brownian_paths_anim(n_frames)
    make_02_ou_paths_anim(n_frames)
    make_03_gbm_paths_anim(n_frames)
    make_04_gbm_terminal_hist_anim(n_frames)
    make_06_forward_diffusion_anim(n_frames)
    make_07_score_particles_anim(n_frames)
    make_08_em_refine_anim(n_frames)
    make_history_einstein_equation()
    make_history_wiener_definition()
    make_history_ito_formula()
    make_history_ito_qv_static()
    make_history_black_scholes_static()
    make_history_diffusion_timeline_static()
    make_history_wiener_paths_static()
    make_history_brownian_particle_anim(n_frames)
    make_history_wiener_paths_anim(n_frames)
    make_history_ito_qv_anim(n_frames)
    make_history_black_scholes_anim(n_frames)
    make_history_diffusion_anim(n_frames)
    print("Saved figures into:")
    print("  - ./figs/")
    print("  - ./figs/history/")
    print("  - ./figs/history/*_anim/ (frames)")

if __name__ == "__main__":
    main()