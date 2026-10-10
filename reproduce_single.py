"""Self-contained reproduction entry point for the paper's numerical model.

The model equations, optimisation order, parameter grids, analysis scans and
figure generation are kept together without importing project-local modules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
EXPECTED = ROOT / "expected_outputs"
OUT.mkdir(exist_ok=True)
SEED = 2024

REF = dict(
    lam=1.0,
    gamma=0.8,
    beta0=0.50,
    beta1=0.85,
    nu=0.20,
    q0=0.85,
    q1=0.95,
    alpha=0.40,
    s=0.05,
    F=0.05,
    c0=0.05,
    c1=0.02,
    lam0=0.08,
    lam1=0.02,
    K_AI=0.03,
    e=1.30,
    r=0.15,
    C0=1.00,
    delta0=0.50,
    mu=1.0,
)


class Params:
    """Broadcastable parameter container using the paper's reference values."""

    def __init__(self, **updates):
        values = dict(REF)
        values.update(updates)
        for key, value in values.items():
            setattr(
                self,
                key,
                (
                    np.asarray(value, dtype=float)
                    if not np.isscalar(value)
                    else float(value)
                ),
            )


def bc(value, shape):
    """Align parameter vectors with the leading axis of a price grid."""
    if np.isscalar(value):
        return float(value)
    value = np.asarray(value, dtype=float)
    if value.ndim == 1 and len(shape) > 1:
        value = value.reshape((value.shape[0],) + (1,) * (len(shape) - 1))
    return np.broadcast_to(value, shape)


def solve_rho(demand, alpha_price_plus_subsidy, fixed_cost):
    """Return the nonnegative root of the physician supply equation."""
    demand = np.maximum(demand, 0.0)
    coefficient = np.maximum(alpha_price_plus_subsidy, 1e-12)
    return (
        -fixed_cost
        + np.sqrt(np.maximum(fixed_cost * fixed_cost + 4 * coefficient * demand, 0.0))
    ) / (2 * coefficient)


def doctor_equilibrium(demand, alpha_price_plus_subsidy, fixed_cost, productivity):
    """Solve congestion and supply, respecting the unit physician-mass cap."""
    tau = solve_rho(demand, alpha_price_plus_subsidy, fixed_cost)
    rho_internal = tau / np.maximum(productivity, 1e-12)
    supply_internal = np.clip(
        alpha_price_plus_subsidy * productivity * rho_internal + fixed_cost, 0.0, 1.0
    )
    saturated = supply_internal >= 1.0 - 1e-10
    rho = np.where(saturated, demand / np.maximum(productivity, 1e-12), rho_internal)
    supply = np.where(saturated, 1.0, supply_internal)
    return rho, supply


def fixed_point(update, initial, iters=90, tol=1e-8, damp=0.55):
    """Apply the paper-run damping and batch-wide stopping criterion."""
    state = np.asarray(initial, dtype=float)
    for _ in range(iters):
        next_state = damp * update(state) + (1 - damp) * state
        if np.max(np.abs(next_state - state)) < tol:
            return next_state
        state = next_state
    return state


def baseline_pi_vec(price, p):
    """Evaluate baseline profit (Eq. 7) after demand/supply equilibration."""
    price = np.asarray(price, dtype=float)
    shape = price.shape
    alpha, subsidy = bc(p.alpha, shape), bc(p.s, shape)
    fixed, gamma = bc(p.F, shape), bc(p.gamma, shape)
    lam, productivity = bc(p.lam, shape), bc(p.mu, shape)
    coefficient = alpha * price + subsidy
    demand = fixed_point(
        lambda d: lam
        * np.maximum(
            0.0,
            1.0
            - price
            - gamma * doctor_equilibrium(d, coefficient, fixed, productivity)[0] ** 2,
        ),
        np.full(shape, 0.4),
    )
    rho, supply = doctor_equilibrium(demand, coefficient, fixed, productivity)
    margin = (1 - alpha) * price - subsidy
    return (
        np.where(margin <= 0, -np.inf, margin * demand - fixed * supply),
        demand,
        rho,
        supply,
    )


def baseline_opt_vec(p, grid=201):
    """Search 201 prices, then refine locally with 30 golden-section steps."""
    n = np.size(p.alpha)
    prices = np.broadcast_to(np.linspace(0, 1, grid), (n, grid))
    profits, _, _, _ = baseline_pi_vec(prices, p)
    best = np.where(np.isfinite(profits), profits, -np.inf)
    index = np.argmax(best, axis=1)
    rows = np.arange(n)
    p0, pi0 = prices[rows, index], best[rows, index]
    valid = np.isfinite(pi0)
    lo, hi = np.maximum(0, p0 - 2 / grid), np.minimum(1, p0 + 2 / grid)
    for _ in range(30):
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        q1 = baseline_pi_vec(m1, p)[0]
        q2 = baseline_pi_vec(m2, p)[0]
        cut = q1 >= q2
        hi, lo = np.where(cut, m2, hi), np.where(cut, lo, m1)
    price = (lo + hi) / 2
    pi, demand, rho, supply = baseline_pi_vec(price, p)
    return dict(p=price, Pi=pi, D=demand, rho=rho, S=supply, valid=valid)


def dual_pi_vec(p0, p1, p):
    """Evaluate exclusive service choice, referral demand and dual profit."""
    shape = np.broadcast_shapes(np.asarray(p0).shape, np.asarray(p1).shape)
    p0, p1 = np.broadcast_to(np.asarray(p0, float), shape), np.broadcast_to(
        np.asarray(p1, float), shape
    )
    lam, gamma = bc(p.lam, shape), bc(p.gamma, shape)
    delta0, C0, nu = bc(p.delta0, shape), bc(p.C0, shape), bc(p.nu, shape)
    q0, q1 = bc(p.q0, shape), bc(p.q1, shape)
    beta0, beta1, alpha = bc(p.beta0, shape), bc(p.beta1, shape), bc(p.alpha, shape)
    subsidy, fixed, referral, productivity = (
        bc(p.s, shape),
        bc(p.F, shape),
        bc(p.r, shape),
        bc(p.mu, shape),
    )
    e, c0, c1 = bc(p.e, shape), bc(p.c0, shape), bc(p.c1, shape)
    lam0, lam1, K_AI = bc(p.lam0, shape), bc(p.lam1, shape), bc(p.K_AI, shape)
    mu1 = e * productivity

    def update(state):
        d0, d1 = state[..., 0], state[..., 1]
        total = d1 + referral * d0
        rho1 = doctor_equilibrium(total, alpha * p1 + subsidy, fixed, mu1)[0]
        waiting = delta0 * (d0 / C0) ** 2
        A0 = (
            p0
            + referral * (p1 - p0)
            + gamma * (waiting + referral * rho1**2)
            + nu * ((1 - referral) * (1 - q0) + referral * (1 - q1))
        )
        A1 = p1 + gamma * rho1**2 + nu * (1 - q1)
        theta0 = 1 - A0 / np.maximum(beta0, 1e-12)
        # Doctor participation follows u1(theta)=0, hence theta1 (Eq. 13 typo).
        theta1 = A1 / np.maximum(beta1, 1e-12)
        theta01 = (beta0 - A0 + A1) / (beta0 + beta1 + 1e-12)
        nd0 = lam * np.clip(np.minimum(np.minimum(1, theta0), theta01), 0, 1)
        nd1 = lam * np.clip(1 - np.maximum(np.maximum(0, theta1), theta01), 0, 1)
        return np.stack([nd0, nd1], axis=-1)

    state = fixed_point(
        update, np.stack([np.full(shape, 0.2), np.full(shape, 0.2)], axis=-1), iters=120
    )
    d0, d1 = state[..., 0], state[..., 1]
    waiting = delta0 * (d0 / C0) ** 2
    total = d1 + referral * d0
    rho1, supply = doctor_equilibrium(total, alpha * p1 + subsidy, fixed, mu1)
    profit = (
        (p0 - c0 - lam0) * (1 - referral) * d0
        + ((1 - alpha) * p1 - subsidy - c0 - c1 - lam1) * referral * d0
        + ((1 - alpha) * p1 - subsidy - c1 - lam1) * d1
        - fixed * supply
        - K_AI
    )
    A0 = (
        p0
        + referral * (p1 - p0)
        + gamma * (waiting + referral * rho1**2)
        + nu * ((1 - referral) * (1 - q0) + referral * (1 - q1))
    )
    A1 = p1 + gamma * rho1**2 + nu * (1 - q1)
    theta0, theta1 = 1 - A0 / np.maximum(beta0, 1e-12), A1 / np.maximum(beta1, 1e-12)
    state_name = np.where(
        (d0 + d1) <= 1e-9, "zero", np.where((d0 + d1) < lam - 1e-9, "partial", "full")
    )
    return profit, d0, d1, rho1, waiting, supply, state_name, theta0, theta1


def dual_opt_vec(p, K0=11, K1=21, rounds=3, starts=3):
    """Refine the three best coarse prices using coordinate golden search."""
    n = np.size(p.alpha)
    beta0, beta1 = np.broadcast_to(np.asarray(p.beta0, float), (n,)), np.broadcast_to(
        np.asarray(p.beta1, float), (n,)
    )
    b0, b1 = np.maximum(1e-6, np.maximum(beta0, beta1)), np.maximum(1e-6, beta1)
    g0, g1 = np.linspace(0, 1, K0)[None, :, None], np.linspace(0, 1, K1)[None, None, :]
    p0g, p1g = np.broadcast_arrays(b0[:, None, None] * g0, b1[:, None, None] * g1)
    result = dual_pi_vec(p0g, p1g, p)
    profits = np.where(np.isfinite(result[0]), result[0], -np.inf).reshape(n, -1)
    count = min(starts, profits.shape[1])
    index = np.argpartition(profits, -count, axis=1)[:, -count:]
    i0, i1 = index // K1, index % K1
    rows = np.arange(n)[:, None]
    p0, p1 = p0g[rows, i0, i1], p1g[rows, i0, i1]
    lo0, hi0 = np.maximum(0, p0 - 0.1), np.minimum(b0[:, None], p0 + 0.1)
    lo1, hi1 = np.maximum(0, p1 - 0.1), np.minimum(b1[:, None], p1 + 0.1)
    for _ in range(rounds):
        for _ in range(25):
            m1, m2 = lo0 + (hi0 - lo0) * 0.382, lo0 + (hi0 - lo0) * 0.618
            q1, q2 = dual_pi_vec(m1, p1, p)[0], dual_pi_vec(m2, p1, p)[0]
            cut = q1 >= q2
            hi0, lo0 = np.where(cut, m2, hi0), np.where(cut, lo0, m1)
        p0 = (lo0 + hi0) / 2
        for _ in range(25):
            m1, m2 = lo1 + (hi1 - lo1) * 0.382, lo1 + (hi1 - lo1) * 0.618
            q1, q2 = dual_pi_vec(p0, m1, p)[0], dual_pi_vec(p0, m2, p)[0]
            cut = q1 >= q2
            hi1, lo1 = np.where(cut, m2, hi1), np.where(cut, lo1, m1)
        p1 = (lo1 + hi1) / 2
    final = dual_pi_vec(p0, p1, p)
    best = np.argmax(np.where(np.isfinite(final[0]), final[0], -np.inf), axis=1)
    pick = lambda value: value[np.arange(n), best]
    return dict(
        p0=pick(p0),
        p1=pick(p1),
        Pi=pick(final[0]),
        d0=pick(final[1]),
        d1=pick(final[2]),
        rho1=pick(final[3]),
        w0=pick(final[4]),
        S1=pick(final[5]),
        market_state=pick(final[6]),
        th0=pick(final[7]),
        th1=pick(final[8]),
    )


def solve_vec(p):
    """Compare optimal scenarios, masking invalid price/demand solutions."""
    baseline, dual = baseline_opt_vec(p), dual_opt_vec(p)
    valid = (
        baseline["valid"]
        & np.isfinite(dual["Pi"])
        & (dual["d0"] >= -1e-8)
        & (dual["d1"] >= -1e-8)
    )
    return baseline, dual, np.where(valid, dual["Pi"] - baseline["Pi"], np.nan), valid


def grid_params(alphas, subsidies, fixed_costs, gammas, beta0s, beta1s):
    """Enumerate in the original order and retain beta1 > beta0."""
    values = np.meshgrid(
        alphas, subsidies, fixed_costs, gammas, beta0s, beta1s, indexing="ij"
    )
    flat = [value.ravel() for value in values]
    mask = flat[5] > flat[4]
    return [value[mask] for value in flat]


def core_grid():
    """Compute and export the fixed 10,368-combination paper experiment."""
    A, S, F, G, B0, B1 = grid_params(
        [0.2, 0.4, 0.6, 0.8],
        [0.02, 0.05, 0.08, 0.12],
        [0, 0.05, 0.10],
        np.linspace(0.1, 2.0, 8),
        np.linspace(0.3, 0.8, 6),
        np.linspace(0.7, 1.0, 5),
    )
    print("core grid combos:", A.size)
    b, d, dPi, valid = solve_vec(Params(alpha=A, s=S, F=F, gamma=G, beta0=B0, beta1=B1))
    rows = []
    for i in np.where(valid)[0]:
        rows.append(
            dict(
                alpha=float(A[i]),
                s=float(S[i]),
                F=float(F[i]),
                gamma=float(G[i]),
                beta0=float(B0[i]),
                beta1=float(B1[i]),
                p=float(b["p"][i]),
                PiA=float(b["Pi"][i]),
                D=float(b["D"][i]),
                rhoA=float(b["rho"][i]),
                p0=float(d["p0"][i]),
                p1=float(d["p1"][i]),
                d0=float(d["d0"][i]),
                d1=float(d["d1"][i]),
                rho1=float(d["rho1"][i]),
                w0=float(d["w0"][i]),
                S1=float(d["S1"][i]),
                PiB=float(d["Pi"][i]),
                dPi=float(dPi[i]),
                market_state=str(d["market_state"][i]),
            )
        )
    with (OUT / "core_grid.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "core_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"n": len(rows)}, handle)
    values = np.array([row["dPi"] for row in rows])
    print(
        "valid:",
        len(rows),
        "positive %.2f%% | negative %.2f%%"
        % (100 * np.mean(values > 1e-6), 100 * np.mean(values < -1e-6)),
    )
    return rows


def run_combos(overrides):
    """Solve a parameter batch without altering its iteration stopping rule."""
    return solve_vec(
        Params(
            **{
                key: np.array([item.get(key, REF[key]) for item in overrides], float)
                for key in REF
            }
        )
    )


def save_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_data(xname, yname, xvals, yvals, fixed):
    overrides = [{**fixed, xname: x, yname: y} for y in yvals for x in xvals]
    b, d, diff, valid = run_combos(overrides)
    rows = []
    for i, (y, x) in enumerate(((y, x) for y in yvals for x in xvals)):
        rows.append(
            dict(
                x=x,
                y=y,
                dPi=float(diff[i]) if valid[i] else np.nan,
                PiB=float(d["Pi"][i]) if valid[i] else np.nan,
                PiA=float(b["Pi"][i]) if valid[i] else np.nan,
            )
        )
    return rows


def analysis():
    """Export phase diagrams, reference counterfactuals and robustness sweeps."""
    base = dict(REF)
    K = np.linspace(0, 0.20, 41)
    B0 = np.linspace(0.30, 0.80, 41)
    G = np.linspace(0.10, 2, 41)
    specs = [
        ("phase_beta0_gamma03.csv", "K_AI", "beta0", K, B0, {**base, "gamma": 0.3}),
        ("phase_beta0_gamma15.csv", "K_AI", "beta0", K, B0, {**base, "gamma": 1.5}),
        ("phase_gamma_beta04.csv", "K_AI", "gamma", K, G, {**base, "beta0": 0.4}),
        ("phase_gamma_beta07.csv", "K_AI", "gamma", K, G, {**base, "beta0": 0.7}),
        (
            "phase_alpha.csv",
            "K_AI",
            "alpha",
            K,
            np.linspace(0.15, 0.85, 35),
            {**base, "gamma": 0.8},
        ),
        (
            "phase_F.csv",
            "K_AI",
            "F",
            K,
            np.linspace(0, 0.20, 35),
            {**base, "gamma": 0.8},
        ),
        (
            "phase_r.csv",
            "K_AI",
            "r",
            K,
            np.linspace(0, 0.50, 35),
            {**base, "gamma": 0.8},
        ),
        ("phase_e.csv", "K_AI", "e", K, np.linspace(1, 2, 35), {**base, "gamma": 0.8}),
    ]
    for name, xname, yname, xs, ys, fixed in specs:
        save_csv(name, phase_data(xname, yname, xs, ys, fixed))
    b, d, diff, valid = run_combos([base])
    pA, piA, DA, SA = b["p"][0], b["Pi"][0], b["D"][0], b["S"][0]
    p0, p1, d0, d1, S1, piB = (
        d["p0"][0],
        d["p1"][0],
        d["d0"][0],
        d["d1"][0],
        d["S1"][0],
        d["Pi"][0],
    )

    def cf(**updates):
        _, dd, _, _ = run_combos([{**base, **updates}])
        return float(dd["Pi"][0])

    components = {
        "AI_non_escalated_net": (p0 - base["c0"] - base["lam0"]) * (1 - base["r"]) * d0,
        "escalated_case_net": (
            (1 - base["alpha"]) * p1
            - base["s"]
            - base["c0"]
            - base["c1"]
            - base["lam1"]
        )
        * base["r"]
        * d0,
        "direct_AI_doctor_net": (
            (1 - base["alpha"]) * p1 - base["s"] - base["c1"] - base["lam1"]
        )
        * d1,
        "baseline_profit_removed": -((1 - base["alpha"]) * pA - base["s"]) * DA,
        "subsidy_change": -base["F"] * S1 + base["F"] * SA,
        "AI_fixed_cost": -base["K_AI"],
    }
    save_csv(
        "mechanism.csv",
        [
            dict(component=k, value=v)
            for k, v in list(components.items())
            + [("dPi_total", float(diff[0])), ("check", float(piB - piA))]
        ],
    )
    save_csv(
        "counterfactual.csv",
        [
            dict(item=k, value=v)
            for k, v in {
                "eff_gain": piB - cf(e=1.0),
                "esc_loss": cf(r=0.0) - piB,
                "cap_wait_loss": cf(C0=100.0) - piB,
                "fixed_cost": base["K_AI"],
            }.items()
        ],
    )
    threshold_rows = []
    for b0 in [0.35, 0.50, 0.65, 0.80]:
        for g in [0.3, 0.8, 1.5]:
            bb, dd, _, _ = run_combos([{**base, "beta0": b0, "gamma": g, "K_AI": 0}])
            threshold_rows.append(
                dict(beta0=b0, gamma=g, Kbar=float(dd["Pi"][0] - bb["Pi"][0]))
            )
    save_csv("Kbar_table.csv", threshold_rows)
    rows = []
    sweeps = {
        "c0": (0, 0.20, 9),
        "c1": (0, 0.10, 9),
        "e": (1, 2, 9),
        "r": (0, 0.50, 11),
        "q0": (0.70, 0.95, 9),
        "nu": (0, 0.50, 9),
        "C0": (0.2, 3, 10),
        "lam0": (0, 0.20, 9),
        "delta0": (0, 2, 9),
    }
    for key, (lo, hi, count) in sweeps.items():
        values = np.linspace(lo, hi, count)
        # Keep each sweep batched: the original stopping test spans the batch.
        bb, dd, delta, _ = run_combos([{**base, key: float(value)} for value in values])
        for i, value in enumerate(values):
            rows.append(
                dict(
                    param=key,
                    value=float(value),
                    dPi=float(delta[i]),
                    PiB=float(dd["Pi"][i]),
                    PiA=float(bb["Pi"][i]),
                )
            )
    save_csv("robustness.csv", rows)


DEST = OUT / "figures"

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.cal": "Times New Roman",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "axes.unicode_minus": False,
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "figure.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": "#5a5a5a",
        "axes.linewidth": 0.8,
        "grid.color": "#d8d8d8",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.8,
    }
)

BLUE = "#3f669f"
RED = "#ad625e"
GREEN = "#718d79"
DARK = "#282828"
POS = "#7c9985"
NEG = "#b16a66"
CMAP = LinearSegmentedColormap.from_list("profit", ["#dce7df", "#f7f7f7", "#8ea4bf"])


def read_csv(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finish(fig, name):
    fig.savefig(DEST / name, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def typeset_fixed_cost(ax, which="x"):
    label = "AI fixed cost, " + r"$K_{\mathrm{AI}}$"
    if which == "x":
        ax.set_xlabel(label, labelpad=10)
    else:
        ax.set_ylabel(label, labelpad=10)


def style_axis(ax, grid=True):
    if grid:
        ax.grid(True, zorder=0)
    ax.tick_params(direction="out", length=4, width=0.8, color="#555555")


def figure1():
    """Plot the full-grid distribution of the profit differential (Figure 1)."""
    rows = read_csv("core_grid.csv")
    values = np.asarray([float(row["dPi"]) for row in rows], dtype=float)
    positive = np.mean(values > 1e-6)
    negative = np.mean(values < -1e-6)

    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    ax.hist(
        values,
        bins=36,
        color="#5d82a8",
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.axvline(0.0, color=RED, linewidth=1.2, zorder=3)
    ax.set_title(
        "Figure 1. Distribution of profit differences over the full grid",
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel(r"Profit difference $\Delta\Pi$", labelpad=10)
    ax.set_ylabel("Number of parameter combinations", labelpad=10)
    limit = max(abs(values.min()), abs(values.max()))
    ax.set_xlim(-limit, limit)
    ax.set_xticks(np.linspace(-0.084, 0.084, 6))
    ax.text(
        0.97,
        0.89,
        f"Profit increase: {positive:.2%}\nProfit decrease: {negative:.2%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox=dict(facecolor="#f3f6f9", edgecolor="#aebdcc", pad=9),
    )
    style_axis(ax)
    finish(fig, "fig1_profit_difference_distribution.png")


def figure2():
    rows = read_csv("Kbar_table.csv")
    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    specs = [(0.3, BLUE), (0.8, RED), (1.5, GREEN)]
    for gamma, color in specs:
        part = sorted(
            (r for r in rows if float(r["gamma"]) == gamma),
            key=lambda r: float(r["beta0"]),
        )
        x = [float(r["beta0"]) for r in part]
        y = [float(r["Kbar"]) for r in part]
        ax.plot(
            x,
            y,
            marker="o",
            lw=2.35,
            ms=6.8,
            color=color,
            label=rf"$\gamma={gamma:g}$",
            zorder=3,
        )
    ax.axhline(0, color=DARK, lw=1.0, zorder=2)
    ax.set_title(
        "Figure 2. Critical AI fixed cost by trust and waiting sensitivity",
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Trust in AI-only service, " + r"$\beta_0$", labelpad=10)
    ax.set_ylabel("Critical AI fixed cost, " + r"$\bar K_{\mathrm{AI}}$", labelpad=10)
    ax.set_xlim(0.33, 0.82)
    ax.set_ylim(-0.02, 0.09)
    ax.set_xticks([0.33, 0.43, 0.53, 0.62, 0.72, 0.82])
    ax.set_yticks([-0.02, 0.00, 0.02, 0.05, 0.07, 0.09])
    ax.legend(frameon=False, loc="upper left", handlelength=2.4, borderaxespad=1.0)
    style_axis(ax)
    finish(fig, "fig2_kbar_trust_waiting.png")


def read_phase_grid(name):
    rows = read_csv(name)
    x = np.array(sorted({float(r["x"]) for r in rows}))
    y = np.array(sorted({float(r["y"]) for r in rows}))
    z = np.full((len(y), len(x)), np.nan)
    ix, iy = {v: i for i, v in enumerate(x)}, {v: i for i, v in enumerate(y)}
    for r in rows:
        z[iy[float(r["y"])], ix[float(r["x"])]] = float(r["dPi"])
    return x, y, z


def phase_panel(ax, x, y, z, title, y_label=None):
    limit = float(np.nanmax(np.abs(z)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    ax.pcolormesh(x, y, z, shading="auto", cmap=CMAP, norm=norm, rasterized=True)
    ax.contour(x, y, z, levels=[0], colors=DARK, linewidths=1.6)
    ax.set_title(title, pad=10, fontweight="bold")
    typeset_fixed_cost(ax)
    ax.set_ylabel(y_label if y_label else "")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xticks(np.linspace(x.min(), x.max(), 5))
    ax.set_yticks(np.linspace(y.min(), y.max(), 5))
    style_axis(ax, grid=True)


def figure3():
    x, y, z1 = read_phase_grid("phase_beta0_gamma03.csv")
    _, _, z2 = read_phase_grid("phase_beta0_gamma15.csv")
    fig, axes = plt.subplots(
        1, 2, figsize=(13.2, 6.6), sharey=True, constrained_layout=True
    )
    phase_panel(axes[0], x, y, z1, r"Low waiting sensitivity ($\gamma=0.3$)", "Trust in AI-only service, " + r"$\beta_0$")
    phase_panel(axes[1], x, y, z2, r"High waiting sensitivity ($\gamma=1.5$)")
    fig.suptitle(
        "Figure 3. Profitability regions: AI trust and fixed cost\n"
        + r"Black line: break-even boundary ($\Delta\Pi=0$)",
        fontweight="bold",
    )
    finish(fig, "fig3_phase_trust_cost.png")


def figure4():
    x, y, z1 = read_phase_grid("phase_r.csv")
    efficiency_x, efficiency_y, z2 = read_phase_grid("phase_e.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), constrained_layout=True)
    phase_panel(axes[0], x, y, z1, "Escalation referral rate and fixed cost", "Escalation referral rate, " + r"$r$")
    phase_panel(axes[1], efficiency_x, efficiency_y, z2, "Physician efficiency and fixed cost", "Physician efficiency multiplier, " + r"$e$")
    fig.suptitle(
        "Figure 4. Profitability regions: referral and physician efficiency\n"
        + r"Black line: break-even boundary ($\Delta\Pi=0$)",
        fontweight="bold",
    )
    finish(fig, "fig4_phase_referral_efficiency.png")


def figure5():
    values_by_key = {
        r["component"]: float(r["value"]) for r in read_csv("mechanism.csv")
    }
    keys = [
        "AI_non_escalated_net",
        "escalated_case_net",
        "direct_AI_doctor_net",
        "baseline_profit_removed",
        "subsidy_change",
        "AI_fixed_cost",
    ]
    labels = [
        "Non-escalated\nAI service",
        "Escalated\ncases",
        "Direct AI +\nphysician service",
        "Baseline profit\nremoved",
        "Subsidy\nchange",
        "AI fixed\ncost",
    ]
    values = [values_by_key[k] for k in keys]
    fig, ax = plt.subplots(figsize=(11.0, 6.4), constrained_layout=True)
    colors = [POS if val >= 0 else NEG for val in values]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.58, zorder=3)
    ax.axhline(0, color=DARK, lw=1.0, zorder=2)
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        shift = 0.001 if value >= 0 else -0.0014
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + shift,
            f"{value:.3f}",
            ha="center",
            va=va,
            fontsize=12,
        )
    ax.set_title("Figure 5. Profit decomposition at the reference parameters", pad=15, fontweight="bold")
    ax.set_ylabel(r"Contribution to $\Delta\Pi$")
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylim(-0.05, 0.04)
    style_axis(ax)
    finish(fig, "fig5_mechanism.png")


def figure6():
    data = {
        key: sorted(
            (float(r["value"]), float(r["dPi"]))
            for r in read_csv("robustness.csv")
            if r["param"] == key
        )
        for key in ["c0", "r", "e", "lam0"]
    }
    panels = [
        ("c0", "Initial AI processing cost", r"$c_0$"),
        ("r", "Escalation referral rate", r"$r$"),
        ("e", "Physician efficiency multiplier", r"$e$"),
        ("lam0", "Quality loss in non-escalated AI cases", r"$\lambda_0$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), constrained_layout=True)
    for i, (ax, (key, title, xlabel)) in enumerate(zip(axes.flat, panels)):
        vals = data[key]
        x = np.array([row[0] for row in vals])
        y = np.array([row[1] for row in vals])
        ax.plot(x, y, color=BLUE, lw=2.15, marker="o", ms=5.6, zorder=3)
        ax.axhline(0, color=DARK, lw=1.0, zorder=2)
        ax.set_title(title, pad=9, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta\Pi$" if i % 2 == 0 else "")
        ax.set_xlim(x.min(), x.max())
        lo, hi = y.min(), y.max()
        margin = max((hi - lo) * 0.13, 0.002)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xticks(np.linspace(x.min(), x.max(), 4))
        style_axis(ax)
    fig.suptitle("Figure 6. One-way robustness checks for key technical parameters", fontweight="bold")
    finish(fig, "fig6_robustness.png")


def run_figures():
    """Render all paper figures from the generated experiment tables."""
    # Fail explicitly rather than silently substituting a publication font.
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    DEST.mkdir(parents=True, exist_ok=True)
    for render in (figure1, figure2, figure3, figure4, figure5, figure6):
        render()


def verify():
    """Require every reference CSV/JSON checksum to match the generated file."""
    failures = []
    checked = 0
    for line in (EXPECTED / "CHECKSUMS.txt").read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, expected = [part.strip() for part in line.split(":", 1)]
        if not name.endswith((".csv", ".json")):
            continue
        checked += 1
        path = OUT / name
        actual = hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            failures.append(name)
    if not checked:
        raise ValueError("The reference checksum manifest contains no result files.")
    if failures:
        print("Verification failed:", ", ".join(failures))
        return False
    print("Verification passed: generated data match expected_outputs/CHECKSUMS.txt")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("core", "analysis", "figures", "verify", "all", "reproduce")
    )
    command = parser.parse_args(argv).command
    np.random.seed(SEED)
    if command in ("core", "all", "reproduce"):
        core_grid()
    if command in ("analysis", "all", "reproduce"):
        analysis()
    if command in ("figures", "all", "reproduce"):
        run_figures()
    if command in ("verify", "all", "reproduce"):
        return 0 if verify() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
