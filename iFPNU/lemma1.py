#!/usr/bin/env python3
"""Numerical verification of Lemma 1 for the iFPU focal loss.

Lemma 1 makes two claims about the focal loss written as a function of the raw
score z = g(x), with p = sigma(z):

    l_+(z) = -(1 - sigma(z))**gamma * log sigma(z)         (target y = +1)
    l_-(z) =  l_+(-z)                                       (target y = -1)

    (i)  Lipschitz:  |l_+'(z)| <= L_gamma = 1 + gamma/e   for all z.
    (ii) Bounded:    0 <= l_+(z) <= C_B = -sigma(B)**gamma * log sigma(-B)
                     whenever |z| <= B, with the max attained at z = -B.

This script checks both empirically (finite differences on a dense grid) and,
for the Lipschitz bound, also exactly via the closed-form derivative

    l_+'(z) = gamma * p * (1-p)**gamma * log p - (1-p)**(gamma+1),   p = sigma(z).

Note: (i) is a *conservative* upper bound; the true peak slope is smaller. A valid
upper bound is all Theorem 1 needs. The assumption |z| <= B in (ii) is load-bearing:
the loss is genuinely unbounded as z -> -inf without it.

Usage:
    python verify_lemma1.py
    python verify_lemma1.py --gammas 0 1 2 3 5 --B 5 --n 4000001
"""

from __future__ import annotations

import argparse

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + e^x), numerically stable."""
    return np.logaddexp(0.0, x)


def sigma(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, stable for the score ranges used here (|z| <= ~40)."""
    return 1.0 / (1.0 + np.exp(-z))


def l_plus(z: np.ndarray, gamma: float) -> np.ndarray:
    """Focal loss for target y = +1 as a function of the score z.

    Stable form: (1 - sigma(z))**gamma * (-log sigma(z))
               = sigma(-z)**gamma      *  softplus(-z).
    """
    return sigma(-z) ** gamma * softplus(-z)


def l_plus_prime(z: np.ndarray, gamma: float) -> np.ndarray:
    """Closed-form derivative d l_+ / d z (used for an exact Lipschitz check)."""
    p = sigma(z)
    log_p = -softplus(-z)  # = log sigma(z), stable
    return gamma * p * (1.0 - p) ** gamma * log_p - (1.0 - p) ** (gamma + 1.0)


def check_lipschitz(gamma: float, z_max: float, n: int, tol: float) -> dict:
    z = np.linspace(-z_max, z_max, n)
    dz = z[1] - z[0]

    emp_L_fd = float(np.max(np.abs(np.gradient(l_plus(z, gamma), dz))))  # finite diff
    emp_L_cf = float(np.max(np.abs(l_plus_prime(z, gamma))))            # closed form
    theo_L = 1.0 + gamma / np.e

    return {
        "gamma": gamma,
        "emp_L_finite_diff": emp_L_fd,
        "emp_L_closed_form": emp_L_cf,
        "L_gamma": theo_L,
        "holds": emp_L_cf <= theo_L + tol and emp_L_fd <= theo_L + tol,
    }


def check_boundedness(gamma: float, B: float, n: int) -> dict:
    z = np.linspace(-B, B, n)
    lp = l_plus(z, gamma)
    emp_max = float(lp.max())
    argmax_z = float(z[lp.argmax()])
    C_B = float(sigma(B) ** gamma * softplus(B))  # = -sigma(B)**gamma * log sigma(-B)

    return {
        "gamma": gamma,
        "B": B,
        "emp_max": emp_max,
        "argmax_z": argmax_z,
        "C_B": C_B,
        "nonneg": bool(lp.min() >= 0.0),
        "holds": np.isclose(emp_max, C_B, rtol=1e-4, atol=1e-6)
        and np.isclose(argmax_z, -B, atol=2 * (z[1] - z[0])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--B", type=float, default=5.0, help="score bound for (ii)")
    ap.add_argument("--z-max", type=float, default=40.0, help="grid half-width for (i)")
    ap.add_argument("--n", type=int, default=4_000_001, help="grid points")
    ap.add_argument("--tol", type=float, default=1e-3, help="finite-diff slack for (i)")
    args = ap.parse_args()

    all_ok = True

    print("Lemma 1(i)  Lipschitz:  |l_+'(z)| <= L_gamma = 1 + gamma/e")
    print(f"{'gamma':>6} {'emp|slope| (fd)':>16} {'emp|slope| (cf)':>16} "
          f"{'L_gamma':>10} {'holds':>6}")
    for g in args.gammas:
        r = check_lipschitz(g, args.z_max, args.n, args.tol)
        all_ok &= r["holds"]
        print(f"{r['gamma']:>6.1f} {r['emp_L_finite_diff']:>16.6f} "
              f"{r['emp_L_closed_form']:>16.6f} {r['L_gamma']:>10.6f} "
              f"{str(r['holds']):>6}")

    print("\nLemma 1(ii) Boundedness:  0 <= l_+(z) <= C_B on |z| <= B "
          "(max at z = -B)")
    print(f"{'gamma':>6} {'B':>5} {'emp max':>12} {'argmax z':>10} "
          f"{'C_B':>12} {'nonneg':>7} {'holds':>6}")
    for g in args.gammas:
        r = check_boundedness(g, args.B, max(args.n // 2, 200_001))
        all_ok &= r["holds"] and r["nonneg"]
        print(f"{r['gamma']:>6.1f} {r['B']:>5.1f} {r['emp_max']:>12.6f} "
              f"{r['argmax_z']:>10.4f} {r['C_B']:>12.6f} "
              f"{str(r['nonneg']):>7} {str(r['holds']):>6}")

    print(f"\nALL CHECKS PASSED: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())