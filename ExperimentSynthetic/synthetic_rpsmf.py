#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Robust Probabilistic Sequential Matrix Factorization

"""

import argparse
import sys
import time

import autograd.numpy as np
from data import generate_t_data
from psmf import rPSMFIter
from psmf.tracking import TrackingMixin


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose mode", action="store_true"
    )
    parser.add_argument(
        "-p", "--live-plot", help="Enable live plotting", action="store_true"
    )
    parser.add_argument("-s", "--seed", help="Random seed to use", type=int)
    parser.add_argument(
        "--output-bases",
        help="Output file for bases figure",
        default="robust_bases.pdf",
    )
    parser.add_argument(
        "--output-cost-y",
        help="Output file for the cost (y) figure",
        default="robust_cost_y.pdf",
    )
    parser.add_argument(
        "--output-cost-theta",
        help="Output file for the cost (theta) figure",
        default="robust_cost_theta.pdf",
    )
    parser.add_argument(
        "--output-fit",
        help="Output file for the fit figure",
        default="robust_fit.pdf",
    )
    return parser.parse_args()


class rPSMFIterSynthetic(TrackingMixin, rPSMFIter):
    def run(
        self,
        y,
        y_obs,
        theta_true,
        C_true,
        x_true,
        T,
        n_iter,
        n_pred,
        adam_gam=1e-3,
        live_plot=False,
        verbose=True,
    ):
        self.adam_init(gam=adam_gam)
        self.errors_init(y_obs, T, n_iter, n_pred, theta_true=theta_true)
        self.figures_init(live_plot=live_plot)
        self.log(0, n_iter, 0, verbose=verbose)
        for i in range(1, n_iter + 1):
            t_start = time.time()
            self.step(y, i, T)
            self.predict(i, T, n_pred)
            self.adam_update(i)
            self.errors_update(i, y_obs, T, n_pred, theta_true=theta_true)
            self.log(i, n_iter, time.time() - t_start, verbose=verbose)
            self.figures_update(y_obs, T, n_pred, live_plot=live_plot, x_true=x_true)

    ### Algorithmic simplifications

    def step_reset(self):
        super().step_reset()
        # We re-initialize V at every iteration for this experiment.
        self._V = {0: self.V0}

    def _predictive_covariance(self, i, k):
        return self._P[k - 1]

    def _compute_eta_k(self, k, P_bar):
        return np.trace(self._R[k - 1]) / self._d

    def _compute_inverse_coefficient_innovation(self, k, mu_bar, P_bar):
        # P_bar is assumed to be zero
        Rbar = self._R[k - 1] + np.kron(
            mu_bar.T @ self._V[k - 1] @ mu_bar, np.eye(self._d)
        )
        return np.linalg.inv(Rbar)

    def _update_coefficient_mean(self, k, yk, Skinv, mu_bar, P_bar):
        self._mu[k] = mu_bar

    def _update_coefficient_covariance(self, k, Skinv, P_bar, yk):
        # P_bar is assumed to be zero
        omega_k = (
            self._lambda[k - 1]
            + (yk - self._y_pred[k]).T @ Skinv @ (yk - self._y_pred[k])
        ) / (self._lambda[k - 1] + self._d)
        self._P[k] = P_bar

        # Q is assumed 0, only R needs updating
        self._Q[k] = self._Q[k - 1]
        self._R[k] = omega_k * self._R[k - 1]
        if not self.fixed_lambda:
            self._lambda[k] = self._lambda[k - 1] + self._d

    def _prune(self, k):
        del self._C[k - 1], self._V[k - 1], self._P[k - 1]

    ### End algorithmic simplifications


B_POINTS = 1800
NUMS = 4


def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# Define the combined nonlinearity function
def nonlinearity(theta, x, t):

    # Simulate one ECG beat using Gaussian functions for PQRST waves
    def simulate_ecg_beat(t, offset):
        ecg_beat = np.zeros_like(t)

        # Define parameters for each wave (mu: center, sigma: width, amplitude)
        waves = [
            {"mu": 0.2, "sigma": 0.025, "amplitude": -0.1},  # P wave
            {"mu": 0.3, "sigma": 0.01, "amplitude": 0.15},  # Q wave
            {"mu": 0.35, "sigma": 0.01, "amplitude": -0.5},  # R wave
            {"mu": 0.4, "sigma": 0.01, "amplitude": 0.2},  # S wave
            {"mu": 0.7, "sigma": 0.05, "amplitude": -0.2},  # T wave
        ]

        for wave in waves:
            ecg_beat += wave["amplitude"] * gaussian(
                t, wave["mu"] + offset, wave["sigma"]
            )

        return ecg_beat

    # Simulate the ECG beat with Gaussian components
    ecg_beat = simulate_ecg_beat(
        (t % (B_POINTS // NUMS)) / (B_POINTS // NUMS), 0
    )  # Modulo to repeat the beat periodically
    # Introduce periodic behavior with the cosine function
    periodic_component = np.cos(2 * np.pi * theta * (t / (B_POINTS // NUMS)) + x)

    return ecg_beat * periodic_component


# def nonlinearity(theta, x, t):
#     return np.cos(2 * np.pi * theta * t + x)


def main():
    args = parse_args()
    seed = args.seed or np.random.randint(10000)
    print("Using seed: %r" % seed)
    np.random.seed(seed)

    d = 12
    r = 6
    T = 1500
    n_pred = 300
    n_iter = 500
    var = 0.0001
    dof = 3.0

    data = generate_t_data(nonlinearity, d=d, T=T, n_pred=n_pred, r=r, var=var, dof=dof)

    C0 = 0.1 * np.random.randn(d, r)
    theta0 = 0.1 * np.random.rand(r, 1)
    v0 = 0.1
    V0 = np.kron(v0, np.eye(r))
    mu0 = np.zeros([r, 1])
    P0 = np.zeros([r, r])
    Q0 = 0 * np.identity(r)
    R0 = np.identity(d)

    lambda0 = 1.8

    rpsmf = rPSMFIterSynthetic(theta0, C0, V0, mu0, P0, Q0, R0, lambda0, nonlinearity)
    rpsmf.run(
        data["y_train"],
        data["y_obs"],
        data["theta_true"],
        data["C_true"],
        data["x_true"],
        T,
        n_iter,
        n_pred,
        adam_gam=1e-3,
        live_plot=args.live_plot,
        verbose=args.verbose,
    )
    output_files = dict(
        fit=args.output_fit,
        bases=args.output_bases,
        cost_y=args.output_cost_y,
        cost_theta=args.output_cost_theta,
    )
    rpsmf.figures_save(
        data["y_obs"],
        n_pred,
        T,
        x_true=data["x_true"],
        output_files=output_files,
    )
    rpsmf.figures_close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(1)
