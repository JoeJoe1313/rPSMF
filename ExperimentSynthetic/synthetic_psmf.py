#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import time

import autograd.numpy as np
from data import generate_normal_data
from psmf import PSMFIter
from psmf.tracking import TrackingMixin

B_POINTS = 1800
NUMS = 4


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
        default="bases.pdf",
    )
    parser.add_argument(
        "--output-cost-y",
        help="Output file for the cost (y) figure",
        default="cost_y.pdf",
    )
    parser.add_argument(
        "--output-cost-theta",
        help="Output file for the cost (theta) figure",
        default="cost_theta.pdf",
    )
    parser.add_argument(
        "--output-fit",
        help="Output file for the fit figure",
        default="fit.pdf",
    )
    return parser.parse_args()


class PSMFIterSynthetic(TrackingMixin, PSMFIter):
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
        # P is assumed 0
        pass

    def _update_coefficient_mean(self, k, yk, Skinv, mu_bar, P_bar):
        self._mu[k] = mu_bar

    def _update_coefficient_covariance(self, k, Skinv, P_bar, yk):
        self._P[k] = P_bar

    def _prune(self, k):
        del self._C[k - 1], self._V[k - 1], self._P[k - 1]

    ### End algorithmic simplifications


def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# def nonlinearity(theta, x, t):
#     def simulate_ecg_beat(t, offset):
#         ecg_beat = np.zeros_like(t)
#         waves = [
#             {"mu": 0.2, "sigma": 0.025, "amplitude": -0.1},  # P wave
#             {"mu": 0.3, "sigma": 0.01, "amplitude": 0.15},  # Q wave
#             {"mu": 0.35, "sigma": 0.01, "amplitude": -0.5},  # R wave
#             {"mu": 0.4, "sigma": 0.01, "amplitude": 0.2},  # S wave
#             {"mu": 0.7, "sigma": 0.05, "amplitude": -0.2},  # T wave
#         ]
#         for wave in waves:
#             ecg_beat += wave["amplitude"] * gaussian(
#                 t, wave["mu"] + offset, wave["sigma"]
#             )

#         return ecg_beat

#     ecg_beat = simulate_ecg_beat(
#         (t % (B_POINTS // NUMS)) / (B_POINTS // NUMS), 0
#     )  # Modulo to repeat the beat periodically
#     # Introduce periodic behavior with the cosine function
#     periodic_component = np.cos(2 * np.pi * theta * (t / (B_POINTS // NUMS)) + x)

#     return ecg_beat * periodic_component


###


def nonlinearity(theta, x, t):
    def simulate_ecg_beat(t, offset):
        ecg_beat = np.zeros_like(t)
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

    # Generate a single ECG beat
    single_beat_duration = 1.0  # duration of a single beat in time units
    ecg_beat = simulate_ecg_beat(
        t % single_beat_duration, 0
    )  # t modulo single_beat_duration

    # Introduce periodic behavior with the cosine function
    periodic_component = np.cos(2 * np.pi * theta * t + x)

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

    data = generate_normal_data(nonlinearity, d=d, T=T, n_pred=n_pred, r=r, var=var)

    C0 = 0.1 * np.random.randn(d, r)
    theta0 = 0.1 * np.random.rand(r, 1)
    v0 = 0.1
    V0 = np.kron(v0, np.eye(r))
    mu0 = np.zeros([r, 1])
    P0 = np.zeros([r, r])
    Qs = {k: 0 * np.identity(r) for k in range(T + 1)}
    Rs = {k: np.identity(d) for k in range(T + 1)}

    psmf = PSMFIterSynthetic(theta0, C0, V0, mu0, P0, Qs, Rs, nonlinearity)
    psmf.run(
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
    psmf.figures_save(
        data["y_obs"],
        n_pred,
        T,
        x_true=data["x_true"],
        output_files=output_files,
    )
    psmf.figures_close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(1)
