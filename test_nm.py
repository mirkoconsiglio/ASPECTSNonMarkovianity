import numpy as np
import pytest

from non_markovianity import ModelConfig, NonMarkovianModel


def sum_rule_error(Gamma=1.0, Omega=1.0, E=0.0, tmax=20.0, nt=2001, n_omega=4097):
	"""max_t |Z_beta(t) - (1 - |G(t)|^2)| with the bath fully occupied."""
	cfg = ModelConfig(Gamma=Gamma, Omega=Omega, E=E, mu=100.0 * Omega,
	                  tmax=tmax, nt=nt, n_omega=n_omega, T=np.array([0.0]))
	model = NonMarkovianModel(cfg, verbose=False)
	model.compute_rates()
	target = 1.0 - np.abs(model.G_t) ** 2
	return float(np.max(np.abs(model.Z_Tt[0] - target)))


@pytest.mark.parametrize("E", [0.0, 0.4, 0.6, 2.0, -2.0])
def test_sum_rule_vs_E(E):
	nt, tmax = 2001, 20.0
	dt = tmax / (nt - 1)
	assert sum_rule_error(E=E, nt=nt, tmax=tmax) < 1.0 * dt ** 2


@pytest.mark.parametrize("Gamma", [0.2, 1.0, 2.5])
def test_sum_rule_vs_Gamma(Gamma):
	nt, tmax = 2001, 20.0
	dt = tmax / (nt - 1)
	assert sum_rule_error(Gamma=Gamma, E=2.0, nt=nt, tmax=tmax) < 4.0 * dt ** 2


@pytest.mark.parametrize("E", [0.0, 2.0])
def test_sum_rule_fourth_order(E):
	errs = [sum_rule_error(E=E, nt=n) for n in (501, 1001, 2001, 4001)]
	ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
	assert all(10.0 < r < 25.0 for r in ratios), (errs, ratios)


def test_sum_rule(Gamma=1.0, Omega=1.0, E=0.0, tmax=20.0, nt=2001, n_omega=4097,
                  atol=1e-6):
	"""Exact check of the whole Z pipeline against an analytic answer.

	Preserving {d(t), d^dag(t)} = 1 gives |G|^2 + sum_q |g_q|^2 = 1, i.e.
	    int dw J_N(w) |g(w,t)|^2 = 1 - |G(t)|^2 .
	Z_beta(t) is that integral weighted by f(w), so putting the chemical
	potential far above the band (f == 1 across the support of J_N) must return
	1 - |G(t)|^2 identically. This exercises the cumulative-Simpson Z path, the
	quadrature table and the bound-state poles in one shot.
	"""
	cfg = ModelConfig(Gamma=Gamma, Omega=Omega, E=E, mu=100.0 * Omega,
	                  tmax=tmax, nt=nt, n_omega=n_omega,
	                  T=np.array([0.0]))
	model = NonMarkovianModel(cfg, verbose=False)
	model.compute_rates()
	target = 1.0 - np.abs(model.G_t) ** 2
	err = float(np.max(np.abs(model.Z_Tt[0] - target)))
	print(f"[sum rule] E={E:g} Gamma={Gamma:g} Omega={Omega:g}: "
	      f"max |Z - (1 - |G|^2)| = {err:.3e}")
	assert err < atol, f"sum rule violated by {err:.3e}"
	return err


def plot_apollaro():
	"""Apollaro Chain: Ω = 2J, Γ = 2J·(J₀/J)² (so Γ=Ω ⇔ J₀=J, the Fig 1 line),
	E = 2(h−h₀), giving δh/J = E/Ω."""
	J = 1
	J_0 = 1
	Omega = 2 * J
	Gamma = 2 * J * (J_0 / J) ** 2
	h_list = np.linspace(0, 4., 101)
	blp_list = []
	for h in h_list:
		cfg = ModelConfig(E=2 * h, Omega=Omega, Gamma=Gamma, Tmax=1., nT=1, tmax=80., nt=1001, n_omega=2049)
		measures = ["BLP"]
		model = NonMarkovianModel(cfg)
		results = model.run(measures)
		blp_list.append(results["BLP"][0])
	
	import matplotlib.pyplot as plt
	
	plt.plot(h_list, blp_list, marker=".")
	plt.show()


def test_zdot_sum_rule(E=2.0, nt=2001):
	cfg = ModelConfig(E=E, mu=100.0, tmax=20.0, nt=nt, T=np.array([0.0]))
	m = NonMarkovianModel(cfg, verbose=False)
	m.compute_rates()
	target = m.gamma_t * np.abs(m.G_t) ** 2  # = -d|G|^2/dt
	assert np.max(np.abs(m.Z_dot_Tt[0] - target)) < 1e-8


def test_fast_matches_slow(E=2.0, nt=201):
	cfg = ModelConfig(E=E, mu=100.0, tmax=5.0, nt=nt, T=np.array([0.0]))
	errs = {}
	for fast in (True, False):
		m = NonMarkovianModel(cfg, verbose=False)
		m.compute_rates(fast=fast)
		errs[fast] = float(np.max(np.abs(m.Z_Tt[0] - (1 - np.abs(m.G_t) ** 2))))
	assert errs[True] < errs[False], errs  # single-integral route is better
