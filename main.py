import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
import colorcet as cc
from matplotlib import rc
import json
import os

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)

_colour_list = np.transpose(list(cc.cm.CET_C8s._segmentdata.values()))[1]


class SingleParticleIsing:
	def __init__(self, h, J_0, N, J_list, t_list):
		self.h = h
		self.J_0 = J_0
		self.N = N
		self.J_list = J_list
		self.t_list = t_list
		self.hamiltonian = self.get_hamiltonian()
		self.f_list = self.transitions()
		self.f_matrix_list = self.get_f_matrix_list()
	
	def get_hamiltonian(self):
		row = np.array([0])
		col = np.array([0])
		data = np.array([self.h])
		for i in range(len(self.J_list)):
			for j in range(self.N - i - 1):
				row = np.append(row, [j, i + 1 + j])
				col = np.append(col, [i + 1 + j, j])
				if j > 0:
					data = np.append(data, [self.J_list[i], self.J_list[i]])
				else:
					data = np.append(data, [self.J_0 * self.J_list[i], self.J_0 * self.J_list[i]])
		
		return csr_matrix((data, (row, col)), shape=(self.N, self.N)).toarray()
	
	def f(self, t, u, v):
		return np.abs(np.sum([v[n] * np.exp(-1j * u[n] * t) for n in range(self.N)]))
	
	def transitions(self):
		u, v = eigh(self.hamiltonian)
		
		v = [np.abs(v[0, n]) ** 2 for n in range(self.N)]
		
		return np.array([self.f(t, u, v) for t in self.t_list])
	
	def get_f_matrix_list(self):
		return np.array([[
			[1, 0, 0, 1 - np.abs(f) ** 2],
			[0, f, 0, 0],
			[0, 0, f.conj(), 0],
			[0, 0, 0, np.abs(f) ** 2]
		] for f in self.f_list
		])
	
	def evolution_func(self, rho):
		_rho = rho.flatten()
		return np.array([np.reshape(f @ _rho, (2, 2)) for f in self.f_matrix_list])
	
	def get_LPP(self, t_list):
		t_step = t_list[1] - t_list[0]
		A_list = np.abs(self.f_list) ** 4
		dA_list = np.gradient(A_list, t_step, edge_order=2)
		return cumulative_simpson(np.clip(dA_list, a_min=0, a_max=np.inf), dx=t_step)[-1]


class NonMarkovianity:
	def __init__(self, evolution_func, t_list, state_dimension=2, measure='BLP'):
		self.state_dimension = state_dimension
		self.t_list = t_list
		self.t_step = t_list[1] - t_list[0]
		self.measure = self.get_measure(measure)
		self.evolution_func = evolution_func
		
		self.f = None
		self.iter = None
		self.nfev = None
	
	def get_measure(self, measure):
		if measure == 'BLP':
			if self.state_dimension == 2:
				return self.one_qubit_BLP
			return self.BLP
	
	def BLP(self, state1, state2):
		pass
	
	def one_qubit_BLP(self, x):
		self.nfev += 1
		
		theta, phi = x
		state1 = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
		state2 = np.array([np.sin(theta / 2), -np.exp(1j * phi) * np.cos(theta / 2)])
		
		rho = np.outer(state1.conj().T, state1)
		sigma = np.outer(state2.conj().T, state2)
		
		rho_list = self.evolution_func(rho)
		sigma_list = self.evolution_func(sigma)
		
		td_list = [self.trace_distance(s1, s2) for s1, s2 in zip(rho_list, sigma_list)]
		dtd_list = np.gradient(td_list, self.t_step, edge_order=2)
		
		self.f = cumulative_simpson(np.clip(dtd_list, a_min=0, a_max=np.inf), dx=self.t_step)[-1]
		
		return -self.f
	
	@staticmethod
	def trace_distance(state1, state2):
		return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(state1 - state2)))
	
	def minimize(self, x0=None, method='BFGS'):
		if x0 is None:
			x0 = np.random.uniform(0, 4 * np.pi, 2)
		
		self.iter = 0
		self.nfev = 0
		result = minimize(self.measure, x0=x0, method=method, callback=self.callback)
		
		return result
	
	def callback(self, *args):
		self.iter += 1
		print(self.iter, self.nfev, self.f)


if __name__ == '__main__':
	# Optimization parameters
	state_dimension = 2
	measure = 'BLP'
	
	# Model parameters
	N = 100
	neighbour_range = 1
	alpha = 1
	J_list = [1 / (2 * (n + 1) ** alpha) for n in range(neighbour_range)]
	t_list = np.linspace(0, 2 / 3 * N, 1000)
	
	fig_BLP, ax_BLP = plt.subplots(figsize=(12, 8))
	ax_BLP.set_xlabel(r'$h$')
	ax_BLP.set_ylabel(r'$\mathcal{N}_\textrm{BLP}$')

	fig_LPP, ax_LPP = plt.subplots(figsize=(12, 8))
	ax_LPP.set_xlabel(r'$h$')
	ax_LPP.set_ylabel(r'$\mathcal{N}_\textrm{LPP}$')
	
	J_0 = 1.0
	h_list = np.linspace(0, 2, 21)
	
	BLP_list = []
	LPP_list = []
	for h in h_list:
		print()
		print(f'h = {h:.2f}')
		print()
		
		model = SingleParticleIsing(h, J_0, N, J_list, t_list)
	
		# Run optimization
		nm = NonMarkovianity(model.evolution_func, t_list, state_dimension, measure)
		result = nm.minimize()
		BLP_list.append(nm.f)
		LPP_list.append(model.get_LPP(t_list))

		print()
	
	ax_BLP.plot(h_list, BLP_list, marker='o', label=fr'$J_0 = {J_0:.2f}$')
	ax_LPP.plot(h_list, LPP_list, marker='o', label=fr'$J_0 = {J_0:.2f}$')
	
	plt.show()
	
	# folder = 'omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00'
	# os.makedirs(folder, exist_ok=True)
	#
	# fig_BLP, ax_BLP = plt.subplots(figsize=(12, 8))
	# ax_BLP.set_xlabel(r'$h$')
	# ax_BLP.set_ylabel(r'$\mathcal{N}_\textrm{BLP}$')
	#
	# fig_LPP, ax_LPP = plt.subplots(figsize=(12, 8))
	# ax_LPP.set_xlabel(r'$h$')
	# ax_LPP.set_ylabel(r'$\mathcal{N}_\textrm{LPP}$')
	#
	# h_list = np.linspace(0, 4, 21)
	# J_0_list = np.linspace(0, 1, 11)
	# for i, J_0 in enumerate(J_0_list):
	# 	BLP = []
	# 	LPP = []
	# 	for h in h_list:
	# 		print()
	# 		print(f'J_0 = {J_0:.2f}', f'h = {h:.2f}')
	# 		print()
	#
	# 		model = SingleParticleIsing(h, J_0, N, J_list, t_list)
	#
	# 		# Run optimization
	# 		nm = NonMarkovianity(model.evolution_func, t_list, state_dimension, measure)
	# 		result = nm.minimize()
	# 		BLP.append([h, nm.f])
	# 		LPP.append([h, model.get_LPP(t_list)])
	#
	# 		print()
	#
	# 	file = f'{folder}/BLP_{neighbour_range}.json'
	# 	if os.path.isfile(file):
	# 		with open(file, 'r') as f:
	# 			omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00 = json.load(f)
	# 	else:
	# 		omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00 = {}
	# 	omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00.update({f'{J_0:.2f}': BLP})
	# 	with open(file, 'w') as f:
	# 		json.dump(omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00, f, indent=4)
	#
	# 	file = f'{folder}/LPP_{neighbour_range}.json'
	# 	if os.path.isfile(file):
	# 		with open(file, 'r') as f:
	# 			omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00 = json.load(f)
	# 	else:
	# 		omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00 = {}
	# 	omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00.update({f'{J_0:.2f}': LPP})
	# 	with open(file, 'w') as f:
	# 		json.dump(omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00, f, indent=4)
	#
	# 	ax_BLP.plot(h_list, BLP, marker='o', color=_colour_list[15 * i], label=fr'$J_0 = {J_0:.2f}$')
	# 	ax_LPP.plot(h_list, LPP, marker='o', color=_colour_list[15 * i], label=fr'$J_0 = {J_0:.2f}$')
	#
	# ax_BLP.legend()
	# ax_LPP.legend()
	#
	# plt.show()
