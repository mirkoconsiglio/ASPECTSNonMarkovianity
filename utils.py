import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.linalg import pinv


class NonMarkovianity:
	def __init__(self, data=None, dynamical_map=None):
		self.data = data
		self.dynamical_map = dynamical_map
		self.rho_1_list = self.data.get('rho_1_list')
		self.rho_2_list = self.data.get('rho_2_list')
		self.rho_3_list = self.data.get('rho_3_list')
		self.rho_4_list = self.data.get('rho_4_list')
		self.tlist = self.data.get('tlist')
		self.tstep = self.tlist[1] - self.tlist[0]
		
		self.I = np.identity(2)
		self.X = np.array([[0, 1], [1, 0]])
		self.Y = np.array([[0, -1j], [1j, 0]])
		self.Z = np.array([[1, 0], [0, -1]])
	
	def get_dynamical_map(self):
		E_t_list = [self.I, self.X, -1j * self.Y, self.Z]
		
		lmbda = np.block([[self.I, self.X], [self.X, -self.I]]) / 2
		
		chi_list = []
		for rho_1, rho_2, rho_3, rho_4 in zip(self.rho_1_list, self.rho_2_list, self.rho_3_list, self.rho_4_list):
			chi_list.append(lmbda @ np.block([[rho_1, rho_2], [rho_3, rho_4]]) @ lmbda)
		chi_list = np.array(chi_list)
		
		A_list = []
		for chi in chi_list:
			E_list = []
			d, U = np.linalg.eigh(chi)
			d[d < 0] = 0
			for i in range(len(E_t_list)):
				E_list.append(np.sqrt(d[i]) * sum(U[j, i].conj() * E_t_list[j] for j in range(len(E_t_list))))
			A_list.append(sum(np.kron(E, E.conj()) for E in E_list))
		
		self.dynamical_map = A_list
		
		return self.dynamical_map
	
	@staticmethod
	def trace_distance(state1, state2):
		return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(state1 - state2)))
	
	@staticmethod
	def vectorize(A):
		a = A.copy().T.flatten()
		a[[2, 4]] = a[[4, 2]]
		a[[3, 5]] = a[[5, 3]]
		a[[10, 12]] = a[[12, 10]]
		a[[11, 13]] = a[[13, 11]]
		
		return a
	
	@staticmethod
	def devectorize(A):
		a = A.copy()
		
		a[[2, 4]] = a[[4, 2]]
		a[[3, 5]] = a[[5, 3]]
		a[[10, 12]] = a[[12, 10]]
		a[[11, 13]] = a[[13, 11]]
		
		size = int(np.sqrt(len(a)))
		return a.reshape(size, size)
	
	def get_BLP(self):
		td_list = [self.trace_distance(s1, s2) for s1, s2 in zip(self.rho_1_list, self.rho_4_list)]
		dtd_list = np.gradient(td_list, self.tstep, edge_order=2)
		return cumulative_simpson(np.clip(dtd_list, a_min=0, a_max=np.inf), dx=self.tstep)[-1]
	
	def get_LPP(self):
		pauli_list = np.array([self.X, self.Y, self.Z])
		
		F_list = []
		for t in range(len(self.tlist)):
			F = np.zeros((3, 3), dtype=np.complex128)
			for i, a in enumerate(pauli_list):
				for j, b in enumerate(pauli_list):
					F[i, j] = np.trace(a @ np.reshape(self.dynamical_map[t] @ b.flatten(), (2, 2))) / 2
			F_list.append(F)
		
		F_det_list = [abs(np.linalg.det(F)) for F in F_list]
		F_det_grad_list = np.gradient(F_det_list, self.tstep, edge_order=2)
		return cumulative_simpson(np.clip(F_det_grad_list, a_min=0, a_max=np.inf), dx=self.tstep)[-1]
	
	def get_RHP(self):
		bell = 0.5 * np.array([[1, 0, 0, 1],
		                       [0, 0, 0, 0],
		                       [0, 0, 0, 0],
		                       [1, 0, 0, 1]])
		iden = np.identity(4)
		
		glist = []
		for t in range(len(self.tlist) - 1):
			A = self.dynamical_map[t + 1] @ pinv(self.dynamical_map[t])
			f = np.sum(np.linalg.svd(self.devectorize(np.kron(A, iden) @ self.vectorize(bell)), compute_uv=False))
			
			g = (f - 1) / self.tstep
			glist.append(g)
		
		return cumulative_simpson(np.clip(glist, a_min=0, a_max=np.inf), dx=self.tstep)[-1]
