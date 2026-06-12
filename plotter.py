import pickle
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'Times New Roman', 'sans-serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)

_colour_map = cc.cm.CET_C8s
_colour_list = np.transpose(list(_colour_map._segmentdata.values()))[1]


def plot(params):
	omega_0, big_omega, omega_cutoff, alpha_list, s, temperature_list = params.values()
	
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.grid(visible=True, which='both', axis='both')
	ax.set_ylabel(r'$\mathcal{N}$')
	ax.set_xlabel(r'$T$')
	ax.set_yscale('symlog', linthresh=10e-6)
	
	for i, alpha in enumerate(alpha_list):
		folder = f'omega_0_{omega_0:.2f}_big_omega_{big_omega:.2f}_omega_cutoff_{omega_cutoff:.2f}_alpha_{alpha:.2f}_s_{s:.2f}'
		BLP_list = []
		LPP_list = []
		RHP_list = []
		for temperature in temperature_list:
			file = f'{folder}/temperature_{temperature:.2f}.pkl'
		
			with open(file, 'rb') as f:
				data = pickle.load(f)
				
			BLP_list.append(data.get('BLP'))
			LPP_list.append(data.get('LPP'))
			RHP_list.append(data.get('RHP'))
		
		ax.plot(temperature_list, BLP_list, marker='o',
		        color=_colour_list[int(len(_colour_list) / (2 * len(alpha_list) + 1)) * i],
		        linestyle='-', label=fr'BLP $\alpha = {alpha:.1f}$')
		ax.plot(temperature_list, LPP_list, marker='x',
		        color=_colour_list[int(len(_colour_list) / (2 * len(alpha_list) + 1)) * i],
		        linestyle='--', label=fr'LPP $\alpha = {alpha:.1f}$')
	
	ax.legend()
	
	plt.savefig(f'plots/omega_0_{omega_0:.2f}_big_omega_{big_omega:.2f}_omega_cutoff_{omega_cutoff:.2f}_s_{s:.2f}.pdf', bbox_inches='tight')
	plt.show()
	

def plot_line(temperature_list, BLP, LPP):
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.grid(visible=True, which='both', axis='both')
	ax.set_ylabel(r'$\mathcal{N}$')
	ax.set_xlabel(r'$T$')
	
	for i, gamma in enumerate([0., 0.5, 1.]):
		for j, h in enumerate([0.5, 1., 1.5]):
			k = 3 * i + j
			ax.plot(temperature_list, np.full(len(temperature_list), BLP[i, j]), marker='o',
			        color=_colour_list[int(len(_colour_list) / 10) * k],
			        linestyle='-', label=fr'BLP $\gamma = {gamma}$, $h = {h}$')
			ax.plot(temperature_list, np.full(len(temperature_list), LPP[i, j]), marker='x',
			        color=_colour_list[int(len(_colour_list) / 10) * k],
			        linestyle='--', label=fr'LPP $\gamma = {gamma}$, $h = {h}$')
	ax.legend(bbox_to_anchor=(1.01, 0.98))
	fig.tight_layout()
	plt.savefig(f'plots/spin_chain.pdf', bbox_inches='tight')
	plt.show()
		

if __name__ == '__main__':
	BLP = np.array([
		[0.066239, 0.0712573, 0.0529786],
		[0.831285, 0.194755, 0.162993],
		[0.245724, 0.160994, 0.136953]
	])
	LPP = np.array([
		[0.0957617, 0.224936, 0.192738],
		[0.246916, 0., 0.],
		[0.331357, 0., 0.000557559]
	])
	plot_line(range(0, 11), BLP, LPP)
	quit()
	
	omega_0 = 1.
	big_omega = 0.
	omega_cutoff = 10.
	alpha_list = [0.1, 0.5, 1.]
	s = 1.
	temperature_list = range(0, 11)
	
	params = dict(
		omega_0=omega_0,
		big_omega=big_omega,
		omega_cutoff=omega_cutoff,
		alpha_list=alpha_list,
		s=s,
		temperature_list=temperature_list
	)
	
	plot(params)
	