import os

import oqupy
import numpy as np
import pickle

from oqupy import TempoParameters

import concurrent.futures

from utils import NonMarkovianity

zero = oqupy.operators.spin_dm("z+")
one = oqupy.operators.spin_dm("z-")


def tempo_func(system_parameters, tempo_parameters):
	omega_0, big_omega, omega_cutoff, alpha, s, temperature = system_parameters.values()
	start_time, end_time, time_steps, dt, tcut, epsrel = tempo_parameters.values()
	
	system = oqupy.System(omega_0 * oqupy.operators.sigma('z') + big_omega * oqupy.operators.sigma('x'))
	correlations = oqupy.PowerLawSD(alpha=np.pi*alpha/8, zeta=s, cutoff=omega_cutoff, temperature=temperature)
	bath = oqupy.Bath(oqupy.operators.sigma('z'), correlations)
	
	# parameters = oqupy.guess_tempo_parameters(system=system,
	#                                           bath=bath,
	#                                           start_time=start_time,
	#                                           end_time=end_time,
	#                                           tolerance=0.01)
	# print(parameters)
	# quit()
	
	process_tensor = oqupy.pt_tempo_compute(bath=bath,
	                                        start_time=start_time,
	                                        end_time=end_time,
	                                        parameters=TempoParameters(dt=dt, tcut=tcut, epsrel=epsrel))
	
	rho_1_dynamics = oqupy.compute_dynamics(system, np.array([[1, 0], [0, 0]]), process_tensor=process_tensor)
	rho_2_dynamics = oqupy.compute_dynamics(system, np.array([[0, 1], [0, 0]]), process_tensor=process_tensor)
	rho_3_dynamics = oqupy.compute_dynamics(system, np.array([[0, 0], [1, 0]]), process_tensor=process_tensor)
	rho_4_dynamics = oqupy.compute_dynamics(system, np.array([[0, 0], [0, 1]]), process_tensor=process_tensor)
	
	data = dict(
		system_parameters=system_parameters,
		tempo_parameters=tempo_parameters,
		system=system,
		process_tensor=process_tensor,
		tlist=np.linspace(start_time, end_time, time_steps + 1),
		rho_1_list=rho_1_dynamics.states,
		rho_2_list=rho_2_dynamics.states,
		rho_3_list=rho_3_dynamics.states,
		rho_4_list=rho_4_dynamics.states
	)
	
	return data


def run_single_temperature(temperature, omega_0, big_omega, omega_cutoff, alpha, s, start_time, end_time, time_steps,
                           dt, tcut, epsrel, folder):
	print(f'Temperature: {temperature}')
	system_parameters = dict(
		omega_0=omega_0,
		big_omega=big_omega,
		omega_cutoff=omega_cutoff,
		alpha=alpha,
		s=s,
		temperature=temperature
	)
	tempo_parameters = dict(start_time=start_time, end_time=end_time, time_steps=time_steps, dt=dt, tcut=tcut,
	                        epsrel=epsrel)
	file = f"{folder}/temperature_{temperature:.2f}.pkl"
	
	data = tempo_func(system_parameters, tempo_parameters)
	nm = NonMarkovianity(data)
	dynamical_map = nm.get_dynamical_map()
	BLP = nm.get_BLP()
	LPP = nm.get_LPP()
	RHP = nm.get_RHP()
	
	data.update(dynamical_map=dynamical_map, BLP=BLP, LPP=LPP, RHP=RHP)
	with open(file, 'wb') as f:
		pickle.dump(data, f)


if __name__ == '__main__':
	omega_0 = 1.
	big_omega = 0.
	omega_cutoff = 10.
	alpha = 1.
	s = 1.
	temperatures = range(11)
	start_time = 0.
	end_time = 3 * np.pi
	time_steps = 150
	dt = (end_time - start_time) / time_steps
	tcut = end_time / 4
	epsrel = 1e-5
	
	folder = f'omega_0_{omega_0:.2f}_big_omega_{big_omega:.2f}_omega_cutoff_{omega_cutoff:.2f}_alpha_{alpha:.2f}_s_{s:.2f}'
	os.makedirs(folder, exist_ok=True)
	
	with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
		futures = [
			executor.submit(
				run_single_temperature, temperature, omega_0, big_omega, omega_cutoff, alpha, s,
				start_time, end_time, time_steps, dt, tcut, epsrel, folder
			) for temperature in temperatures
		]
		
		for future in concurrent.futures.as_completed(futures):
			future.result()
