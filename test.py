import os

for i in range(11):
	os.rename(f'omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00/omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00_temperature_{i:.2f}.pkl', f'omega_0_0.00_big_omega_1.00_omega_cutoff_10.00_alpha_0.10_s_1.00/temperature_{i:.2f}.pkl')