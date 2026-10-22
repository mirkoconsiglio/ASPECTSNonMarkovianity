"""Master-equation cross-check of BLP and LPP only (no direct T-scan of the
other measures). Prints the numbers for the Figs. 3 and 5 captions.

    python mesolve_gaps.py --E 2.0 --Gamma 0.5 --tmax 320 --nt 32001 --T 0.01 0.1 1.0 10.0
"""
import argparse
import time
import warnings

import numpy as np

from non_markovianity import ModelConfig, NonMarkovianModel

warnings.filterwarnings("ignore")

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--E", type=float, default=2.0)
p.add_argument("--Gamma", type=float, default=0.5)
p.add_argument("--Omega", type=float, default=1.0)
p.add_argument("--mu", type=float, default=0.0)
p.add_argument("--tmax", type=float, default=320.0)
p.add_argument("--nt", type=int, default=32001)
p.add_argument("--T", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
a = p.parse_args()

T = np.array(a.T, dtype=float)
t0 = time.time()
m = NonMarkovianModel(ModelConfig(E=a.E, Gamma=a.Gamma, Omega=a.Omega, mu=a.mu,
                                  tmax=a.tmax, nt=a.nt, T=T), verbose=False)
blp_d, lpp_d = m.blp_direct(), m.lpp_direct()
print(f"direct:  BLP = {blp_d:.10f}   LPP = {lpp_d:.10f}   ({time.time() - t0:.0f}s)",
      flush=True)

r = m.run(["BLP", "LPP"], direct=False)  # same route as --mesolve-check
print(f"master equation done ({time.time() - t0:.0f}s)")
for i, Ti in enumerate(T):
    print(f"  T={Ti:6.2f}:  BLP_ODE = {r['BLP'][i]:.10f} (diff {r['BLP'][i] - blp_d:+.3e})"
          f"   LPP_ODE = {r['LPP'][i]:.10f} (diff {r['LPP'][i] - lpp_d:+.3e})")

for k, d in (("BLP", blp_d), ("LPP", lpp_d)):
    v = np.asarray(r[k], dtype=float)
    print(f"{k}: max |direct - ODE| = {np.max(np.abs(v - d)):.2e}   "
          f"spread across T = {np.ptp(v):.2e}")
