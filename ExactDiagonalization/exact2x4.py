import numpy as np
import netket as nk
import netket.experimental as nkx
import pandas as pd
import json
import matplotlib.pyplot as plt 
from exact_dynamics import ExactDynamics
from functools import reduce

g = nk.graph.Grid((2,4), pbc=True)
#g.draw()
nup, ndn = 4,4
hi = nk.hilbert.SpinOrbitalFermions(g.n_nodes, 1/2, n_fermions_per_spin=(nup, ndn))
hi_tmp = nk.hilbert.SpinOrbitalFermions(g.n_nodes, 1/2)

def cdag(site, *, sz):
    return nk.operator.fermion.create(hi_tmp, site, sz=sz)
    
# create the vaccuum
vac_idx = hi_tmp.states_to_numbers(np.zeros((hi.size,)))
vac = np.zeros((hi_tmp.n_states,), dtype=float)
vac[vac_idx] = 1

############ preparing initial state ############

def get_pair_op(i, j):
    """Returns the operator: (up_i dn_j + dn_i up_j)"""
    # Note: NetKet handles the 1/sqrt(2) normalization later when we normalize the vector
    op_a = nk.operator.fermion.create(hi_tmp, i, sz=1) @ \
           nk.operator.fermion.create(hi_tmp, j, sz=-1)
    
    op_b = nk.operator.fermion.create(hi_tmp, i, sz=-1) @ \
           nk.operator.fermion.create(hi_tmp, j, sz=1)
    return op_a + op_b

pairs = [(0, 4), (1, 2), (5, 6), (3, 7)] 
total_op = reduce(lambda a, b: a @ b, [get_pair_op(i, j) for i, j in pairs])
# numbers are the modes, not sites
# NOTE: ordering here: first all sites of down spins, then all sites of up spins (different from Majorana Propagation)
print("Creation from vacuum = ", total_op.operator_string())
x = total_op.to_sparse() @ vac
del vac
x /= np.linalg.norm(x)

# now go to the smaller hilbert space with this amount of particles
nz_idxs = x.nonzero()[0]
nz_states = hi_tmp.numbers_to_states(nz_idxs)
nz_values = x[nz_idxs]
del x
v0 = np.zeros((hi.n_states,), dtype=float)
v0[hi.states_to_numbers(nz_states)] = nz_values
v0 /= np.linalg.norm(v0)

print("Check:\n", hi.numbers_to_states(v0.nonzero()[0]))#check if the states are correct

############ system ############
dt = 0.01
U=4
t=1
# can also build custom operators, but we use Hubbard here
ham = nkx.operator.FermiHubbardJax(hi, g, t=t, U=U)

############ second order integrator ############
integrator = nkx.dynamics.Heun(dt)
ed = ExactDynamics(hi, ham, v0, integrator)

############ other operators ############
obs_up = {
    f"nup_{i}":nk.operator.FermionOperator2ndJax(hi, terms=[f"{i}^ {i}"], weights=[1.0]) for i in range(g.n_nodes)
}
obs_dn = {
    f"ndn_{i}":nk.operator.FermionOperator2ndJax(hi, terms=[f"{g.n_nodes+i}^ {g.n_nodes+i}"], weights=[1.0]) for i in range(0, g.n_nodes)
}

obs = {**obs_up, **obs_dn}
#obs["N_total"] = sum(obs_up.values()) + sum(obs_dn.values())
obs["hopping"] = nk.operator.FermionOperator2ndJax(hi, terms=[f"0^ 1", f"1^ 0"], weights=[1.0, 1.0]) 
obs["n0_up"] = nk.operator.FermionOperator2ndJax(hi, terms=[f"0^ 0"], weights=[1.0]) #n1_up in Majorana Propagation

############ Doublon operator, preparing terms and weights ############
doublon_terms = []
doublon_weights = []

for i in range(g.n_nodes):
    idx_up = i
    idx_dn = i + g.n_nodes
    term = f"{idx_up}^ {idx_up} {idx_dn}^ {idx_dn}"
    
    doublon_terms.append(term)
    doublon_weights.append(1.0)

obs["N_doublon"] = nk.operator.FermionOperator2ndJax(hi, terms=doublon_terms, weights=doublon_weights)
print("Operator = ", obs["N_doublon"].operator_string())#check if the doublon operator is correct

############ simulation ############
T = 2
exp_name = f"data/benchmark_TDVP_ED_U{U}"
ed.run(T, out=exp_name, obs=obs, tstops=np.arange(0, T, dt))

with open(f"{exp_name}.log", "r") as f:
    data = json.load(f)

#print(data.keys())

df = []
for k, v in data.items():
    iters = np.array(v["iters"])
    if isinstance(v, dict) and "Mean" in v:
        v = v["Mean"]
    if isinstance(v, dict) and "real" in v:
        v = np.array(v["real"]) + 1j*np.array(v["imag"])
    if isinstance(v, dict) and "value" in v:
        v = np.array(v["value"])
    v = np.array(v)
    _df = pd.DataFrame({"iters":iters, k:v})
    _df = _df.set_index("iters")
    df.append(_df)
df = pd.concat(df, axis=1)
#print(df)

############ getting axises ############
df.to_csv(f"{exp_name}_full.csv")

x_axis = df.index
y_doublon = df["N_doublon"].values.real
y_hopping = df["hopping"].values.real
y_n0_up = df["n0_up"].values.real
y_nup_avg = df.filter(like="nup_").mean(axis=1).values.real
#y_n_tot = df["N_total"].values.real
print("x-axis:", x_axis)
print("y_doublon:", y_doublon)
############ plotting ############
plt.figure(figsize=(10, 6))
plt.plot(x_axis, y_doublon, label="Total Doublon Count", color="black", linewidth=2)
plt.plot(x_axis, y_nup_avg, label="Avg Spin-Up Density", linestyle="--", alpha=0.8)
plt.plot(x_axis, y_hopping, label="Hopping Correlation", alpha=0.8)
plt.plot(x_axis, y_n0_up, label="Site 0 Spin-Up Density", linestyle="-.", alpha=0.8)
#plt.plot(x_axis, y_n_tot, label="Total Particle Count", linestyle=":", alpha=0.8)
plt.title(f"Fermi-Hubbard Dynamics (U={U}, Lattice 2x4)")
plt.xlabel("Time ($t$)")
plt.ylabel("Expectation Value")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.savefig("hubbard_analysis2x4.png", dpi=300)