using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MajoranaPropagation

L_sites = 10
t_hop = 1.0
U_repulsion = 2.0
dt = 0.1
n_steps = 6
topo = bricklayertopology(L_sites)

circ_single_step = []
thetas_single_step = []
for (i, j) in topo
    push!(circ_single_step, FermionicGate(:hop, [i, j]))
    push!(thetas_single_step, -t_hop * dt)
end
for (i, j) in topo
    push!(circ_single_step, FermionicGate(:nn, [i, j]))
    push!(thetas_single_step, U_repulsion * dt)
end


# Set initial Fock state
# specify the sites where fermions are created
fock_state = FockState(L_sites, 1:2:L_sites)

# Set observable 
obs = MajoranaSum(L_sites, :n, 1) * MajoranaSum(L_sites, :n, 5)
@show obs 

# Propagate 
for step = 1:n_steps
    propagate!(circ_single_step, obs, thetas_single_step; min_abs_coeff=1.e-8)
    @show overlapwithfock(obs, fock_state)
end