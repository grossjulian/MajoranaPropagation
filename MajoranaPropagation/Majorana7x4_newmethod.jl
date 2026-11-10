using Pkg
Pkg.activate(".")
#Pkg.add("MajoranaPropagation")
#Pkg.add(url="https://github.com/SparqleSim/MajoranaPropagation.jl", rev="phase_hopping")
using MajoranaPropagation
using PauliPropagation
using JLD2
using Plots
using LaTeXStrings
using Measures
using Serialization
using ArgParse
using LinearAlgebra
using Base.Threads

using Combinatorics
using TimerOutputs
using ProgressMeter

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nx"
            arg_type = Int
            default = 7 
        "--ny"
            arg_type = Int
            default = 4 
        "--U"
            arg_type = Float64
            default = 4.0
        "--t"
            arg_type = Float64
            default = 1.0
        "--n_layers"
            arg_type = Int
            default = 2
        "--dt"
            arg_type = Float64
            default = 0.1
        "--min_abs_coeff"
            arg_type = Float64
            default = 1e-4
        "--max_penalties"
            arg_type = Int
            default = 4
        "--from_checkpoint"
            arg_type= Bool
            default = false
        "--save_dir"
            arg_type= String
            default = "/cluster/scratch/jugross/MajoranaData/default/" 
    end
    return parse_args(s)
end

#new method
"""
count_penalties_and_truncate
returns true if tot_penalties > max_penalties, coherent with MP convention
"""
function count_penalties_and_truncate(ms::TT, site_pairs::Vector{Tuple{Int64,Int64}}, fock_like_sites::Vector{Int64}, valid_pairs::Vector{Bool}, unpaired_mask::FTT, max_penalties::Int) where {TT<:Integer,FTT<:Integer}
    tot_penalties = 0

    for (i, j) in site_pairs
        #start by putting i-site bits in right position
        ms_ij = extract_spinful_site(ms, i)
        #extract site j
        ms_j = extract_spinful_site(ms, j)
        #push site j indices to right position
        ms_ij = (ms_ij & ~(0xF << 4)) | ((ms_j & 0xF) << 4)
        #valid_pairs is a 256-entry lookup table over all two-site nibble patterns
        if !(@inbounds valid_pairs[Int(ms_ij)+1])
            tot_penalties += 1
            if tot_penalties > max_penalties
                return true
            end
        end
    end

    ms_fock_like = extract_spinful_site(ms, fock_like_sites[1])
    temp = extract_spinful_site(ms, fock_like_sites[2])
    ms_fock_like = (ms_fock_like & ~(0xF << 4)) | ((temp & 0xF) << 4)
    tot_penalties += compute_unpaired(ms_fock_like, unpaired_mask)
    
    if tot_penalties > max_penalties
        return true
    end
    
    return false
end

#important functions for initial state
function expand_tensor(pairs)
    result = [Dict{Int,Any}()]
    for pair in pairs
        new_result = []
        for r in result
            for p in pair
                merged = copy(r)
                merge!(merged, p)
                push!(new_result, merged)
            end
        end
        result = new_result
    end

    return result
end

function to_fock(state, N)
    ups = Int[]
    dns = Int[]

    for (site, spin) in state

        if spin == :updn || spin == :dnup
            push!(ups, site)
            push!(dns, site)

        elseif spin == :up
            push!(ups, site)

        elseif spin == :dn
            push!(dns, site)
        end
    end

    return FockState(N, sort(ups), sort(dns))
end

function precompute_overlap_dict()
    N_spinful_sites = 2
    unpaired_mask = create_unpaired_mask(2 * N_spinful_sites)
    fock1 = FockState(N_spinful_sites, [1], [2])
    fock2 = FockState(N_spinful_sites, [2], [1])

    superpos = [fock1, fock2]
    coeffs = [1 / sqrt(2), 1 / sqrt(2)]

    n_modes = 4 * N_spinful_sites
    TT = getinttype(2 * N_spinful_sites)
    strings_to_check = TT[]

    for k in 0:2:8
        for combo in combinations(1:n_modes, k)
            ms = MajoranaString(2 * N_spinful_sites, combo)
            push!(strings_to_check, ms.gammas)
        end
    end

    two_sites_overlap = Dict{TT,ComplexF64}()

    for string in strings_to_check
        overlap = overlapwithfock(string, superpos, coeffs, 2 * N_spinful_sites, unpaired_mask)
        if abs(overlap) > 1.e-12
            two_sites_overlap[string] = overlap / (1im)^omega_L_mult(string)
        end
    end
    return two_sites_overlap
end

function extract_spinful_site(ms::TT, j::Int64) where {TT<:Integer}
    return UInt8((ms >> (4 * (j - 1))) & 0xF)
end

function overlapwithsuperposition(ms::TT, site_pairs::Vector{Tuple{Int64, Int64}}, fock_like_sites::Vector{Int64}, pairs_overlap_dict::Dict{STT,ComplexF64}, fock_like_part::FockState) where {TT<:Integer, STT<:Integer}
    res = (1im)^omega_L_mult(ms)

    for (i, j) in site_pairs
        #start by putting i-site bits in right position
        ms_ij = extract_spinful_site(ms, i)
        #extract site j
        ms_j = extract_spinful_site(ms, j)
        #push site j indices to right position
        ms_ij = (ms_ij & ~(0xF << 4)) | ((ms_j & 0xF) << 4)
        if haskey(pairs_overlap_dict, ms_ij)
            res *= pairs_overlap_dict[ms_ij]
        else
            #println("breaking at pair $i, $j")
            return 0.
        end
    end

    #@assert length(fock_like_sites) == 2
    unpaired_mask = create_unpaired_mask(2 * length(fock_like_sites))
    ms_fock_like = extract_spinful_site(ms, fock_like_sites[1])
    temp = extract_spinful_site(ms, fock_like_sites[2])
    ms_fock_like = (ms_fock_like & ~(0xF << 4)) | ((temp & 0xF) << 4)
    res *= overlapwithfock(ms_fock_like, unpaired_mask, fock_like_part) / (1im)^omega_L_mult(ms_fock_like)
    return real(res)
end

function overlapwithsuperposition(msum::VectorMajoranaSum, site_pairs, fock_like_sites, pairs_overlap_dict::Dict{STT,ComplexF64}, fock_like_part::FockState) where {STT<:Integer}
    #currently this only work for length(fock_like_sites) == 2
    @assert length(fock_like_sites) == 2

    ts = msum.terms
    cs = msum.coeffs
    all_res = zeros(ComplexF64, length(msum))
    @threads for j in eachindex(all_res)
        all_res[j] = cs[j] * overlapwithsuperposition(ts[j], site_pairs, fock_like_sites, pairs_overlap_dict, fock_like_part)
    end
    return real(sum(all_res))
end

function separate_pairs(topo)
    separated_lists = []
    for pair in topo
        found_list = false
        for k=1:length(separated_lists)
            is_good = true
            for pair_already_in in separated_lists[k]
                if (pair[1] in pair_already_in) || (pair[2] in pair_already_in)
                    is_good = false
                    break 
                end 
            end
            if is_good 
                push!(separated_lists[k], pair)
                found_list = true
                break 
            end
        end

        if found_list == false 
            push!(separated_lists, [pair])
        end
    end
    #@show length(separated_lists)
    new_topo = []
    for list in separated_lists
        #@show list 
        append!(new_topo, list)
    end
    return new_topo
end

function buildphase_up(i, j, circ_single, thetas_single)
    @assert i<j #should never be the case 

    if j-i==21 #vertical boundary
        push!(circ_single, FermionicRotation(:hopup_minuspifourth_phase, [i, j]))
        push!(thetas_single, dt/2)
    end
    if j-i==7 #normal vertical
        push!(circ_single, FermionicRotation(:hopup_pifourth_phase, [i, j]))
        push!(thetas_single,  dt/2)
    end
    if j-i==1 || j-i==6 # horizontal
        push!(circ_single, FermionicRotation(:hopup, [i, j]))
        push!(thetas_single, -t * magnetic * dt/2)
    end
end

function buildphase_dn(i,j,circ_single, thetas_single)
    @assert i<j #should never be the case 

    if j-i==21 #vertical boundary
        push!(circ_single, FermionicRotation(:hopdn_minuspifourth_phase, [i, j]))
        push!(thetas_single, dt/2)
    end
    if j-i==7 #normal vertical
        push!(circ_single, FermionicRotation(:hopdn_pifourth_phase, [i, j]))
        push!(thetas_single, dt/2)
    end
    if j-i==1 || j-i==6 # horizontal
        push!(circ_single, FermionicRotation(:hopdn, [i, j]))
        push!(thetas_single, -t * magnetic * dt/2)
    end
end

#System
args = parse_commandline()

is_spinful = true
N_x = args["nx"]
N_y = args["ny"]
N_spinful_sites = N_x * N_y

U = args["U"]
t = args["t"]
magnetic = 1. #for now: e^{0}=1, no flux dependency

n_layers = args["n_layers"]
dt = args["dt"]

min_abs_coeff = args["min_abs_coeff"]
max_penalties = args["max_penalties"]
unpaired_mask = create_unpaired_mask(2 * N_spinful_sites);

topo = rectangletopology(N_x, N_y,periodic=true)
topo = separate_pairs(topo)
circ_single = []
thetas_single = []

#up hoppings 
for (i, j) in topo
    buildphase_up(i,j,circ_single, thetas_single)
end

#down hoppings 
for (i, j) in topo
    buildphase_dn(i,j,circ_single, thetas_single)
end

#on-site repulsion 
for i = 1:N_spinful_sites
    push!(circ_single, FermionicRotation(:nupndn, i))
    push!(thetas_single, U * dt)
end

#down hoppings 
for (i, j) in reverse(topo)
    buildphase_dn(i,j,circ_single, thetas_single)
end

#up hoppings 
for (i, j) in reverse(topo)
    buildphase_up(i,j,circ_single, thetas_single)
end

#construction of initial state from doublon and pairs
site_doublon = 19 
site_hole = 2 
fock_like_sites = [site_hole, site_doublon]
fock_like_part = FockState(2, [2], [2])# hole is site 1, doublon is site 2

site_pairs = [(1,8), (3,4), (5,12), (6,13), (7,14), (9,16), (10,17), (11,18), (15,22), (23,24), (25,26), (20,27), (21,28)]
pairs = [ [Dict(site_doublon => :updn)] ] #doublon
two_sites_overlap = precompute_overlap_dict()

for (s1, s2) in site_pairs
    push!(pairs, [Dict(s1 => :up, s2 => :dn), Dict(s1 => :dn, s2 => :up)])
end

#preparing inital states
states = expand_tensor(pairs)
fock_states = [to_fock(s, N_spinful_sites) for s in states]
initial_state = [fock_states[i] for i in 1:length(fock_states)]
coeffs = [1/(sqrt(2)^13) for i in 1:length(initial_state)] #13 superpositions, each with coeff 1/sqrt(2)

#preparing truncation function
two_sites_strings = collect(keys(two_sites_overlap))
unpaired_mask_fock_like_part = create_unpaired_mask(2 * length(fock_like_sites))
# lookup table: valid_pairs[k+1] is true iff two-site pattern has nonzero overlap
valid_pairs = let v = fill(false, 256)
    for k in two_sites_strings
        v[Int(k)+1] = true
    end
    v
end

compute_penalty = let sp = site_pairs, fs = fock_like_sites, vp = valid_pairs, mask = unpaired_mask_fock_like_part, mp = max_penalties
    (mstr, coeff) -> count_penalties_and_truncate(mstr, sp, fs, vp, mask, mp)
end

#observables
obs_doubl = MajoranaSum(Float64, N_spinful_sites, is_spinful)
for i=1:N_spinful_sites
    add!(obs_doubl, :nupndn, i)
end
obs_doubl = VectorMajoranaSum(obs_doubl)

save_dir = args["save_dir"]
from_checkpoint = args["from_checkpoint"]

if from_checkpoint == false
    #no checkpoint, initialize from beginning
    n_doubl = zeros(n_layers+1)
    length1 = zeros(n_layers+1)
    norm1 = zeros(n_layers+1)
    n_doubl[1] = overlapwithsuperposition(obs_doubl, site_pairs, fock_like_sites, two_sites_overlap, fock_like_part)
    length1[1] = length(obs_doubl)
    norm1[1] = LinearAlgebra.norm(obs_doubl.coeffs)
    if U==0
        save_path = save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/"
        mkpath(dirname(save_path))
        save_path_n_doubl = save_path * "n_doubl.jld2"
        save_path_length = save_path * "length.jld2"
        save_path_sum_obs = save_path * "Majorana_sum_obs.jls"
        save_path_norm = save_path * "norm.jld2"
        for j = 1:n_layers 
            print("$j ")
            propagate!(circ_single, obs_doubl, thetas_single; min_abs_coeff=min_abs_coeff, customtruncfunc=compute_penalty)
            n_doubl[j+1] = overlapwithsuperposition(obs_doubl, site_pairs, fock_like_sites, two_sites_overlap, fock_like_part)
            length1[j+1] = length(obs_doubl)
            norm1[j+1] = LinearAlgebra.norm(obs_doubl.coeffs)
            save_object(save_path_n_doubl, n_doubl)
            save_object(save_path_length, length1)
            save_object(save_path_norm, norm1)
            open(save_path_sum_obs, "w") do io
                        serialize(io, obs_doubl)
                    end
        end
    else
        save_path = save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/"
        mkpath(dirname(save_path))
        save_path_n_doubl = save_path * "n_doubl.jld2"
        save_path_length = save_path * "length.jld2"
        save_path_sum_obs = save_path * "Majorana_sum_obs.jls"
        save_path_norm = save_path * "norm.jld2"
        for j = 1:n_layers 
            print("$j ")
            propagate!(circ_single, obs_doubl, thetas_single; min_abs_coeff=min_abs_coeff, customtruncfunc=compute_penalty)
            n_doubl[j+1] = overlapwithsuperposition(obs_doubl, site_pairs, fock_like_sites, two_sites_overlap, fock_like_part)
            length1[j+1] = length(obs_doubl)
            norm1[j+1] = LinearAlgebra.norm(obs_doubl.coeffs)
            save_object(save_path_n_doubl, n_doubl)
            save_object(save_path_length, length1)
            save_object(save_path_norm, norm1)
            open(save_path_sum_obs, "w") do io
                        serialize(io, obs_doubl)
                    end
        end
    end
else
    if U==0
        #load previous data
        n_doubl = load_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/n_doubl.jld2")
        length1 = load_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/length.jld2")
        norm1 = load_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/norm.jld2")
        obs_doubl = open(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/Majorana_sum_obs.jls", "r") do io
                deserialize(io)
            end
        start_layer = findlast(!iszero, n_doubl)
        for j = start_layer:n_layers
            print("$j ")
            propagate!(circ_single, obs_doubl, thetas_single; min_abs_coeff=min_abs_coeff, customtruncfunc=compute_penalty)
            n_doubl[j+1] = overlapwithsuperposition(obs_doubl, site_pairs, fock_like_sites, two_sites_overlap, fock_like_part)
            length1[j+1] = length(obs_doubl)
            norm1[j+1] = LinearAlgebra.norm(obs_doubl.coeffs)
            save_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/n_doubl.jld2", n_doubl)
            save_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/length.jld2", length1)
            save_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/norm.jld2", norm1)
            open(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/Majorana_sum_obs.jls", "w") do io
                        serialize(io, obs_doubl)
                    end
        end
    else
        #load previous data
        n_doubl = load_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/n_doubl.jld2")
        length1 = load_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/length.jld2")
        norm1 = load_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/norm.jld2")
        obs_doubl = open(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/Majorana_sum_obs.jls", "r") do io
                deserialize(io)
            end
        start_layer = findlast(!iszero, n_doubl)
        for j = start_layer:n_layers
            print("$j ")
            propagate!(circ_single, obs_doubl, thetas_single; min_abs_coeff=min_abs_coeff, customtruncfunc=compute_penalty)
            n_doubl[j+1] = overlapwithsuperposition(obs_doubl, site_pairs, fock_like_sites, two_sites_overlap, fock_like_part)
            length1[j+1] = length(obs_doubl)
            norm1[j+1] = LinearAlgebra.norm(obs_doubl.coeffs)
            save_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/n_doubl.jld2",n_doubl)
            save_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/length.jld2",length1)
            save_object(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/norm.jld2",norm1)
            open(save_dir * "n_layers$n_layers/U$U/min_abs_coeff$min_abs_coeff/max_penalties$max_penalties/Majorana_sum_obs.jls", "w") do io
                        serialize(io, obs_doubl)
                    end
        end
    end
end
