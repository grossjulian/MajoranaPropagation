using Pkg
Pkg.activate(".")
#Pkg.rm("MajoranaPropagation") 
#Pkg.add(url="https://github.com/SparqleSim/MajoranaPropagation.jl", rev="check_superpos")
using MajoranaPropagation
using PauliPropagation
using JLD2
using Serialization
using Plots
using LaTeXStrings
using Measures
using ArgParse
using ColorSchemes


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
            default = 0.0
        "--t"
            arg_type = Float64
            default = 1.0
        "--n_layers"
            arg_type = Int
            default = 20
        "--dt"
            arg_type = Float64
            default = 0.1
        "--min_abs_coeff"
            arg_type = Float64
            default = 1e-5
        "--max_penalties"
            arg_type = Int
            default = 4
        "--from_checkpoint"
            arg_type= Bool
            default = false
        "--save_dir"
            arg_type = String
            default = "/cluster/scratch/jugross/MajoranaData/7x4/New_method/"
    end
    return parse_args(s)
end
args = parse_commandline()

function norm_preparation(data)
    norm_first = zeros(length(data))
    for i in 1:length(data)
        if data[i] != 0.0
            norm_first[i] = data[1]
        else
            #do nothing
        end
    end
    return norm_first - data 
end

is_spinful = true
N_x = args["nx"]
N_y = args["ny"]
N_spinful_sites = N_x * N_y
#site_doublon = 19

U = args["U"]
t = args["t"]
magnetic=1. #corresponding to exp(1j*pi)

n_layers = args["n_layers"]
dt = args["dt"]

min_abs_coeff = args["min_abs_coeff"]
max_penalties = args["max_penalties"]

save_dir = args["save_dir"] 
save_plots = save_dir * "Plots/"
mkpath(dirname(save_plots * "n_layers$n_layers/"))


save_dir_obs = save_dir * "Data/n_layers$n_layers/"
x_axis=(0.0:0.1:2.0)[1:n_layers+1]
x_axis_offset = (0.1:0.1:2.0)[1:n_layers]

#------------- initializing the plots -------------
ev_plot1 = plot(title = "U=0", xlabel = "Time", ylabel = L"n_{doublons}(t)", legend = :bottomright, legend_font_pointsize = 8,size = (600, 450), dpi = 300, fontfamily = "Computer Modern")
ev_plot2 = plot(title = "U=4", xlabel = "Time", ylabel = L"n_{doublons}(t)", legend = :bottomright, legend_font_pointsize = 8,size = (600, 450), dpi = 300, fontfamily = "Computer Modern")
length_plot1 = plot(title = "U=0", xlabel = "Layer",ylabel = "Majorana strings", yscale = :log10, legend = :bottomright,legend_font_pointsize = 12,size = (600, 450), dpi = 300, fontfamily = "Computer Modern")
length_plot2 = plot(title = "U=4", xlabel = "Layer",ylabel = "Majorana strings", yscale = :log10, legend = :bottomright,legend_font_pointsize = 12,size = (600, 450), dpi = 300, fontfamily = "Computer Modern")

ev_plot3 = plot(title = "Operatornorm", xlabel = "Time", ylabel = L"||n_{doublons}(0)||-||n_{doublons}(t)||", yscale = :log10, legend = :bottomright, legend_font_pointsize = 8,size = (600, 450), dpi = 300, fontfamily = "Computer Modern")
#=
#----------- U=0 --------------
### importing data from paper ###
res_trappedion_MitigatedU0=[1., 1.36232697, 2.1693349,  3.22204191, 4.30410087, 5.23239744, 5.89355431, 6.25840319, 6.37222126, 6.3264048,  6.22250238, 6.14067089, 6.12158351, 6.16509001, 6.24284158, 6.31785468, 6.36285208, 6.37112212, 6.35740461, 6.35027782, 6.38023339] 
res_trappedion_MitigatedU0=res_trappedion_MitigatedU0[1:n_layers+1]
    
res_trappedion_MPU0=[1., 1.33634955, 2.22654388, 3.37773866, 4.46683622, 5.28266153, 5.79154978, 6.04449803, 6.12486844, 6.13160517, 6.13155021, 6.1532513,  6.21240808, 6.29366129, 6.35827395, 6.39532788, 6.37885112, 6.31761698, 6.20876133, 6.06110856, 5.88921636]
res_trappedion_MPU0=res_trappedion_MPU0[1:n_layers+1]

res_rawU0=[2.75, 3.725, 4.10625, 5.0375, 5.925, 6.575, 6.3875, 6.68125, 6.46875, 6.675, 6.71875, 6.84375, 6.68125, 6.6875, 6.99375, 6.6875, 6.74375, 6.81875, 6.80625, 6.85]  
res_rawU0=res_rawU0[1:n_layers]

mac_strings = ["0.0001", "1.0e-6", "1.0e-8", "1.0e-10"]

blue_shades = ColorSchemes.Blues_9.colors[2:2:8]

for (i, mac) in enumerate(mac_strings)
    data   = load_object(save_dir_obs * "U0.0/min_abs_coeff$mac/n_doubl.jld2")[1:n_layers+1]
    maj_length = load_object(save_dir_obs * "U0.0/min_abs_coeff$mac/length.jld2")[1:n_layers+1]
    plot!(ev_plot1, x_axis, data, label = "Coeff. Truncation $mac", linestyle = :dash, marker = :circle, color = blue_shades[i])
    plot!(length_plot1, 1:n_layers+1, maj_length, label = "Coeff. Truncation $mac", linestyle = :dash, marker = :circle, color = blue_shades[i])
end

scatter!(ev_plot1, x_axis,          res_trappedion_MitigatedU0, label="TFLO + GPR",     color=:red,  marker=:utriangle)
scatter!(ev_plot1, x_axis,          res_trappedion_MPU0,        label="MP Trapped Ion", color=:black, marker=:circle)
scatter!(ev_plot1, x_axis_offset,   res_rawU0,                  label="Raw",            color=:gray,   marker=:square)

# ----------- Save plots -----------
savefig(ev_plot1, save_plots * "n_layers$n_layers/n_doublonU0.png")
savefig(length_plot1, save_plots * "n_layers$n_layers/majorana_stringsU0.png")
=#

#----------- U=4 ------------
res_trappedion_MitigatedU4=[1., 1.3545207, 2.05291137, 2.91851, 3.75966626, 4.42860535, 4.84644241, 5.00966931, 4.97521595, 4.83119816, 4.6644016, 4.53534766, 4.46782283, 4.45380012, 4.46912079, 4.49213935, 4.51763983, 4.56124297, 4.65377483, 4.82889322, 5.10932587] 
res_trappedion_MitigatedU4=res_trappedion_MitigatedU4[1:n_layers+1]

res_trappedion_MPU4=[1., 1.33133454, 2.15653355, 3.09474725, 3.79110822, 4.11548705, 4.19728702, 4.19541411, 4.12284926, 4.05651392, 3.94204099, 3.90222159, 3.86832259, 3.84955277, 3.82928514, 3.82889834, 3.82034817, 3.84162375, 3.90593041, 4.02339119, 4.20750945]
res_trappedion_MPU4=res_trappedion_MPU4[1:n_layers+1]

res_rawU4=[3.08125, 3.41875, 4.175, 4.69375, 5.125, 5.28125, 5.48125, 5.24375, 5.25, 5.01875, 5.41875, 5.4, 5.1125, 4.975, 5.18125, 5.05625, 5.23125, 5.01875, 5.53125, 5.70625]
res_rawU4=res_rawU4[1:n_layers]

###################### ausnahmen
#=
save_dir_old = "/cluster/scratch/jugross/MajoranaData/7x4/Phasecorrected_U4_2/"
save_dir_obs_old = save_dir_old * "Data/n_layers$n_layers/"
mac_strings      = ["1.0e-6"] #"0.0001 " , "1.0e-5"
max_unpaired_values = [4, 6, 8]

exponent_colors  = ColorSchemes.Blues_9.colors[3:4:7] # must add the 9th color if 3 mac_strings are considered
unpaired_markers = [:circle, :square, :utriangle]#

for (i,  mac) in enumerate(mac_strings)
    for (j, unp) in enumerate(max_unpaired_values)  
        path = save_dir_obs_old * "U4.0/min_abs_coeff$mac/max_unpaired$unp/n_doubl.jld2"
        data = load_object(path)[1:n_layers+1]
        maj_length = load_object(save_dir_obs_old * "U4.0/min_abs_coeff$mac/max_unpaired$unp/length.jld2")[1:n_layers+1]
        norm_data = load_object(save_dir_obs_old * "U4.0/min_abs_coeff$mac/max_unpaired$unp/norm.jld2")[1:n_layers+1]
        plot!(ev_plot2, x_axis, [y <= 0 ? NaN : y for y in data], label = "coeff=$mac, unp=$unp", linestyle = :dash, color  = exponent_colors[i], marker = unpaired_markers[j])
        plot!(length_plot2, 1:n_layers+1, maj_length, label = "coeff=$mac, unp=$unp", linestyle = :dash, color  = exponent_colors[i], marker = unpaired_markers[j])
        plot!(ev_plot3, x_axis, [y <= 0 ? NaN : y for y in norm_preparation(norm_data)], label = "coeff=$mac, unp=$unp", linestyle = :dash, color  = exponent_colors[i], marker = unpaired_markers[j])
    end
end
=#
#######################

mac_strings      = ["1.0e-6"] #"0.0001 " , "1.0e-5" , "1.0e-6"
max_penalties_values = [3, 4, 5, 6] #, 6, 8

exponent_colors  =  ColorSchemes.managua10.colors[1:1:4]  #  if 3 mac_strings are considered get(ColorSchemes.Greens_9, range(0.2, 0.8, length=length(max_penalties_values)))
penalties_markers = [:circle, :square, :utriangle, :dtriangle]#

for (i,  mac) in enumerate(mac_strings)
    for (j, pen) in enumerate(max_penalties_values)
        println(j)
        path = save_dir_obs * "U4.0/min_abs_coeff$mac/max_penalties$pen/n_doubl.jld2"
        data = load_object(path)[1:n_layers+1]
        maj_length = load_object(save_dir_obs * "U4.0/min_abs_coeff$mac/max_penalties$pen/length.jld2")[1:n_layers+1]
        norm_data = load_object(save_dir_obs * "U4.0/min_abs_coeff$mac/max_penalties$pen/norm.jld2")[1:n_layers+1]
        plot!(ev_plot2, x_axis, [y <= 0 ? NaN : y for y in data], label = "coeff=$mac, pen=$pen", linestyle = :dash, color  = exponent_colors[j], marker = penalties_markers[j])
        plot!(length_plot2, 1:n_layers+1, [y <= 0 ? NaN : y for y in maj_length], label = "coeff=$mac, pen=$pen", linestyle = :dash, color  = exponent_colors[j], marker = penalties_markers[j])
        plot!(ev_plot3, x_axis, [y <= 0 ? NaN : y for y in norm_preparation(norm_data)], label = "coeff=$mac, pen=$pen", linestyle = :dash, color  = exponent_colors[j], marker = penalties_markers[j])
    end
end
#=
save_dir_old = "/cluster/scratch/jugross/MajoranaData/7x4/New_method1e-6/"
save_dir_obs_old = save_dir_old * "Data/n_layers$n_layers/"
mac_strings      = ["1.0e-6"] #"0.0001 " , "1.0e-5"
max_penalties_values = [3, 4, 5, 6]

exponent_colors  = ColorSchemes.navia10.colors[1:1:4]  # must add the 9th color if 3 mac_strings are considered
unpaired_markers = [:circle, :square, :utriangle, :dtriangle]#

for (i,  mac) in enumerate(mac_strings)
    for (j, pen) in enumerate(max_penalties_values)  
        path = save_dir_obs_old * "U4.0/min_abs_coeff$mac/max_penalties$pen/n_doubl.jld2"
        data = load_object(path)[1:n_layers+1]
        maj_length = load_object(save_dir_obs_old * "U4.0/min_abs_coeff$mac/max_penalties$pen/length.jld2")[1:n_layers+1]
        norm_data = load_object(save_dir_obs_old * "U4.0/min_abs_coeff$mac/max_penalties$pen/norm.jld2")[1:n_layers+1]
        plot!(ev_plot2, x_axis, [y <= 0 ? NaN : y for y in data], label = "coeff=$mac, pen=$pen", linestyle = :dash, color  = exponent_colors[j], marker = unpaired_markers[j])
        plot!(length_plot2, 1:n_layers+1, maj_length, label = "coeff=$mac, pen=$pen", linestyle = :dash, color  = exponent_colors[j], marker = unpaired_markers[j])
        plot!(ev_plot3, x_axis, [y <= 0 ? NaN : y for y in norm_preparation(norm_data)], label = "coeff=$mac, pen=$pen", linestyle = :dash, color  = exponent_colors[j], marker = unpaired_markers[j])
    end
end

path = "/cluster/scratch/jugross/MajoranaData/7x4/Importantfirst_nounpaired/Data/n_layers6/U4.0/min_abs_coeff1.0e-7/max_unpaired-1/"
data = vcat(load_object(path * "n_doubl.jld2"),zeros(21-length(load_object(path * "n_doubl.jld2"))))
data = data[1:n_layers+1]
data2 = vcat(load_object(path * "norm.jld2"),zeros(21-length(load_object(path * "norm.jld2"))))
data2 = data2[1:n_layers+1]
plot!(ev_plot2, x_axis, [y <= 0 ? NaN : y for y in data], label = "coeff=1.0e-7, unp=-1", linestyle = :dash, color  = :orange)
plot!(ev_plot3, x_axis, [y <= 0 ? NaN : y for y in norm_preparation(data2)], label = "coeff=1.0e-7, unp=-1", linestyle = :dash, color  = :orange)

path = "/cluster/scratch/jugross/MajoranaData/7x4/Importantfirst_nounpaired/Data/n_layers6/U4.0/min_abs_coeff1.0e-8/max_unpaired-1/"
data = vcat(load_object(path * "n_doubl.jld2"),zeros(21-length(load_object(path * "n_doubl.jld2"))))
data = data[1:n_layers+1]
data2 = vcat(load_object(path * "norm.jld2"),zeros(21-length(load_object(path * "norm.jld2"))))
data2 = data2[1:n_layers+1]
plot!(ev_plot2, x_axis, [y <= 0 ? NaN : y for y in data], label = "coeff=1.0e-8, unp=-1", linestyle = :dash, color  = :green)
plot!(ev_plot3, x_axis, [y <= 0 ? NaN : y for y in norm_preparation(data2)], label = "coeff=1.0e-8, unp=-1", linestyle = :dash, color  = :green)
##
=#

scatter!(ev_plot2, x_axis,          res_trappedion_MitigatedU4, label="TFLO + GPR",     color=:red,  marker=:utriangle)
scatter!(ev_plot2, x_axis,          res_trappedion_MPU4,        label="MP Trapped Ion", color=:black, marker=:circle)
scatter!(ev_plot2, x_axis_offset,   res_rawU4,                  label="Raw",            color=:gray,   marker=:square)

# ----------- Save plots -----------
savefig(ev_plot2, save_plots * "n_layers$n_layers/n_doublonU4.png")
savefig(length_plot2, save_plots * "n_layers$n_layers/majorana_stringsU4.png")
savefig(ev_plot3, save_plots * "n_layers$n_layers/operatornormU4.png")