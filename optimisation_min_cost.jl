using JuMP
using Clp
using Plots
using DataFrames, CSV
using Gurobi
include("helper_functions.jl")

# dictzip
zipcols(df::DataFrame, x::Symbol) = df[:,x] |> Vector
zipcols(df::DataFrame, x::Vector) = zip(eachcol(df[:,x])...) |> collect

function dictzip(df::DataFrame, x::Pair)
    dictkeys = zipcols(df, x[1])
    dictvalues = zipcols(df, x[2])
    return zip(dictkeys, dictvalues) |> collect |> Dict
end


dir = dirname(Base. source_path())
time_series = CSV.read(joinpath(dir, "data\\timedata_hourly.csv"), DataFrame)

tech_data = CSV.read(joinpath(dir, "data\\technologies_pipeline.csv"), DataFrame)

### data preprocessing ###
T_len = time_series.hour |> unique |> length
T = 1:size(time_series, 1) |> collect
P = tech_data[:,:technology] |> Vector
DISP = tech_data[tech_data[!,:dispatchable] .== 1 ,:technology]
NONDISP = tech_data[tech_data[!,:dispatchable] .== 0 ,:technology]
S = tech_data[tech_data[!,:investment_storage] .> 0 ,:technology]
# Heat generation (requires heat balance)
HEAT = tech_data[(tech_data[:,:thermal_eff] .!= 0), :technology]
P_CHP = intersect(DISP,HEAT)
H_only = symdiff(HEAT,P_CHP)

# Import/Export with National Grid
# Z = unique(zones[:,:zone])
# P = zones[:,:id] |> Vector

### parameters ###
annuity_factor(n,r) = r * (1+r)^n / (((1+r)^n)-1)

interest_rate = 0.08

pipeline_cost = 1000 # Pipeline cost per meter
n_pipeline = 30 # years



ic_generation_cap = Dict{String, Float64}()
ic_charging_cap = Dict{String, Float64}()
ic_storage_cap = Dict{String, Float64}()
eff_in = Dict{String, Float64}()
eff_out = Dict{String, Float64}()
vc = Dict{String, Float64}()
# Heat 
heat_ratio = dictzip(tech_data, :technology => :heat_ratio)



for row in eachrow(tech_data)
    af = annuity_factor(row.lifetime, interest_rate)
    
    ic_generation_cap[row.technology] = row.investment_generation * af + row.o_and_m + row.pipeline_m * pipeline_cost * annuity_factor(n_pipeline,interest_rate)# Invetment costs plus O&M costs
    # # Costs for pipeline over lifetime



    iccc = row.investment_charge * af
    iccc > 0 && (ic_charging_cap[row.technology] = iccc)

    icsc = row.investment_storage * af
    icsc > 0 && (ic_storage_cap[row.technology] = icsc)

    row.storage_efficiency_in > 0 && (eff_in[row.technology] = row.storage_efficiency_in)
    row.storage_efficiency_out > 0 && (eff_out[row.technology] = row.storage_efficiency_out)

    vc[row.technology] = row.vc
end

ic_generation_cap
tech_data.investment_generation
ic_storage_cap

### time series ###
# demand_elec_res = time_series[:,:demand_elec_res] |> Array
# demand_heat_res = time_series[:,:demand_heat_res] |> Array
# demand_elec_inst = time_series[:,:demand_elec_inst] |> Array
# demand_heat_inst = time_series[:,:demand_heat_inst] |> Array
demand_elec = time_series[:,:demand_elec] |> Array
demand_heat = time_series[:,:demand_heat] |> Array

demand_elec
demand_heat

# Feed in of renewable energy
availability = Dict(nondisp => time_series[:,nondisp] for nondisp in NONDISP)
# Storage level at (t+1)
successor(arr, x) = (x == length(arr)) ? 1 : x + 1

# How many hours in t
dispatch_scale = 8760/length(T)


### model ### 
# m = Model(JuMP.Optimizer)
m = Model(Gurobi.Optimizer)

@variables m begin
    # variables dispatch (include heat)
    G[P, T] >= 0
    H[HEAT, T] >= 0
    CU[T] >= 0
    HeatDump[T] >= 0
    D_stor[S,T] >= 0
    L_stor[S,T] >= 0

    # GRID[T] >= 0
    # variables investment model
    CAP_G[P] >= 0
    CAP_D[S] >= 0
    CAP_L[S] >= 0
end

@objective(m, Min,
    sum(vc[disp] * G[disp,t] for disp in DISP, t in T) #* dispatch_scale
    + sum(vc[h] * H[h,t] for h in H_only, t in T) #* dispatch_scale
    + sum(ic_generation_cap[p] * CAP_G[p] for p in P)
    #+ sum(ic_charging_cap[s] * CAP_D[s] for s in S if haskey(ic_charging_cap, s))
    + sum(ic_storage_cap[s] * CAP_L[s] for s in S)
)

# Renewable generation
@expression(m, feed_in[ndisp=NONDISP, t=T], availability[ndisp][t]*CAP_G[ndisp])

@constraint(m, ElectricityBalance[t=T],
    sum(G[disp,t] for disp in DISP)
    + sum(feed_in[ndisp,t] for ndisp in NONDISP)
    - sum(D_stor[s,t] for s in S)
    - CU[t]
    ==
    demand_elec[t] #/ dispatch_scale
    + H["heatpumps",t] / 3.5 # Considering electricity consumption of heat pumps
    )

# Dispatchable electricity generation must equal installed capacity (CHP)
@constraint(m, MaxElecGeneration[disp=DISP, t=T],
    G[disp,t] <= CAP_G[disp])

@constraint(m,HeatBalance[t=T],
    sum(H[ht,t] for ht in HEAT)
    ==
    demand_heat[t] #/ dispatch_scale
    )

# Generation of heat must equal installed capacity (CHP, heatpumps)
@constraint(m,MaxHeatGeneration[ht=HEAT,t=T],
    H[ht,t] <= CAP_G[ht])

# Cogeneration of CHP plant
@constraint(m, CoGeneration[chp=P_CHP,t=T],
    G[chp,t] == 1/heat_ratio[chp] * H[chp, t])


##########
# Battery

@constraint(m, MaxCharge[s=S, t=T; haskey(ic_charging_cap, s)],
    D_stor[s,t] <= CAP_D[s])

@constraint(m, SymmetricChargingPower[s=S, t=T; !(haskey(ic_charging_cap, s))],
    CAP_G[s] == CAP_D[s])

@constraint(m, MaxLevel[s=S, t=T],
    L_stor[s,t] <= CAP_L[s])

@constraint(m, StorageLevel[s=S, t=T],
    L_stor[s, successor(T,t)]
    ==
    L_stor[s, t]
    + eff_in[s]*D_stor[s,t]
    - (1/eff_out[s]) * G[s,t] )


@constraint(m,MaxPV[ndisp=NONDISP],
    CAP_G[ndisp] <= 26228.18) # Max is 26MW


# CONSTRAINT FOR CHP locations based on pipeline network model
@constraint(m,MaxHospital,
    CAP_G["chp_hospital"]<= 769.0068)

# @constraint(m,MaxShopping,
#     CAP_G["chp_shopping"]<= 2321.447)

@constraint(m,Maxchp1_2cells,
    CAP_G["chp1_2cells"]<= 690.411)

# @constraint(m,Maxchp2_2cells,
#     CAP_G["chp2_2cells"]<= 728.7671)

# @constraint(m,Maxchp_4cells,
#     CAP_G["chp_4cells"]<= 859.069)



optimize!(m)

objective_value(m)

value.(CAP_G)
value.(H)
#################################
## Plots
#################################

#### Plot ELECTRICITY BALANCE

colordict = Dict(
    "pv" => :yellow,
    "chp_hospital" => :dodgerblue1,
    # "chp_shopping" => :midnightblue, # chp_shopping,1,52.3,0.082222222,25,0.59,1046,-1,-1,-1,-1,0.7,0.27,0.45,2341.421
    "chp1_2cells" => :dodgerblue4,
    # "chp2_2cells" => :dodgerblue3, #chp2_2cells,1,52.3,0.082222222,25,0.59,1046,-1,-1,-1,-1,0.7,0.27,0.45,1600
    # "chp_4cells" => :steelblue3, # chp_4cells,1,52.3,0.082222222,25,0.59,1046,-1,-1,-1,-1,0.7,0.27,0.45,2200
    "heatpumps" => :yellow, # 
    # "gas_boiler" => :slateblue2, #gas_boiler,-1,6.5,0.11,25,0.31,259.5,-1,-1,-1,-1,0,0.9,0,0
    "battery" => :lightseagreen,
    "demand" => :darkgrey,
    "curtailment" => :red,
    #"heat_dump" => :red,
)


######## plot electricity balance ###########

result_G = get_result(G, [:technology, :hour])
result_feed_in = get_result(feed_in, [:technology, :hour])

result_charging = get_result(D_stor, [:technology, :hour])
result_CU = get_result(CU, [:hour])
result_CU[!,:technology] .= "curtailment"
df_demand = DataFrame(hour=T, technology="demand", value=demand_elec)

result_generation = vcat(result_feed_in, result_G)
result_demand = vcat(result_charging, result_CU, df_demand)

table_gen = unstack(result_generation, :hour, :technology, :value,combine=sum)
table_gen = table_gen[!,[NONDISP..., DISP...]]
labels = names(table_gen) |> permutedims
colors = [colordict[tech] for tech in labels]
data_gen = Array(table_gen)

balance_plot = areaplot(
    data_gen,
    label=labels,
    color=colors,
    width=0,
    leg=:outertopright
)

table_dem = unstack(result_demand, :hour, :technology, :value)
table_dem = table_dem[!,["demand", S...,"curtailment"]]
labels2 = names(table_dem) |> permutedims
colors2 = [colordict[tech] for tech in labels2]
replace!(labels2, [item => "" for item in intersect(labels2, labels)]...)
data_dem = -Array(table_dem)

areaplot!(
    balance_plot,
    data_dem,
    label=labels2,
    color=colors2,
    width=0,
    leg=:outertopright
)

hline!(balance_plot, [0], color=:black, label="", width=2)

savefig("results\\Dispatch_Electricity.pdf")

######## plot heat balance ###########

result_H = get_result(H, [:technology, :hour])

#result_HeatDump = get_result(HeatDump, [:hour])
#result_HeatDump[!,:technology] .= "heat_dump"

df_demand_heat = DataFrame(hour=T, technology="demand", value=demand_heat)

result_demand = vcat(result_charging, df_demand)


result_H.technology
table_gen = unstack(result_H, :hour, :technology, :value)
table_gen = table_gen[!,[P_CHP..., H_only...]]

table_gen = table_gen[!,[HEAT...]]
labels = names(table_gen) |> permutedims
colors = [colordict[tech] for tech in labels]
data_gen = Array(table_gen)

balance_plot = areaplot(
    data_gen,
    label=labels,
    color=colors,
    width=0,
    leg=:outertopright
)

# Displaying demand
table_dem = unstack(df_demand_heat, :hour, :technology, :value)
table_dem = table_dem[!,["demand"]]
labels2 = names(table_dem) |> permutedims
colors2 = [colordict[tech] for tech in labels2]
replace!(labels2, [item => "" for item in intersect(labels2, labels)]...)
data_dem = -Array(table_dem)

areaplot!(
    balance_plot,
    data_dem,
    label=labels2,
    color=colors2,
    width=0,
    leg=:outertopright
)

hline!(balance_plot, [0], color=:black, label="", width=2)

savefig("results\\Dispatch_Heat.pdf")


#################################


df_installed_gen = get_result(CAP_G, [:technology])
x = df_installed_gen[!,:technology]
y = df_installed_gen[!,:value] ./ 1000
p1 = bar(
    x,
    y,
    leg=false,
    title="Installed power generation",
    ylabel="MW",
    guidefontsize=8,
    rotation=45
)

# df_installed_charge = get_result(CAP_D, [:technology])
# x = df_installed_charge[!,:technology]
# y = df_installed_ch
arge[!,:value] ./ 1000
# p2 = bar(
#     x,
#     y,
#     leg=false,
#     title="Installed power charging",
#     ylim=ylims(p1),
#     rotation=45
# )

# df_installed_storage = get_result(CAP_L, [:technology])
# x = df_installed_storage[!,:technology]
# y = df_installed_storage[!,:value] ./ 1e6

# p3 = bar(
#     x,
#     y,
#     leg=false,
#     title="Installed storage capacity",
#     ylabel="TWh",
#     guidefontsize=8,
#     rotation=45
# )

plot(
    p1,
    # p2,
    #p3,
    layout=(1,3),
    titlefontsize=8,
    tickfontsize=6
)

savefig("results\\Power_generation.pdf")