using JuMP
using Clp
using Plots
using DataFrames, CSV
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
time_series = CSV.read(joinpath(dir, "data\\timedata.csv"), DataFrame)
tech_data = CSV.read(joinpath(dir, "data\\technologies.csv"), DataFrame)

### data preprocessing ###
T_len = time_series.hour |> unique |> length
T = 1:size(time_series, 1) |> collect
P = tech_data[:,:technology] |> Vector
DISP = tech_data[tech_data[!,:dispatchable] .== 1 ,:technology]
NONDISP = tech_data[tech_data[!,:dispatchable] .== 0 ,:technology]
S = tech_data[tech_data[!,:investment_storage] .> 0 ,:technology]
# Heat generation (requires heat balance)
HEAT = tech_data[tech_data[:,:heat_ratio] .!= 0, :technology]
# Import/Export with National Grid
# Z = unique(zones[:,:zone])
# P = zones[:,:id] |> Vector

### parameters ###
annuity_factor(n,r) = r * (1+r)^n / (((1+r)^n)-1)

interest_rate = 0.08
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
    ic_generation_cap[row.technology] = 1000*row.investment_generation * af

    iccc = row.investment_charge * af
    iccc > 0 && (ic_charging_cap[row.technology] = iccc)

    icsc = row.investment_storage * af
    icsc > 0 && (ic_storage_cap[row.technology] = icsc)

    row.storage_efficiency_in > 0 && (eff_in[row.technology] = row.storage_efficiency_in)
    row.storage_efficiency_out > 0 && (eff_out[row.technology] = row.storage_efficiency_out)

    vc[row.technology] = row.vc
end

### time series ###
# demand_elec_res = time_series[:,:demand_elec_res] |> Array
# demand_heat_res = time_series[:,:demand_heat_res] |> Array
# demand_elec_inst = time_series[:,:demand_elec_inst] |> Array
# demand_heat_inst = time_series[:,:demand_heat_inst] |> Array
demand_elec = time_series[:,:demand_elec] |> Array
demand_heat = time_series[:,:demand_heat] |> Array

availability = Dict(nondisp => time_series[:,nondisp] for nondisp in NONDISP)
successor(arr, x) = (x == length(arr)) ? 1 : x + 1

#  month
dispatch_scale = 8760/length(T)

### model ### 
m = Model(Clp.Optimizer)
