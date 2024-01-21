
generic_names(len::Int) = [Symbol("x$i") for i in 1:len]

function get_result(x::JuMP.Containers.DenseAxisArray,
    names::AbstractVector{Symbol}=generic_names(length(x.axes));
    kwargs...)

    if length(names) != length(x.axes)
        throw(ArgumentError("Length of argument 'names' is $(length(names)) and
        does not fit the variable's dimension of $(length(x.axes))"))
    end
    push!(names, :value)

    iter = Iterators.product(x.axes...) |> collect |> vec
    df = DataFrame([(i..., value(x[i...])) for i in iter])
    rename!(df, names)
    return df
end