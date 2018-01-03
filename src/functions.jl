export
    sigmoid,
    relu,
    softmax,
    onehot,
    crossentropyerror

function sigmoid(x::T) where T <: Real
    return 1.0 / (1.0 + exp(-x))
end

function relu(x::T) where T <: Real
    return max(zero(x), x)
end

function softmax(x::AbstractVector{T}) where T <: Real
    c = maximum(x)
    exp_a = exp.(x .- c)
    return exp_a ./ sum(exp_a)
end

function softmax(x::AbstractMatrix{T}) where T <: Real
    return mapslices(softmax, x, 1)
end

function onehot(::Type{T}, t::AbstractVector, l::AbstractVector) where T
    r = zeros(T, length(l), length(t))
    for i in 1:length(t)
        r[findfirst(l, t[i]), i] = 1
    end
    return r
end

onehot(t::AbstractVector, l::AbstractVector) = onehot(Int, t, l)

function crossentropyerror(y::Vector, t::Vector)
    δ = 1.0e-7
    # -sum(t .* log.(y .+ δ))
    return -(t ⋅ log.(y .+ δ))
end

function crossentropyerror(y::Matrix, t::Matrix)
    batch_size = size(y, 2)
    δ = 1.0e-7
    # -sum(t .* log.(y .+ δ)) ./ batch_size
    return -vecdot(t, log.(y .+ δ)) ./ batch_size
end
