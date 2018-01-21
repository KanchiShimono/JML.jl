export
    AffineLayer,
    ReluLayer,
    SoftmaxWithLossLayer,
    forward!,
    backward!

abstract type AbstractLayer end

mutable struct AffineLayer{T} <: AbstractLayer
    W::Matrix{T}
    b::Vector{T}
    x::Array{T}
    dW::Matrix{T}
    db::Vector{T}
    function AffineLayer{T}(W::Matrix{T}, b::Vector{T}) where T
        return new{T}(W,b)
    end
end

@inline (::Type{AffineLayer})(W::Matrix{T}, b::Vector{T}) where T = AffineLayer{T}(W, b)

function forward!(lyr::AffineLayer{T}, x::Array{T}) where T
    lyr.x = x
    return lyr.W * x .+ lyr.b
end

function backward!(lyr::AffineLayer{T}, dout::Array{T}) where T
    lyr.dW = dout * lyr.x'
    lyr.db = _sumvec(dout)
    # dx = lyr.W' * dout
    return lyr.W' * dout
end

@inline _sumvec(dout::Vector{T}) where T = dout
@inline _sumvec(dout::Matrix{T}) where T = vec(mapslices(sum, dout, 2))
@inline _sumvec(dout::Array{T,N}) where {T,N} = vec(mapslices(sum, dout, 2:N))


mutable struct ReluLayer <: AbstractLayer
    mask::Array{Bool}
    function ReluLayer()
        return new()
    end
end

function forward!(lyr::ReluLayer, x::Array{T}) where T
    lyr.mask = (x .<= zero(T))
    out = copy(x)
    out[lyr.mask] = zero(T)
    return out
end

function backward!(lyr::ReluLayer, dout::Array{T}) where T
    dout[lyr.mask] = zero(T)
    return dout
end


mutable struct SoftmaxWithLossLayer{T} <: AbstractLayer
    loss::T
    y::Array{T}
    t::Array{T}
    function SoftmaxWithLossLayer{T}() where T
        return new{T}()
    end
end

function forward!(lyr::SoftmaxWithLossLayer{T}, x::Array{T}, t::Array{T}) where T
    lyr.t = t
    lyr.y = softmax(x)
    lyr.loss = crossentropyerror(lyr.y, lyr.t)
    return lyr.loss
end

function backward!(lyr::SoftmaxWithLossLayer{T}, dout::T=one(T)) where T
    return dout .* _swlvec(lyr.y, lyr.t)
end

@inline _swlvec(y::Array{T}, t::Vector{T}) where T = y .- t
@inline _swlvec(y::Array{T}, t::Matrix{T}) where T = (y .- t) / size(t)[2]
