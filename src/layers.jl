export
    AffineLayer,
    ReluLayer,
    SoftmaxWithLossLayer,
    forward!,
    backward!

abstract type AbstractLayer end

mutable struct AffineLayer{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},A<:AbstractArray{T}} <: AbstractLayer
    W::M
    b::V
    x::A
    dW::M
    db::V
    function AffineLayer{T,M,V,A}(W::M, b::V) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},A<:AbstractArray{T}}
        return new{T,M,V,A}(W,b)
    end
    function AffineLayer{T,M,V}(W::M, b::V) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
        return new{T,M,V,AbstractArray{T}}(W, b)
    end
end

@inline (::Type{AffineLayer})(W::M, b::V) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}} = AffineLayer{T,M,V}(W, b)
@inline (::Type{AffineLayer})(W::M, b::V, ::Type{A}) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},A<:AbstractArray{T}} = AffineLayer{T,M,V,A}(W, b)

function forward!(lyr::AffineLayer{T,M,V,A}, x::A) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},A<:AbstractArray{T}}
    lyr.x = x
    return lyr.W * x .+ lyr.b
end

function backward!(lyr::AffineLayer{T,M,V,A}, dout::A) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},A<:AbstractArray{T}}
    lyr.dW = dout * lyr.x'
    lyr.db = _sumvec(dout)
    # dx = lyr.W' * dout
    return lyr.W' * dout
end

@inline _sumvec(dout::AbstractVector{T}) where {T<:Real} = dout
@inline _sumvec(dout::AbstractMatrix{T}) where {T<:Real} = vec(mapslices(sum, dout, 2))
@inline _sumvec(dout::AbstractArray{T,N}) where {T<:Real,N} = vec(mapslices(sum, dout, 2:N))


mutable struct ReluLayer{B<:AbstractArray{Bool}} <: AbstractLayer
    mask::B
    function ReluLayer{B}() where B <: AbstractArray{Bool}
        return new{B}()
    end
end

@inline (::Type{ReluLayer})() = ReluLayer{AbstractArray{Bool}}()

function forward!(lyr::ReluLayer{B}, x::A) where {T<:Real,B<:AbstractArray{Bool},A<:AbstractArray{T}}
    lyr.mask = (x .<= zero(T))
    out = copy(x)
    out[lyr.mask] = zero(T)
    return out
end

function backward!(lyr::ReluLayer{B}, dout::A) where {T<:Real,B<:AbstractArray{Bool},A<:AbstractArray{T}}
    dout[lyr.mask] = zero(T)
    return dout
end


mutable struct SoftmaxWithLossLayer{T,A<:AbstractArray{T}} <: AbstractLayer
    loss::T
    y::A
    t::A
    function SoftmaxWithLossLayer{T,A}() where {T<:Real,A<:AbstractArray{T}}
        return new{T,A}()
    end
end

function forward!(lyr::SoftmaxWithLossLayer{T,A}, x::A, t::A) where {T<:Real,A<:AbstractArray{T}}
    lyr.t = t
    lyr.y = softmax(x)
    lyr.loss = crossentropyerror(lyr.y, lyr.t)
    return lyr.loss
end

function backward!(lyr::SoftmaxWithLossLayer{T,A}, dout::T=one(T)) where {T<:Real,A<:AbstractArray{T}}
    return dout .* _swlvec(lyr.y, lyr.t)
end

@inline _swlvec(y::AbstractArray{T}, t::AbstractVector{T}) where T <: Real = y .- t
@inline _swlvec(y::AbstractArray{T}, t::AbstractMatrix{T}) where T <: Real = (y .- t) / size(t)[2]
