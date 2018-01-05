export
    AffineLayer,
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

AffineLayer(W::M, b::V) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}} = AffineLayer{T,M,V}(W, b)
AffineLayer(W::M, b::V, ::Type{A}) where {T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},A<:AbstractArray{T}} = AffineLayer{T,M,V,A}(W, b)

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
