module ERPs

using LinearAlgebra, Statistics

import Statistics: mean

export
   mean


# create the Toeplitz matrix for estimating ERP means via
# multivariate regression. See congedo et al.(2016)
# `ns` is the number of samples
# `wl` is the window length, i.e., the ERPs duration, in samples
# `cstim` is a vector of c integer vectors, where c is the number of classes,
#         holding all samples where there is a stimulation (1, 2...)
# Optional keyword arguments:
# `offset` is an offset for determining that trial starting sample
#         with respect to the samples in `cstim`. Can be zero, positive or negative.
# `weights` is a vector of non-negative real weights for the trials
#         corresponding to the stimulations in `cstim`.
#         The weights must be appropriately normalized.
#         By default, no weights are used.
function toep(ns::Int, wl::Int, cstim::Vector{Vector{S}};
              offset :: Int = 0,
              weights:: Union{Vector{Vector{R}}, Symbol}=:none) where R<:Real where S<:Int
  T=zeros(wl*length(cstim), ns)
  nc=length(cstim)
  if     weights==:none
           @inbounds for c=1:nc, j=1:wl, i=1:length(cstim[c])
                     T[(c-1)*wl+j, cstim[c][i]+j-1+offset]=1 end
  else
           @inbounds for c=1:nc, j=1:wl, i=1:length(cstim[c])
                    T[(c-1)*wl+j, cstim[c][i]+j-1+offset]= weights[c][i] end
  end
  return T
end


# estimate the mean ERPs using the arithmetic mean if `overlap` is false
# or the multivariate regression method explained # in Congedo et al.(2016)
# if `overlap` is true.
# `X` is the whole EEG recording of size # of samples x # of electrodes
# `wl` is the window length, i.e., the ERPs duration, in samples
# `cstim` is a vector of c integer vectors, where c is the number of classes,
#         holding all samples where there is a stimulation (1, 2...)
# Optional keyword arguments:
# `offset` is an offset for determining that trial starting sample
#         with respect to the samples in `cstim`. Can be zero, positive or negative.
# `weights` is a vector of non-negative real weights for the trials
#         corresponding to the stimulations in `cstim`. They don't need to be
#         normalized. If `weights=:a` is passed, adaptive weights are
#         computed as the inverse of the squared Frobenius norm of the
#         trials data, along the lines of Congedo et al. (2016).
#         By default, no weights are used.
function mean(X::Matrix{R}, wl::Int, cstim::Vector{Vector{S}}, overlap::Bool=false;
         offset :: Int = 0,
         weights:: Union{Vector{Vector{R}}, Symbol}=:none) where R<:Real where S<:Int

  if overlap
    if      weights==:none
            T=toep(size(X, 1), wl, cstim; offset=offset)
            Xbar=inv(T*T') * (T*X)
    else
            weights==:a ? weights=[[1/(norm(X[cstim[i][j]:cstim[i][j]+wl-1,:])^2) for j=1:length(cstim[i])] for i=1:length(cstim)] : nothing
            T=toep(size(X, 1), wl, cstim; offset=offset, weights=weights)
            w=[sum(weight.^2)/sum(weight) for weight in weights]
            Xbar=Diagonal(vcat(fill.(w, wl)...)) * (inv(T*T') * (T*X))
    end
    return [Xbar[wl*(c-1)+1:wl*c, :] for c=1:length(cstim)]
  else
    if      weights==:none
            return [mean(X[cstim[c][j]+offset:cstim[c][j]+offset+wl-1, :] for j=1:length(cstim[c])) for c=1:length(cstim)]
    else
            weights==:a ? weights=[[1/(norm(X[cstim[i][j]:cstim[i][j]+wl-1,:])^2) for j=1:length(cstim[i])] for i=1:length(cstim)] : nothing
            w=[weight./sum(weight) for weight in weights]
            return [sum(X[cstim[c][j]+offset:cstim[c][j]+offset+wl-1, :]*w[c][j] for j=1:length(cstim[c])) for c=1:length(cstim)]
    end
  end
end



end # module


#=
push!(LOAD_PATH, homedir()*"\\Documents\\Code\\julia\\Modules")
using EEGpreprocessing, EEGio, EEGtopoPlot, System

X=Matrix(readASCII("C:\\temp\\data")')

XTt=Matrix(readASCII("C:\\temp\\XTt")')

stims=[Vector{Int64}(readASCII("C:\\temp\\stim1")[:]), Vector{Int}(readASCII("C:\\temp\\stim2")[:]) ]

A=mulTX(X, stims, 128)

using LinearAlgebra
norm(A-XTt)
=#
