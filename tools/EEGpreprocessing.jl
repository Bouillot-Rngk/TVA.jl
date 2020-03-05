module EEGpreprocessing

using StatsBase, Statistics, LinearAlgebra, DSP

# functions:

#resample | rational rate resampling using Kaiser FIR filters


import DSP:resample

export
    standardizeEEG,
    resample

# standardize the whole data in `X` using the winsor mean and std dev.
# the winsor statistics are computed in module StatsBase excluding
# the `prop` proportion of data at both sides.
# If the data elements are not Float64, they are converted to Float64.
function standardizeEEG(X::AbstractArray{T}; prop::Real=0.25) where T<:Real
    vec=X[:]
    μ=mean(winsor(vec; prop=prop))
    σ=√trimvar(vec; prop=prop)
    return eltype(X) === Float64 ? ((X.-μ)./σ) : (convert(Array{Float64}, (X.-μ)./σ))
end



# integer or rational rate resampling using Kaiser FIR filters (from DSP.jl).
# rate is an integer or rational number, e.g., 1//4 downsample by 4
# rel_bw and attenuation are parameter of the FIR filter, see line 637
# in function resample_filter in https://github.com/JuliaDSP/DSP.jl/blob/master/src/Filters/design.jl
# `X` is a data matrix with samples along rows.
# If stim is passes as oka, it must be a vector of as many integers as samples
# in `X`, holding 0 (zero) for samples when there was no stimulation and
# a natural number for samples where there was a stimulation.
# For resampling stimulations, blocks of rate or 1/rate samples are considered
# and if a stimulation appears in those blocks, it is rewritten in the
# first position of the resampled block.
# Examples: Y=resample(X, 1//4); Y=resample(X, 4); Y, newstim=resample(X, 4; stim=s);
function resample(X::Matrix{T},
                  rate::Union{Integer, Rational},
                  rel_bw = 1.0,
                  attenuation = 60;
                  stim::Vector=[]) where T<:Real

    if rate==1 return isempty(stim) ? X : (X, stim) end

    # resample data
    ne = size(X, 2) # of electrodes
    h = DSP.resample_filter(rate, rel_bw, attenuation)
    # first see how long will be the resampled data
    x = DSP.resample(X[:, 1], rate, h)
    t = length(x)
    Y = Matrix{eltype(X)}(undef, t, ne)
    Y[:, 1] = x
    for i=2:ne
        x = DSP.resample(X[:, i], rate, h)
        Y[:, i] = x
    end
    # check that upsampling constructs an extct multiple number of samples
    if rate>1
      diff=size(X, 1)*rate-size(Y, 1)
      if diff>0 Y=vcat(Y, zeros(diff, size(Y, 2))) end
      if diff<0 Y=Y[1:size(X, 1)*rate, :] end
    end

    # resample stimulation channel
    if !isempty(stim)
        l=length(stim)
        if rate<1
            # downsample
            irate=Int(inv(rate))
            s=reshape(stim[1:t*irate], (irate, :))'
            u=[filter(x->x≠0, s[i, :]) for i=1:size(s, 1)]
            for i=1:length(u)
                if length(u[i])>1
                    @error "function `resampling`: the interval between stimulations does not allow the desired downsampling of the stimulation channel" rate
                    return Y, nothing
                end
            end
            newstim=[isempty(v) ? 0 : v[1] for v ∈ u]
        else
            # upsample
            r=vcat(stim, zeros(eltype(stim), l*(rate-1)))
            newstim=reshape(reshape(r, (l, rate))', (l*rate))
        end
        length(newstim)≠size(Y, 1) && @warn "the size of the resampled data and stimulation channel do not match" size(Y, 1) length(newstim)
    end

    return isempty(stim) ? Y : (Y, newstim)
end

end


# useful code:

# of classes, 0 is no stimulation and is excluded
# z=length(unique(stim))-1

# vector with number of stim. for each class 1, 2, ...
# nTrials=counts(stim, z) # or: [count(x->x==i, stim) for i=1:z]
