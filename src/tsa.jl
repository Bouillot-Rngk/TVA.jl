module tsa
using LinearAlgebra, PosDefManifold, PosDefManifoldML, CovarianceEstimation,
      Dates, Distributions, PDMats, Revise, BenchmarkTools, Diagonalizations,
      Random, DataFrames, CSV, Plots, JLD
using tva

push!(LOAD_PATH,homedir()*"/Julia/MultiProcessing.jl/src")
using EEGio, FileSystem, EEGpreprocessing, System, ERPs,
    Tyler, EEGtomography, Processdb,MPTools


export getTV,
	   alignTVacc,
	   eegTangentVector

function eegTangentVector(o; estimator=:Tyler, verbose=false, sendCov=false, ncomp=4)
	# Target and Non-target labels
	TAlabel = findfirst(isequal("Target"), o.clabels)
	NTlabel = findfirst(isequal("NonTarget"), o.clabels)

	# PCA to keep only 4 components
	X̄ = mean(o.X, o.wl, o.cstim, true; weights=:a)[TAlabel]
	X̃ₘ = X̄ * eigvecs(cov(SimpleCovariance(), X̄))[:, o.ne-ncomp+1:o.ne]

	if estimator == :Tyler
		# normalized and regularized Tyler M-estimator
		C = ℍVector([ℍ(Tyler.nrtme([X X̃ₘ]'; reg=:LW, verbose=verbose)) for X ∈ o.trials])
	else
		# Ledoit-Wolf shrinkage
		C = ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X X̃ₘ])) for X ∈ o.trials])
	end

	# Barycenter of the covariance matrices
	C̄, _, _ = gMean(ℍVector([C[i] for i in eachindex(C) if o.y[i]==TAlabel]); verbose=verbose)

	# Projection on tangent space
	W = invsqrt(C̄)
	𝔾 = ℍVector([ℍ(W*C[i]*W') for i in eachindex(C)])
	return sendCov ? (hcat([vecP(log(G), range=1:o.ne) for G ∈ 𝔾]...), 𝐂) : hcat([vecP(log(G), range=1:o.ne) for G ∈ 𝔾]...)
end

####Data formatting####

function getTV(db, target_s, source_s)
    ### From tsalign.jl with MPTools modifications
	Dir = homedir()*"/Documents/My Data/EEG data/npz/BI.EEG.2013-Sorted"
	dbSearch = Dir*"/Sujet$db/Base$db/"
    files = loadNYdb(dbSearch)
    odt = readNY(files[target_s]; bandpass = (1,16))
    ods = [readNY(files[s]; bandpass = (1,16)) for s ∈ source_s]
    vDT = eegTangentVector(odt, verbose=false, ncomp=4, estimator="Wolf")
    vDS = [eegTangentVector(s, verbose=false, ncomp=4, estimator="Wolf") for s ∈ ods]

    d = size(vDT, 1)
    TAlabel = findfirst(isequal("Target"), odt.clabels)
    NTlabel = findfirst(isequal("NonTarget"), odt.clabels)
    minNT = min(cat(length(odt.cstim[NTlabel]), [length(o.cstim[NTlabel]) for o ∈ ods], dims=1)...)
	minTA = min(cat(length(odt.cstim[TAlabel]), [length(o.cstim[TAlabel]) for o ∈ ods], dims=1)...)

	xDT=zeros(eltype(odt.X), d, minNT+minTA) # vDT
	xDS=[copy(xDT) for _ ∈ ods] # vDS
	xDT[:, 1:minNT] = vDT[:, 1:minNT]
	xDT[:, minNT:end] = vDT[:, size(vDT, 2)-minTA:end]
	for (x, v) ∈ zip(xDS, vDS)
		x[:, 1:minNT] = v[:, 1:minNT]
		x[:, minNT:end] = v[:, size(v, 2)-minTA:end]
	end
	𝕏 = [xDT, xDS...]
	𝕪 = [repeat([NTlabel], minNT); repeat([TAlabel], minTA)]
	return 𝕏, 𝕪

end #getTV

####
function alignTVacc(𝕏, 𝕪, source_s, align=true; gmca_args=nothing)
	#### Everyone is aligned with everyone
	if align
		gm = gmca(𝕏)

		𝕏aligned = [gm.F[i]'*𝕏[i] for i=1:length(𝕏)]
		𝕏Tr = vcat([Matrix(𝕏aligned[i]') for i in range(2, stop=length(source_s)+1)]...)
		𝕪Tr = repeat(𝕪, 2)
		model = fit(ENLR(Fisher), 𝕏Tr, 𝕪Tr)
		𝕏Te = Matrix(𝕏aligned[1]')
	else
		𝕏Tr = vcat([Matrix(𝕏[i]') for i in range(2, stop=length(source_s)+1)]...) # Training data sans rotation
		𝕪Tr = repeat(𝕪, 2)
		model = fit(ENLR(Fisher), 𝕏Tr, 𝕪Tr)
		𝕏Te = Matrix(𝕏[1]')
	end
	𝕪Pr = predict(model, 𝕏Te)
	predErr = predictErr(𝕪, 𝕪Pr)
	acc = 100. - predErr
	return acc
end


end #module
