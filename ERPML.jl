module ERPML

using LinearAlgebra, PosDefManifold, PosDefManifoldML, CovarianceEstimation,
      Dates, Distributions, PDMats, Revise, BenchmarkTools, Diagonalizations,
      Random

push!(LOAD_PATH, homedir()*"/src/julia/Modules")
using EEGio, FileSystem, EEGpreprocessing, System, ERPs, Tyler, EEGtomography

export eegERPCovariances, eegTangentVector, getTV, getCov, alignTVacc, baselineTransfer, findBestSourceSubjects, findBestTargetSubjects, intraSessionAccuracy
"""
Return covariance matrices from the EEG of a subject

The covariance matrices are estimated with a normalized regularized Tyler's M-estimator or a Ledoit-Wolf estimator.

Parameters:
- o is EEG
- estimator is String, default value :lw (Ledoit-Wolf) or :Tyler
- ncomp is int for number of components
- verbose is Bool = false,
- normalization is String, default value is nothing, or :det or :tr
"""
function eegERPCovariances(o; estimator=:lw, verbose=false, ncomp=4, normalization=nothing)
    TAlabel = findfirst(isequal("Target"), o.clabels)
    NTlabel = findfirst(isequal("NonTarget"), o.clabels)

    # PCA to keep only 4 components
    X̄ = mean(o.X, o.wl, o.cstim, true; weights=:a)[TAlabel]
    X̃ₘ = X̄ * eigvecs(cov(SimpleCovariance(), X̄))[:, o.ne-ncomp+1:o.ne]
    if estimator == :Tyler
        # normalized and regularized Tyler M-estimator
        𝐂 = ℍVector([ℍ(Tyler.nrtme([X X̃ₘ]'; reg=:LW, verbose=verbose)) for X ∈ o.trials])
    else
        # Ledoit-Wolf shrinkage
        𝐂 = ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X X̃ₘ])) for X ∈ o.trials])
    end
    if normalization == :det
        # det normalization
        for i=1:length(𝐂) 𝐂[i]=det1(𝐂[i]) end
    elseif normalization == :tr
        ### trace normalization
        for i=1:length(𝐂) 𝐂[i]=𝐂[i]/tr(𝐂[i]) end
    end
    return 𝐂
end

"""
Return tangent vectors from the EEG of a subject

The tangent vectors are obtained from a tangent mapping
at the geometric mean of the session. The covariance matrices
are estimated with a normalized regularized Tyler's M-estimator.

Parameters:
- o is EEG
- estimator is String, default value :Tyler or :lw (Ledoit-Wolf)
- ncomp is int for number of components
- verbose is Bool = false,
"""
function eegTangentVector(o; estimator=:Tyler, verbose=false, ncomp=4)
    # Target and Non-target labels
    TAlabel = findfirst(isequal("Target"), o.clabels)
    NTlabel = findfirst(isequal("NonTarget"), o.clabels)
    𝐂 = eegERPCovariances(o, estimator=estimator, verbose=verbose, ncomp=ncomp, normalization=:det)

    𝕋, _ = tsMap(Fisher, 𝐂)
    return transpose(𝕋)
end

"""
Return tangent vectors for target and source subjects

Parameters:
- db is path to EEG dataset
- target_s is int coding for target subject
- source_s is array{Int} for source subjects
- verbose is Bool = false
- estimator is String, default value :Tyler or :lw (Ledoit-Wolf)
- ncomp is int for number of components (default: 4)
"""
function getTV(db, target_s, source_s; verbose=false, estimator=:Tyler, ncomp=4)
    files = loadNYdb(db)
    odt  = readNY(files[target_s];  bandpass=(1, 16))
    ods = [readNY(files[s]; bandpass=(1, 16)) for s ∈ source_s]
    vDT = eegTangentVector(odt, verbose=verbose, ncomp=ncomp, estimator=estimator)
    vDS = [eegTangentVector(s, verbose=verbose, ncomp=ncomp, estimator=estimator) for s ∈ ods]

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
end

"""
Return covariance matrices for target and source subjects

Parameters:
- db is path to EEG dataset
- target_s is int coding for target subject
- source_s is array{Int} for source subjects
- verbose is Bool = false
- estimator is String, default value :Tyler or :lw (Ledoit-Wolf)
- ncomp is int for number of components (default: 4)
"""
function getCov(db, target_s, source_s; verbose=false, estimator=:Tyler, ncomp=4)
	files = loadNYdb(db)
    odt  = readNY(files[target_s];  bandpass=(1, 16))
    ods = [readNY(files[s]; bandpass=(1, 16)) for s ∈ source_s]
    𝐂DT = eegERPCovariances(odt, verbose=verbose, ncomp=ncomp, estimator=estimator)
    𝐂DS = [eegERPCovariances(s, verbose=verbose, ncomp=ncomp, estimator=estimator) for s ∈ ods]

    d = size(𝐂DT, 1)
    TAlabel = findfirst(isequal("Target"), odt.clabels)
    NTlabel = findfirst(isequal("NonTarget"), odt.clabels)
    minNT = min(cat(length(odt.cstim[NTlabel]), [length(o.cstim[NTlabel]) for o ∈ ods], dims=1)...)
    minTA = min(cat(length(odt.cstim[TAlabel]), [length(o.cstim[TAlabel]) for o ∈ ods], dims=1)...)

    xDT = [𝐂DT[1:minNT] ; 𝐂DT[size(𝐂DT, 1)-minTA+1:end]]
    xDS = [[𝐂DS[i][1:minNT] ; 𝐂DS[i][size(𝐂DS[i], 1)-minTA+1:end]] for i in eachindex(𝐂DS)]

    ℙ = [xDT, xDS...]
    𝕪 = [repeat([NTlabel], minNT); repeat([TAlabel], minTA)]
    return ℙ, 𝕪
end

"""
Return transfer learning accuracy

From a target subject, whom tangent vectors are stored as the
first element of 𝕏 array, and tangent vectors of source subjects,
stored next in 𝕏, this function align subjects with Generalized
Maximum Covariance Analysis if the flag align is true and classify
tangent vectors with ENLR. This function return the model accuracy.

Parameters:
- 𝕏 is array of tangent vectors,
- 𝕪 is array of labels,
- source_s is Int, indexes of sources subjects,
- align is Bool, to align sources and target with GMCA (default is true);
- gmca_args... are arguments for GMCA
"""
function alignTVacc(𝕏, 𝕪, source_s, align=true; gmca_args=nothing)
    if align
        gm = gmca(𝕏; gmca_args...)

        𝕏aligned = [gm.F[i]'*𝕏[i] for i=1:length(𝕏)]
        𝕏Tr = vcat([Matrix(𝕏aligned[i]') for i in range(2, stop=length(source_s)+1)]...)
        𝕏Te = Matrix(𝕏aligned[1]')
    else
        𝕏Tr = vcat([Matrix(𝕏[i]') for i in range(2, stop=length(source_s)+1)]...)
        𝕏Te = Matrix(𝕏[1]')
    end
    𝕪Tr = repeat(𝕪, length(source_s))
    model = fit(ENLR(Fisher), 𝕏Tr, 𝕪Tr)
    𝕪Pr = predict(model, 𝕏Te)
    predErr = predictErr(𝕪, 𝕪Pr)
    acc = 100. - predErr
    return acc
end

"""
Return transfer learning accuracy for covariance matrices

From a target subject, whom covariance matrices are stored as the
first element of ℙ array, and tangent vectors of source subjects,
stored next in ℙ, this function classify covariance matrices with
provided model. This function return the model accuracy.

Parameters:
- ℙ is array of tangent vectors,
- 𝕪 is array of labels,
- source_s is Int, indexes of sources subjects;
- model is a model from PosDefManifoldML
"""
function baselineTransfer(ℙ, 𝕪, source_s; model=ENLR(Fisher))
    ℙTr = vcat([ℙ[i] for i in range(2, stop=length(source_s)+1)]...)
    𝕪Tr = repeat(𝕪, length(source_s))
    model = fit(model, ℙTr, 𝕪Tr)
    ℙTe = ℙ[1]
    𝕪Pr = predict(model, ℙTe)
    predErr = predictErr(𝕪, 𝕪Pr)
    acc = 100. - predErr
    return acc
end

"""
Return  list of source subjects ordered by alignment with target

The alignment is estimated with the eigenvalues of the CCA computed
on tangent vectors.

Parameters:
- db is path to EEG dataset
- target_s is int, target subject id
- ncomp is Int, number of mean component for extended signal
- estimator is String, default value :lw (Ledoit-Wolf) or :Tyler
- verbose is Bool
"""
function findBestSourceSubjects(db, target_s, ncomp; verbose=false, estimator=:lw)
    files = loadNYdb(db)
    ot_ = readNY(files[target_s];  bandpass=(1, 16))
    vot_ = eegTangentVector(ot_, verbose=false, ncomp=ncomp, estimator=:lw)
    d_ = size(vot_, 1)
    # evcca = Matrix{Float64}(undef, d_, length(files))
    evcca = zeros(Float64, d_, length(files))
    ⌚ = verbose && now()
    for (i, file) ∈ enumerate(files)
        os_ = readNY(files[i];  bandpass=(1, 16))
        verbose && println("file ", i, ", target ", target_s)
        if os_.subject ≠ ot_.subject
            verbose && println("subject ", os_.subject, ", target ", ot_.subject)
            𝕏, _ = getTV(db, target_s, i; verbose=verbose, ncomp=ncomp, estimator=estimator)
            model = cca(𝕏[1], 𝕏[2]; dims=2, simple=false)
            evcca[1:size(model.D)[1], i] = diag(model.D)
        end
        waste(ot_)
    end
    verbose && println("Estimating CCA done in ", now()-⌚)
    return sortperm(evcca[1, :], rev=true), evcca
end

"""
Return average accuracy for intrasession of subject

Parameters:
- o is EEG
- nFolds is Int for k-fold cross-validation
- shuffle is Bool
"""
function intraSessionAccuracy(o; nFolds=10, shuffle=false, ncomp=4)
    𝐂lw = eegERPCovariances(o, estimator=:lw, ncomp=ncomp, normalization=:det)
    # classification
    args=(shuffle=shuffle, tol=1e-6, verbose=false, nFolds=nFolds, lambda_min_ratio=1e-4)
    cvlw = cvAcc(ENLR(Fisher), 𝐂lw, o.y; args...)

    return cvlw.avgAcc
end

"""
Return subject ordered by intrasession accuracy

Parameters:
- db is path to EEG dataset
- nFolds is Int for number of k-fold cross-validation
- verbose is Bool
"""
function findBestTargetSubjects(db; nFolds=10, verbose=true, ncomp=4)
    files = loadNYdb(db)
    ⌚ = verbose && now()
    subjAcc = [intraSessionAccuracy(readNY(files[i];  bandpass=(1, 16)), nFolds=nFolds, ncomp=ncomp) for i in eachindex(files)]
    bestTargetIdx = sortperm(subjAcc, rev=true)
    verbose && println("Estimating intrasession arevccuracy done in ", now()-⌚)
    return bestTargetIdx, subjAcc
end



"""
Return a vector of covariances matrices with the same
amount of target and non-target elements

Parameters :
- ℙ is a vector of covariance matrices
- 𝕪 is the vector of labels vectors
"""

function balanceERP(ℙ :: HermitianVector , 𝕪:: Vector{IntVector} )


end #balanceERP

"""
Return a vector of tangent vectors with the same
amount of target and non-target elements

Parameters :
- 𝕏 is a vector of Tangent Vectors
- 𝕪 is the vector of labels vectors
"""

function balanceERP(𝕏 :: Vector{Matrix} , 𝕪 :: Vector{IntVector})
	
end #balanceERP

end # Module
