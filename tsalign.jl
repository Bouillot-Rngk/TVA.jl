Dir = homedir()*"\\Documents\\ENSE3\\Thesis"
push!(LOAD_PATH,Dir*"\\Julia\\TVA.jl\\src")
push!(LOAD_PATH,Dir*"/Julia/MultiProcessing.jl/src")


using LinearAlgebra, PosDefManifold, PosDefManifoldML, CovarianceEstimation,
      Dates, Distributions, PDMats, Revise, BenchmarkTools, Diagonalizations,
      Random, DataFrames, CSV, Plots, JLD


using EEGio, FileSystem, EEGpreprocessing, System, ERPs,
      Tyler, EEGtomography


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
- sendCov is Bool = false
"""
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
	evcca = Matrix{Float64}(undef, d_, length(files))
	⌚ = verbose && now()

	for (i, file) ∈ enumerate(files)
		os_ = readNY(files[i];  bandpass=(1, 16))
		verbose && println("file ", i, ", target ", target_s)
		if os_.subject ≠ ot_.subject
			verbose && println("subject ", os_.subject, ", target ", ot_.subject)
			𝕏, _ = getTV(db, target_s, i; verbose=verbose, ncomp=ncomp, estimator=estimator)
			model = cca(𝕏[1], 𝕏[2]; dims=2, simple=true)
			evcca[:, i] = diag(model.D)
		else
			evcca[:, i] = zeros(d_)
		end
		waste(ot_)
	end
	verbose && println("Estimating CCA done in ", now()-⌚)
	return sortperm(evcca[1, :]), evcca
end

"""
Return average accuracy for intrasession of subject

Parameters:
- o is EEG
- nFolds is Int for k-fold cross-validation
- shuffle is Bool
"""
function intraSessionAccuracy(o; nFolds=10, shuffle=false)
	# multivariate regression ERP mean with data-driven weights
    w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]

    # PCA to keep only 4 components
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
    Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]

	# Ledoit-Wolf
    𝐂lw=ℍVector([ℍ(cov(SimpleCovariance(), [X Y])) for X ∈ o.trials])

	# det normalization and regu
	for i=1:length(𝐂lw) 𝐂lw[i]=det1(𝐂lw[i]) end

	R=Hermitian(Matrix{eltype(𝐂lw[1])}(I, size(𝐂lw[1]))*0.0001)
	for C in 𝐂lw C+=R end
	out = 0.4
    # classification
	try
	    args=(shuffle=shuffle, tol=1e-6, verbose=false, nFolds=nFolds, ⏩=false)
	    cvlw = cvAcc(ENLR(Fisher), 𝐂lw, o.y; args...)
		out = cvlw.avgAcc
	catch
		out = 0.5
	end
	print("\n")

    return out
end

"""
Return subject ordered by intrasession accuracy

Parameters:
- db is path to EEG dataset
- nFolds is Int for number of k-fold cross-validation
- verbose is Bool
"""
function findBestTargetSubjects(db; nFolds=10, verbose=true)
	files = loadNYdb(db)
	⌚ = verbose && now()
	subjAcc = [intraSessionAccuracy(readNY(files[i];  bandpass=(1, 16)), nFolds=nFolds) for i in eachindex(files)]
	bestTargetIdx = sortperm(subjAcc; rev=true)
	verbose && println("Estimating intrasession accuracy done in ", now()-⌚)
	return bestTargetIdx, subjAcc
end

# Global variables

db = homedir()*"/Documents/My Data/EEG data/npz/P300/BI.EEG.2012-GIPSA"
df = DataFrame(accuracy = Float32[],
               aligned = String[], # GMCA, GCCA, no
		   classifier = String[], # ENLR, MDM
		   dataset = String[],
		   targetSubject = Int[],
		   nbSubjSource = Int[],
		   ncomp = Int[])
ncomp = 4
nbSubjSource = 2
nbSubjTarget = 20
nFolds = 10
gmca_args = (eVar=0.9, verbose=true, tol=10e-6, dims=2, algorithm=:OJoB)
# target_s = 30

# Rank subjects based on their intra-sesssion accuracy
if isfile("./Data/best_target.jld")
    o = load("./best_target.jld")
    bestTargetIdx = o["bestTargetIdx"]
    subjAcc = o["subjAcc"]
    println("Best subject target already computed, using previous results")
else
    bestTargetIdx, subjAcc = findBestTargetSubjects(db; nFolds=nFolds, verbose=true)
    save("./best_target.jld", "bestTargetIdx", bestTargetIdx, "subjAcc", subjAcc)
end

for target_s in bestTargetIdx[1:nbSubjTarget]
    if isfile("./bestSources-target"*string(target_s)*".jld")
        o = load("./bestSources-target"*string(target_s)*".jld")
        idx_bestsources, evcca = o["idx_bestsources"], o["evcca"]
        println("Best source subjects already computed for subject "*string(target_s)*", using previous results")
    else
        idx_bestsources, evcca = findBestSourceSubjects(db, target_s, ncomp; verbose=true, estimator=:lw)
        save("./bestSources-target"*string(target_s)*".jld", "idx_bestsources", idx_bestsources, "evcca", evcca)
    end
#=    plot(evcca, title="Subj "*string(target_s));
    savefig("EVcca-s"*string(target_s)*".png")
	 bar(evcca[1, :], ylims=(0.95, 1.0), title="Subj "*string(target_s));
    savefig("largestEVcca-s"*string(target_s)*".png")

	 =# #source_s = (21, 22)
	 source_s = idx_bestsources[1:nbSubjSource]
	 println("Best source subjects are ", source_s)

	 𝕏, 𝕪 = getTV(db, target_s, source_s; verbose=false)
	 acc = alignTVacc(𝕏, 𝕪, source_s, true, gmca_args=gmca_args)
	 push!(df, Dict(:accuracy => acc, :aligned =>"GMCA", :classifier => "ENLR",
                   :dataset => "Gipsa2012", :targetSubject => target_s,
		   		    :nbSubjSource => length(source_s), :ncomp => ncomp))

	 accNR = alignTVacc(𝕏, 𝕪, source_s, false)
	 push!(df, Dict(:accuracy => accNR, :aligned =>"none", :classifier => "ENLR",
	                :dataset => "Gipsa2012", :targetSubject => target_s,
    			       :nbSubjSource => length(source_s), :ncomp => ncomp))
    CSV.write("results_gmca.csv", df)
end
# CSV.write("results_gmca.csv", df)
