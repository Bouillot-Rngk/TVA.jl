
Dir = homedir()*"\\Documents\\Etienne"


push!(LOAD_PATH,Dir*"\\git\\MP\\src") #Pour recuperer les modules EEGio etc
push!(LOAD_PATH,Dir*"\\data\\BI.EEG.2012-GIPSA") #emplacement de la DB

using LinearAlgebra, PosDefManifold, PosDefManifoldML, CovarianceEstimation,
      Dates, Distributions, PDMats, Revise, BenchmarkTools, Diagonalizations,
      Random, DataFrames, CSV, Plots, JLD   # + NPZ, YAML, StatsBase, DSP, libGR

using EEGio, FileSystem, EEGpreprocessing, System, ERPs,
      Tyler, EEGtomography, MPTools


#### Compute Alignement covariances ####

#Charmgements des données obrtenues
o = load("data\\heatmap.jld")
heatmap = o["Heatmap"]
idAlign = o["spFromResults"]


#Nouveaux calculs, et affichage
heatmap, idAlign = alignement()
plotHeatmap(heatmap,idAlign)
save("data\\heatmap.jld", "Heatmap", heatmap, "spFromResult", idAlign)

####Useful function ####

function plotHeatmap(heatmap, idAlign)
	Cmax = maximum(abs.(heatmap))
	Imax = maximum(abs.(idAlign))
	gr()
	# Ici on ne prend pas les 4 derniers fichiers car la CV n'a pas pu être calculée
	h1 = Plots.heatmap(1:size(heatmap,1)-4,1:size(heatmap,2)-4, heatmap[1:end-4,1:end-4],
	    clim = (50, Cmax),c=:bluesreds,aspect_ratio=1,xlim=(1,21),ylim=(1,21),yflip=true,
	    title="MCA between calibration files :\n fit on target_calibration only,\n predict on source_test only");
	h2 = Plots.heatmap(1:size(idAlign,1)-4, 1:size(idAlign,2)-4, idAlign[1:end-4,1:end-4],
		clim = (0.95, 1), c=:bluesreds, aspect_ratio=1, yflip=true,
		xlim=(1,21),ylim=(1,21),
		title="Confidence Index of alignement \n(1 is perfect alignement)");
	plot(h1,h2, size = (700,400))
end #plotHeatmap

function getData()
	bestTargetIdx, subjAcc = getBestTargets(1)
	files, base =  load1()
	dbsorted = [files[i] for i in bestTargetIdx]

	EEGInfo = [readNY(i) for i in files]
	dbsorted = [EEGInfo[i] for i in bestTargetIdx]
	infos = [getSession(i) for i in dbsorted]
	return infos, subjAcc, base
end #getData

function getBestSubjects()
	infos, subjAcc, base = getData()
	df = DataFrame(Subject = [s[1] for s in infos], Session = [s[2] for s in infos], Run = [s[3] for s in infos], Acc = subjAcc)

	Sub = unique(df[:, :Subject])
	moy = Vector{Float64}(undef, length(Sub))

	subset = df[∈([24]).(df.Subject), :]

	for i in 1:length(Sub)
		subset =  df[∈([Sub[i]]).(df.Subject), :]
		moy[i] = mean(subset[!,:Acc])
	end

	dfSub = DataFrame(Subject = Sub, Acc = moy)
	sort!(dfSub, [:Acc]; rev = true)
	bestTargetSubjectmean = dfSub[!,:Subject]
	subjAccmean = dfSub[!,:Acc]

	return bestTargetSubjectmean, subjAccmean
end #getBestSubjects


function getBestTargets(db;
                        nFolds = 10,
                        verbose=true)
  #-------------------------------------------------------------------------------#
  #               Rank subjects based on their intra-sesssion accuracy            #

	if isfile("data\\best_target.jld")
    #Useless to compute the accuracy if it's already done
	    o = load("data\\best_target.jld")
	    bestTargetIdx = o["bestTargetIdx"]
	    subjAcc = o["subjAcc"]
	    println("Best subject target already computed, using previous results")
	else
	    files, base = load1()
	    files = getFiles(('1','1',('1','2')))
	    ⌚ = verbose && now()
	    subjAcc = [intraSessionAccuracy(readNY(files[i];  bandpass=(1, 16)), nFolds=nFolds) for i in eachindex(files)]
	  	bestTargetIdx = sortperm(subjAcc; rev=true)
	  	verbose && println("Estimating intrasession accuracy done in ", now()-⌚)
	    save("data\\best_target.jld", "bestTargetIdx", bestTargetIdx, "subjAcc", subjAcc)
  	end
  return bestTargetIdx, subjAcc;
end #getBestTargets

function getSession(o)
  Subject = o.subject
  Session = o.session
  Run = o.run
  return info = (Subject, Session, Run)
end #getSession

function getFiles(info)
  #Get specific files for Subject, session and run
  Subject, Session, Run = info
  file = "BI.EEG.2012-GIPSA_subject_"*Subject*"_session_"*Session*"_run_"*Run*".npz"

  return file
end #getFiles


#Function specific to load files on my setup
function load1()
      #-----------------------------------------------------------------------------------#
      #Load a npz database using the name of the database or the index in the dbList (see MPTools)
      #corresponding to the alphabetical position in the folder
      #-----------------------------------------------------------------------------------#
      #Output :
      #     files::Vector{String} with N elements of DB 2012-Gipsa
      dbDir = homedir()*"\\Documents\\Etienne\\data\\BI.EEG.2012-GIPSA\\"
      #sub = ["/Sujet1","/Sujet2"]
      dbList = readdir(dbDir)
      try
            files = loadNYdb(dbDir)
			base = dirname(files[1])*"\\"
            return files, base
      catch e
            println("Base de donnees inexistante")
      end
end #load

function intraSessionAccuracy(o; nFolds=10, shuffle=false)
	# multivariate regression ERP mean with data-driven weights
    	w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]

    	# PCA to keep only 4 components
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
    	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]

	# Ledoit-Wolf
    	𝐂lw=ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X Y])) for X ∈ o.trials])

	# det normalization and regu
	#for i=1:length(𝐂lw) 𝐂lw[i]=det1(𝐂lw[i]) end

	#R=Hermitian(Matrix{eltype(𝐂lw[1])}(I, size(𝐂lw[1]))*0.0001)
	#for C in 𝐂lw C+=R end
	out = 0.4
    	# classification
	try
	    args=(shuffle=shuffle, tol=1e-6, verbose=false, nFolds=nFolds, ⏩=true)
	    cvlw = cvAcc(ENLR(Fisher), 𝐂lw, o.y; args...)
		out = cvlw.avgAcc
	catch
		out = 0.1
		print("skipping")
	end
	print("\n")

    return out
end #intraSessionAccuracy

function alignement()
	infos, subjAcc, base = getData()
	bestTSM, subjAM = getBestSubjects()

	calibSorted =  [base*getFiles(("$s","1","1")) for s in bestTSM]
	testSorted =  [base*getFiles(("$s","1","2")) for s in bestTSM]

	if isfile("data\\TangeantVectors.jld")
    #Useless to compute the tangeaent vectors if it's already done
		o = load("data\\TangeantVectors.jld")
	  	X = o["Xcalib"]
  		y = o["yCalib"]
  		Xtest =o["Xtest"]
  		ytest = o["ytest"]
  		println("Tangeant Vector already computed, using previous results")
	else
	#deux options pour le getTV :
	# => Soit on met tout, et toutes les tailles seront rapportées au plus petit EEG
	#Calcul des vecteurs tangeants priviliegiant la vitesse
		X, y = getTV(calibSorted, 1,[k for k in 2:length(calibSorted)])
		Xtest, ytest = getTV(testSorted, 1 , [k for k in 2:length(calibSorted)])
		# => Soit on calcul deux a deux, mais plus long
		# => pour le xtest on peut se permettre de ne pas prendre en compte l'équilibre des classes
		save("data\\TangeantVectors.jld", "Xcalib", X, "yCalib", y, "Xtest", Xtest, "ytest", ytest)
	end #if
	heatmap = Array{Float64}(undef,length(calibSorted),length(calibSorted))
	idAlign = copy(heatmap)

	for i in 1:25
		idAlign[i,i] = 1
	end

	for source in 1:length(calibSorted)
		heatmap[source,source] = subjAM[source]*100
		for target in 1:length(calibSorted)
			if source !=target
				filt = mca(X[source],X[target])
				id = spForm(filt.F[1]'*filt.F[2])
				idAlign[source,target] = id
				print("$source, $target/25 \n")
				#Fit
				acc = 50
				al1 = X[source]'*filt.F[1]
				al2 = Xtest[target]'*filt.F[2]
				try
					model = fit(ENLR(Fisher), al1, y; verbose=false, ⏩=false)
					yPR = predict(model, al2; verbose = false)
					predErr = predictErr(ytest, yPR)
					acc = 100. - predErr
				catch
					acc = 50
					print("skipping \n")
				end


				heatmap[source, target] = acc
			end #if
		end #for source
	end #for target
	save("data\\heatmap.jld", "Heatmap", heatmap, "spFromResult", idAlign)
	return heatmap, idAlign
end #alignement


#### Fonctions de tsalign.jl ##"#
function getTV(files, target_s, source_s; verbose=false, estimator=:Tyler, ncomp=4)

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
end #getTV

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
end #eegTangentVector
