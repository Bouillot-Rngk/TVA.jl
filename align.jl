module align
Dir = homedir()*"\\Documents\\ENSE3\\Thesis"

push!(LOAD_PATH,Dir*"\\TVA.jl\\src")
push!(LOAD_PATH,Dir*"\\MultiProcessing.jl\\src")
push!(LOAD_PATH,Dir*"\\data\\BI.EEG.2012-GIPSA")

using LinearAlgebra, PosDefManifold, PosDefManifoldML, CovarianceEstimation,
      Dates, Distributions, PDMats, Revise, BenchmarkTools, Diagonalizations,
      Random, DataFrames, CSV, Plots, JLD

using EEGio, FileSystem, EEGpreprocessing, System, ERPs,
      Tyler, EEGtomography, MPTools

export getBestTargets

function getBestTargets(db;
                        nFolds = 10,
                        verbose=true)
  #-------------------------------------------------------------------------------#
  #               Rank subjects based on their intra-sesssion accuracy            #

#=  if isfile("data\\best_target1.jld")
    #Useless to compute the accuracy if it's already done
    o = load("data\\best_target1.jld")
    bestTargetIdx = o["bestTargetIdx"]
    subjAcc = o["subjAcc"]
    println("Best subject target already computed, using previous results")
else=#
    files = load()
    #files = getFiles(('1','1',('1','2')))

    ⌚ = verbose && now()
    subjAcc = [intraSessionAccuracy(readNY(files[i];  bandpass=(1, 16)), nFolds=nFolds) for i in eachindex(files)]
  	bestTargetIdx = sortperm(subjAcc; rev=true)
  	verbose && println("Estimating intrasession accuracy done in ", now()-⌚)
    #save("data\\best_target1.jld", "bestTargetIdx", bestTargetIdx, "subjAcc", subjAcc)
  #end
  return bestTargetIdx, subjAcc;
end #getBestTargets

function getSession(file)
  file = "BI.EEG.2012-GIPSA_subject_1_session_1_run_1.yml"
  Subject = file[27]
  Session = file[37]
  Run = file[43]
  return info = (Subject, Session, ('1','2'))
end #getSession

function getFiles(info)
  #Get specific files for Subject, session and run
  Subject, Session, Run = info
  files = ["BI.EEG.2012-GIPSA_subject_"*Subject[k]*"_session_"*Session[j]*"_run_"*Run[i]*".npz"
                for k in 1:length(Subject) for j in 1:length(Session) for i in 1:length(Run)]
  return files
end #getFiles

function load()
      #-----------------------------------------------------------------------------------#
      #Load a npz database using the name of the database or the index in the dbList (see MPTools)
      #corresponding to the alphabetical position in the folder
      #-----------------------------------------------------------------------------------#
      #Output :
      #     files::Vector{String} with N elements of DB 2012-Gipsa
      Dir = homedir()*"\\Documents\\ENSE3\\Thesis\\data\\BI.EEG.2012-GIPSA\\"
      #sub = ["/Sujet1","/Sujet2"]
      dbList = readdir(Dir)
      try
            files = loadNYdb(Dir)
            return files
      catch e
            println("Base de donnees inexistante")
      end

end #loadDBPTVA

function intraSessionAccuracy(o; nFolds=10, shuffle=false)
	# multivariate regression ERP mean with data-driven weights
    	w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]

    	# PCA to keep only 4 components
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
    	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]

	# Ledoit-Wolf
    	𝐂lw=ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X Y])) for X ∈ o.trials])

	# det normalization and regu
	for i=1:length(𝐂lw) 𝐂lw[i]=det1(𝐂lw[i]) end

	#R=Hermitian(Matrix{eltype(𝐂lw[1])}(I, size(𝐂lw[1]))*0.0001)
	#for C in 𝐂lw C+=R end
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
end #intraSessionAccuracy


end #module
