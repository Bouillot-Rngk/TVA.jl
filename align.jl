module align

push!(LOAD_PATH, homedir()*"/src/julia/Modules")
push!(LOAD_PATH,homedir()*"/Julia/TVA.jl/src")
push!(LOAD_PATH,homedir()*"/Julia/MultiProcessing.jl/src")


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

  if isfile("./Data/best_target.jld")
    #Useless to compute the accuracy if it's already done
    o = load("./best_target.jld")
    bestTargetIdx = o["bestTargetIdx"]
    subjAcc = o["subjAcc"]
    println("Best subject target already computed, using previous results")
  else
    files = loadBDP300(db)
    ⌚ = verbose && now()
    subjAcc = [intraSessionAccuracy(readNY(files[i];  bandpass=(1, 16)), nFolds=nFolds) for i in eachindex(files)]
  	bestTargetIdx = sortperm(subjAcc; rev=true)
  	verbose && println("Estimating intrasession accuracy done in ", now()-⌚)
    save("./best_target.jld", "bestTargetIdx", bestTargetIdx, "subjAcc", subjAcc)
  end
  return bestTargetIdx, subjAcc

end #getBestTargets
