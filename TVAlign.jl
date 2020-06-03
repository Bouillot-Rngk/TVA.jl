#### 0 - Initialization

Dir = homedir()*"\\Documents\\Etienne"


push!(LOAD_PATH,Dir*"\\git\\MP\\src") #Pour recuperer les modules EEGio etc
push!(LOAD_PATH,Dir*"\\git\\TSA")
push!(LOAD_PATH,Dir*"\\data\\BI.EEG.2012-GIPSA") #emplacement de la DB

using LinearAlgebra, PosDefManifold, PosDefManifoldML, CovarianceEstimation,
      Dates, Distributions, PDMats, Revise, BenchmarkTools, Diagonalizations,
      Random, DataFrames, CSV, Plots, MAT, Statistics   # + NPZ, YAML, StatsBase, DSP, libGR

using EEGio, FileSystem, EEGpreprocessing, System, ERPs,
      Tyler, EEGtomography, MPTools

using ERPML

#### BalanceERP test
files, base = load1()
Framebase = getBaseData(files)


function getCovtest(files; verbose=false, estimator=:Tyler, ncomp=4)

    ods = [readNY(s; bandpass=(1, 16)) for s ∈ files]

    ℙ = [eegERPCovariances(s, verbose=verbose, ncomp=ncomp, estimator=estimator) for s ∈ ods]
	𝕪 = [s.y for s ∈ ods ]

    return ℙ, 𝕪
end

#### Seriation and plot

heat1 = copy(TL[1])
heat2 =copy(TL[2])
sort!(heat2; dims=2,rev = true)
sort!(heat2; dims=1, rev=true)

Y = [heat1, heat2, TL[1], TL[2]]

heatmapsPlots(Y,"Results.png"; layout = (2,2), size = (1800,1300), titles=["TL Accuracy with nrTME estimator and seriation",
		"TL Accuracy with Wolf estimator and seriation","TL Accuracy with nrTME estimator", "TL Accuracy with Wolf estimator"],
		)
allcal, alltest = computeTV(base, Framebase, [:SCM])

function mean2(heat1,heat2)
	v1 = []
	v2 = []

	for i in 1:length(Subjects)
		for j in 1:length(Subjects)
			if heat1[i,j] != 50 && i!=j
				v1 = [v1; heat1[i,j]]
			end
			if heat2[i,j] != 50 i!=j
				v2 = [v2; heat2[i,j]]
			end
		end
	end
	return v1,v2
end

v1, v2 = mean2(heat1,heat2)
mean(heat1)
mean(heat2)
mean(v1)
mean(v2)
85.4 & 88.7
#### test zone

files, base = load1()
Framebase = getBaseData(files)
TL, idAlign = computeTLAcc2(Framebase, base, [:TME,:Wolf])





#### 2 - Get data of base

#get a Dataframe with Subject, Session, Run and filename of a file
function getBaseData(files)
	EEGfiles = [readNY(o) for o in files]
	infos = [getSSR(i) for i in EEGfiles]
	df = DataFrame(Subject = [s[1] for s in infos], Session = [s[2] for s in infos], Run = [s[3] for s in infos], Files =EEGfiles)
	return  df
end #getBaseData

#get Subjects, Session, Run of an EEG file
function getSSR(o)
  Subject = o.subject
  Session = o.session
  Run = o.run
  return info = (Subject, Session, Run)
end #getSSR

#get a file name from its Subject, Session and Run info tupple
function getFileName(info)
  #Get specific files for Subject, session and run
  Subject, Session, Run = info
  file = "BI.EEG.2012-GIPSA_subject_"*Subject*"_session_"*Session*"_run_"*Run*".npz"

  return file
end #getFiles

#### 1 - Load base

# Specific function to load files, dbDir must be modified according to database management
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
            println("Database not found, check dbDir path")
      end
end #load
