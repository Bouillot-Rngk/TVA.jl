module Processdb

using PosDefManifoldML, CovarianceEstimation, LinearAlgebra,
      Dates, PDMats, Revise, BenchmarkTools, Plots, GR, DataFrames,
      CSV, Statistics
using EEGio, System, ERPs, Diagonalizations,
      DelphiApplications, Tyler, EEGtomography, Estimators,MPTools


export
      loadDBP300,
      multiProcessCSV




function loadDBP300(dbName)
      #-----------------------------------------------------------------------------------#
      #Load a npz database using the name of the database or the index in the dbList (see MPTools)
      #corresponding to the alphabetical position in the folder
      #-----------------------------------------------------------------------------------#
      #Input :
      #     dbName::String or Int
      #Output :
      #     files::Vector{String} with N elements

      Dir, dbList, t = MPTools.init()

      if dbName isa String && dbName in dbList
            dbSearch = Dir*"/P300/"*dbName;
      end
      if dbName isa Int
            dbSearch = Dir*"/P300/"*dbList[dbName];
      end
      try
            files = loadNYdb(dbSearch)
            return files
      catch e
            println("Base de donnees inexistante");
      end
end #loadDBP300



function multiProcessCSV(
      databases,
      estimators,
      CSVFile,
      x = 1,
      donnees = DataFrame(meanA = Float64[], sdA = Float64[], Method = String[],
      Database = String[], Sujet_Session_Run = String[],Time = DateTime[])
      )
#-----------------------------------------------------------------------------------#
#Processing of length(databases) databases with length(estimators) covariance estimator
#The if-elseif must be update when a new estimator function is available
#Return a csv file with all meanA, sdA and time computed whilst crossvalidation that
#can be read and treated with MPTools.plotResults() function
#-----------------------------------------------------------------------------------#
#Input :
#     databases::Vector{String} containing names or index of to-be-processed DB
#     estimators::Vector{String} comtaining names of covariance estimator to be tested
#             The list of DB and estimators available is described in MPTools.jl
#Output :
#     bool::Bool = true if processing is successfully complete
#     output_finished.csv::CSV File stored in data/ folder
#     output.csv in data/ folder is a backup in case of any kind of crash during computing

Dir, dbList, estimatorList = MPTools.init()
ListofDB = copy(databases)
for (i,base) ∈ enumerate(databases)
#Maybe check if inputs are correct before launching code ?
      Cfile = "/nethome/bouillet/Julia/MultiProcessing.jl/data/"*CSVFile
      #loading of 1 database
      files = loadDBP300(base)

      #Memory allocation
      meanA=Vector{Float64}(undef, length(files)); sdA = similar(meanA)

      #display in REPL for control => used base and number of elements
      if typeof(base)==Int print("base ",dbList[base], "  w/ ", length(files)," elements \n"); base=dbList[base]
      else print("base", base, "  w/ ", length(files)," elements \n")
      end #end if


            #Data processing
            for (i, file) ∈ enumerate(files[x:end])
                  o=readNY(file; bandpass=(1, 16)) # read files and create the o structure
                  #mean = Statistics.mean(abs.(o.X))
                  #print(mean, " \n")
                  print(i+x-1, "/",length(files), " ", rpad("sj: $(o.subject), ss: $(o.session), run $(o.run): ", 26)," \n");
                  ⌚ = now()
                  for (j,method) ∈ enumerate(estimators)
                        #REPL Display for control

                  #Choice of covariance estimator => work great

                        if method == "SCM"
                              Clw = SCMP300(o)

                        elseif method == "TME"
                              Clw = TMEP300(o)

                        elseif method == "nrTME"
                              Clw = nrTMEP300(o)

                        elseif method == "Wolf"
                              Clw = Wolf(o)

                        elseif method == "nrTMEFilt"
                              Clw = nrTMEFiltP300(o)

                        else print("Estimator doesn't exist \n"); break

                        end #switch-case

                        methodCSV = method*"no_norm"
                        #Clw = detnorm(Clw); methodCSV = method*"_detnorm"

                        w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
                        Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
                        Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]
                  #      print(size(o.X), " : X   \n")
                  #      print(size(Clw), " : Clw \n")
                        wX = whitening([o.X Y];eVar=0.99);
                  #      print(size(wX.F), " : wX.F  \n")

                        for i=1:length(Clw)
                              Clw[i] = Hermitian(wX.F'*Clw[i]*wX.F)
                        end #Whitening
                        methodCSV = method*"_whitened"
                        #
                        print("Methode ",methodCSV, " :   ")
                        #beginning of crossvalidation

                        try
                              args=(shuffle=false, tol=1e-6, verbose=false)
                              cvlw = cvAcc(PosDefManifoldML.ENLR(), Clw, o.y; args...)
                              meanA[i] = cvlw.avgAcc
                              sdA[i] = cvlw.stdAcc


                              time = now()-⌚
                              #REPL Display for control
                              println(rpad(round(meanA[i]; digits=4), 6), " (", rpad(round(sdA[i]; digits=4), 6), ") done in ", time)

                              #Datastorage
                              data = [cvlw.avgAcc, cvlw.stdAcc, methodCSV, base,
                                    rpad("sj: $(o.subject), ss: $(o.session), run $(o.run): ", 26), time ]

                              push!(donnees,data)
                        catch
                              print("skipping $file \n")
                              #mv(file, "/nethome/bouillet/Documents/My Data/EEG data/npz/P300/.Deprecated/"*basename(file))
                              #yml = file[1:length(file)-3]*"yml"
                              #mv(yml, "/nethome/bouillet/Documents/My Data/EEG data/npz/P300/.Deprecated/"*basename(yml))
                              print("error in $file \n Restarting \n")
                              multiProcessCSV(ListofDB,estimators,CSVFile,x+i+1,donnees)
                        end #trycatch


                        #Back up csv

                        CSV.write(Cfile,donnees)
                        print("writing in $Cfile \n")


                  end   #for method
            end #for file
            #CSV.write(Cfile,donnees)
            try
            deleteat!(ListofDB,1)
            catch
                  break
            end

#Julia/Multiprocessing.jl/
end #base
      print("Finished")

end


end #module
