#### ALIGNEMENT OF VECTORS ####
module tva
#=

For one set
Ck : Covariance matrices
G : Center of mass of the set C1,..,CK
vk : Tangent vector of Ck at center of mass G
Vk : Tangent matrix of set C1,...,CK at center of mass G

Let be Cmk a covariance matrix of realization k from set m
Cm the set m of covariances matrix
Let be Gm, vmk and Vmk just as before

=#

push!(LOAD_PATH,homedir()*"/Julia/TVA.jl/tools")
push!(LOAD_PATH,homedir()*"/Julia/TVA.jl/src")
push!(LOAD_PATH,homedir()*"/Julia/MultiProcessing.jl/src")

using PosDefManifold, CovarianceEstimation, Diagonalizations,LinearAlgebra, MPTools


using EEGio


export
        genSet,
        genMassCenter,
        genTangentVector

#### Gen Data ####
# => For o an EEG object
function genSet(o)
    w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
    Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
    Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]

    #Calcul de la matrice de covariance ponderee Ledoit Wolf
    Clw=ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X Y])) for X ∈ o.trials])
    return Clw
end

#### Center of Mass ####

#is it useful to implement other means ?

function genMassCenter( o::EEG;
                        meantype::String = "gmean",
                        metric::Metric = "Fisher"
                        )

    Cm = genSet(o)  #Holding K matrices

    if meantype == "mean"
        Gm = means(metric,Cm)
    elseif meantype == "gmean"
        Gm, iter, set = geometricMean(Cm)
    else
        print("mean method doesn't exist")
    end

    return Cm, Gm
end

function genMassCenter( Cm::HermitianVector;
                        meantype::String = "mean",
                        metric::Metric = "Fisher"
                        )

    if meantype == "mean"
        Gm = generalizedMean(Cm,0)
    elseif meantype == "gmean"
        Gm, iter, set = geometricMean(Cm)
    else
        print("mean method doesn't exist")
    end
    return Gm
end


#### Computation of Vmk ####
#=function genTangentVector(  Cm::HermitianVector;
                            meantype::String = "gmean"
                            )

    Gm = genMassCenter(Cm; meantype = meantype, metric = Fisher)
    Vmk = logMap(Fisher,Cm,Gm)
    return Gm, Vmk

end=#

function genTangentVector( o::EEG;
                            meantype::String = "gmean"
                            )
    Cm = genSet(o)
    Gm = genMassCenter(Cm; meantype = "gmean", metric = Fisher)
    Vmk = logMap(Fisher, Cm, Gm)
    V = hcat([vecP(log(G), range=1:o.ne) for G ∈ Vmk]...)
    return V
end


#### 2-Set Alignement ####
#Considering Ci and Cj two sets of realizations => Cross session or Cross subject in our case from files o1 and o2

end #Module
