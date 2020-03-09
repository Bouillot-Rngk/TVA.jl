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
push!(LOAD_PATH, homedir()*"/src/julia/Modules")
push!(LOAD_PATH,homedir()*"/Julia/TVA.jl/tools")
push!(LOAD_PATH,homedir()*"/Julia/TVA.jl/src")
push!(LOAD_PATH,homedir()*"/Julia/MultiProcessing.jl/src")

using PosDefManifold, CovarianceEstimation, Diagonalizations,LinearAlgebra
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

    #Calcul de la matrice de covariance ponderee
    Clw=ℍVector([ℍ(cov(SimpleCovariance(), [X Y])) for X ∈ o.trials])
    return Clw
end

#### Center of Mass ####

#is it useful to implement other means ?

function genMassCenter( o::EEG;
                        meantype::String = "mean",
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
function genTangentVector(  Cm::HermitianVector;
                            meantype::String = "mean"
                            )

    Gm = genMassCenter(Cm; meantype = mean, metric = Fisher)
    Vmk = logMap(Fisher,Cm,Gm)
    return Vmk

end


#### 2-Set Alignement ####
#Considering Ci and Cj two sets of realizations => Cross session or Cross subject in our case from files o1 and o2

function twoSetAlignement(o1,o2)
    Ci = genSet(o1)
    Gi = genMassCenter(Ci; meantype = "mean", metric = Fisher)
    Vi = logMap(Fisher,Ci,Gi)


    Cj = genSet(o2)
    Gj = genMassCenter(Cj; meantype = "mean", metric = Fisher)
    Vj = logMap(Fisher,Cj,Gj)

    #Vic = convert(Vector{Matrix{Float64}},Vi)
    #Vjc = convert(Vector{Matrix{Float64}},Vj)
    Vj = Vj[1:180]

    yi = IntVector(o1.y)
    yj = IntVector(o2.y[1:180])

    U = mca(convert(Vector{Matrix{Float64}},Vi),convert(Vector{Matrix{Float64}},Vj))

    Di = copy(Vi)
    Dj = copy(Vj)

    for i = 1:length(Vi)
        Di[i] =Hermitian(U.F'[1]*Vi[i]*U.F[1])
        Dj[i] =Hermitian(U.F'[2]*Vj[i]*U.F[2])
    end
end

end #Module
