#### TANGENT VECTOR ALIGNEMENT = TRANSFER LEARNING ####
#=
Ck and yk NxN symetric positive definite matrices and associated labels k=1...K
G : appropriate Mass Center to define the tangent space for the set Ck
G : Mean of covariances, or covariance of mean targets => to test
V = [v1...vk] matrix DxK tangent vectors with D = N(N+1)/2 upper triangle of the matrix arg
vk = vec log(G₋½*Ck*G₋½) => vec give a weight 1 to diagonal element and √2 for off diagonal elements

V1, V2, VM => matrix obtained in M independant realizations of the same experiment
Cross session or cross subject matrices

=> Find M orthogonal transformations U1, U2, UM so UmVm for m=1...M are as colinear as possible
2 by 2 => MCA argmax tr(U1V1V2ᵀU2ᵀ)

for M>2 => gMCA argmin || off(Σᵢ≆ⱼ(UᵢVᵢVⱼᵀUⱼᵀ))||²

Then, the alignement can be on :
    The Source
    A Target
    All the targets with a gMCA
    all targets by cluster (how to define the clustering)

Once the alignement is done we fit a model with the aligned vectors, but we need to consider
the portion of each session/subject (Real) we want to give to the model :
    One realization only, then consider the predict w/out adjustement
    One realization + a portion of another realization (how much ?)
    Multiple realizations

Questions to answer :
    What do we win (in term of precision of prediction) by adding realization data to the model
    When do we start loosing precision ?
    How much adaptative data do we need to feed in the model ?



Algorithm process :
    Get V1,V2,...VM (starting with M=2)
    Aligned V1, V2, ...,VM (starting with V1 and V2 aligned together)
    Fit a model with V1, V1+20%ofV2, etc.
    Compute accuracy in the first case, and make a crossvalidation for the second case
    For M>2, we'll see later

=#


push!(LOAD_PATH,homedir()*"/Julia/TVA.jl/src")
push!(LOAD_PATH,homedir()*"/Julia/MultiProcessing.jl/src")


using tva, tsa
using MPTools, Processdb, EEGio, LinearAlgebra,PosDefManifold, PosDefManifoldML, Diagonalizations
using GLMNet



target_s = 3
source_s = [1,2,4]
db = 1
Dir = homedir()*"/Documents/My Data/EEG data/npz/BI.EEG.2013-Sorted"
dbSearch = Dir*"/Sujet$db/Base$db/"
dbSearch = homedir()*"/Documents/My Data/EEG data/npz/P300/BI.EEG.2012-GIPSA"
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


gm = gmca(𝕏)

𝕏aligned = [gm.F[i]'*𝕏[i] for i=1:length(𝕏)]
𝕏Tr = vcat([Matrix(𝕏aligned[i]'') for i in range(2, stop=length(source_s)+1)]...)
𝕪Tr = repeat(𝕪, 2)
model = fit(ENLR(Fisher), 𝕏Tr, 𝕪)
𝕏Te = Matrix(𝕏aligned[1]')

𝕪Pr = PosDefManifoldML.predict(model, 𝕏Te)
predErr = predictErr(𝕪, 𝕪Pr)
acc = 100. - predErr

o1=readNY(files[3]; bandpass=(1, 16))
o2=readNY(files[4]; bandpass=(1, 16))

vDT = genTangentVector(o1)
Ci = genSet(o1)



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

#Di = Vector{Matrix{Float64}}(undef,length(Vi))
#Dj = Vector{Matrix{Float64}}(undef,length(Vj))
Di = copy(Vi)
Dj = copy(Vj)

for i = 1:length(Vi)
    Di[i] =Hermitian(U.F'[1]*Vi[i]*U.F[1])
    Dj[i] =Hermitian(U.F'[2]*Vj[i]*U.F[2])
end

model = ENLR()
M = fit_TVA(model,Di,yi)
