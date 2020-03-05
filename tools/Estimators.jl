module Estimators
using LinearAlgebra, PosDefManifold, PosDefManifoldML, CovarianceEstimation,
      Dates, Distributions, PDMats, Revise, BenchmarkTools, Plots, GR
using EEGio, System, ERPs,
      DelphiApplications, Tyler, EEGtomography

# ? ¤ CONTENT ¤ ? #

# STRUCTURES
# EEG | holds data and metadata of an EEG recording

# FUNCTIONS:
# SCMP300 	 | read an EEG struct anc compute simple Sample Covariance Matrix
# TMEP300  	 | read an EEG struct anc compute Tyler's M-estimator Covariance Matrix
# nrTMEP300  | read an EEG struct anc compute normalize regularized Tyler's M-estimator Covariance Matrix
# Wolf 		 | read an EEG struct anc compute Ledoit Wolf Covariance Matrix

export
	SCMP300,
	TMEP300,
	nrTMEP300,
	nrTMEFiltP300,
	detnorm,
	Wolf



function SCMP300(o)
#-----------------------------------------------------------------------------------#
##Computation of Covariance Matrix w/ Sample Covariance Matrix estimator
#& regularization & det normalization
#-----------------------------------------------------------------------------------#
#Input :
#     o::EEG => Structure de EEGio.jl apres lecture de la bdd par readNY
#Output :
#     Clw : Matrice de covariance etendue (supertrials)

#Calcul des poids et moyenne + PCA pour ne conserver que 4 elements

    w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]

#Calcul de la matrice de covariance ponderee
	Clw=ℍVector([ℍ(cov(SimpleCovariance(), [X Y])) for X ∈ o.trials])
	#regularization
	R=Hermitian(Matrix{eltype(Clw[1])}(I, size(Clw[1]))*0.01)
	for C in Clw C+=R end
	Clw = ℍVector(Clw)
	#for i=1:length(Clw) Clw[i]=det1(Clw[i]) end

#Calculs annexes (normalization de det, etc)


	return Clw
end

function TMEP300(o)
#-----------------------------------------------------------------------------------#
#Computation of Covariance Matrix w/ Tyler estimator & regularization & det normalization
#-----------------------------------------------------------------------------------#
#Input :
#     o::EEG => Structure de EEGio.jl apres lecture de la bdd par readNY
#Output :
#     Clw : Matrice de covariance etendue, calcul TME (supertrials)
#Calcul des poids et moyenne + PCA pour ne conserver que 4 elements
	w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]
#Calcul de la ;atrice de covariance par estimateur de Tyler
	Clw=ℍVector([ℍ(tme([X Y]')) for X ∈ o.trials])
	#regularization
	R=Hermitian(Matrix{eltype(Clw[1])}(I, size(Clw[1]))*0.0001)
	for C in Clw C+=R end
	for i=1:length(Clw) Clw[i]=det1(Clw[i]) end
	return Clw
end

function nrTMEP300(o)
#-----------------------------------------------------------------------------------#
#Computation of Covariance Matrix w/ normalized regularized Tyler estimator
#& regularization & det normalization
#-----------------------------------------------------------------------------------#
#Input :
#     o::EEG => Structure de EEGio.jl apres lecture de la bdd par readNY
#Output :
#     Clw : Matrice de covariance etendue, calcul nrTME (supertrials)
#Calcul des poids et moyenne + PCA pour ne conserver que 4 elements
	w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]
#Calcul de la matrice de covariance par estimateur de Tyler non regularized
	Clw=ℍVector([ℍ(nrtme([X Y]'; reg=:LW)) for X ∈ o.trials])
	#regularization
	R=Hermitian(Matrix{eltype(Clw[1])}(I, size(Clw[1]))*0.0001)
	for C in Clw C+=R end
	return Clw
end

function Wolf(o)
#-----------------------------------------------------------------------------------#
#Computation of Covariance Matrix w/ Ledoit Wolf estimator & regularization & det normalization
#-----------------------------------------------------------------------------------#
	w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]
	Clw = ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X Y])) for X ∈ o.trials])

	return Clw

end


function nrTMEFiltP300(o)
	#-----------------------------------------------------------------------------------#
#Input :
#     o::EEG => Structure de EEGio.jl apres lecture de la bdd par readNY
#Output :
#     Clw : Matrice de covariance etendue, calcul nrTME (supertrials)
#Calcul des poids et moyenne + PCA pour ne conserver que 4 elements
	w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]
#Calcul de la matrice de covariance par estimateur de Tyler non regularized
	Clw=ℍVector([ℍ(nrtmeFilt([X Y]',o.sr; reg=:LW)) for X ∈ o.trials])
	#regularization
	R=Hermitian(Matrix{eltype(Clw[1])}(I, size(Clw[1]))*0.0001)
	for C in Clw C+=R end
	return Clw

end

function nlseP300(o)
#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#
#Input :
#     o::EEG => Structure de EEGio.jl apres lecture de la bdd par readNY
#Output :
#     Clw : Matrice de covariance etendue, calcul nrTME (supertrials)
#Calcul des poids et moyenne + PCA pour ne conserver que 4 elements
	w=[[1/(norm(o.X[o.cstim[i][j]+o.offset:o.cstim[i][j]+o.offset+o.wl-1,:])^2) for j=1:length(o.cstim[i])] for i=1:o.nc]
	Y=mean(o.X, o.wl, o.cstim; weights=w)[2]
	Y=Y*eigvecs(cov(SimpleCovariance(), Y))[:, o.ne-3:o.ne]
#Calcul de la matrice de covariance par estimateur non linear shrinkage
	Clw=ℍVector([ℍ(cov(nlse(), [X Y])) for X ∈ o.trials])
#Regularization


	return Clw
end

function detnorm(Clw)
	for i=1:length(Clw) Clw[i]=det1(Clw[i]) end
	return Clw
end

function regu(Clw,
			α::Real = 0.0001)

	R=Hermitian(Matrix{eltype(Clw[1])}(I, size(Clw[1]))*α)
	for C in Clw C+=R end

end


end #module
