module fitTVA

using GLMNet

export fit_TVA,
		predict_TVA



function fit_TVA(model  :: ENLRmodel,
               𝐏Tr  :: Union{HermitianVector, Matrix{Float64}},
               yTr  :: IntVector;
		   # parameters for projection onto the tangent space
           w        	:: Union{Symbol, Tuple, Vector} = [],
           meanISR  	:: Union{Hermitian, Nothing} = nothing,
		   meanInit 	:: Union{Hermitian, Nothing} = nothing,
           vecRange 	:: UnitRange = 𝐏Tr isa HermitianVector ? (1:size(𝐏Tr[1], 2)) : (1:size(𝐏Tr, 2)),
           fitType  	:: Symbol = :best,
		   verbose  	:: Bool = true,
           ⏩      	   :: Bool = true,
           # arguments for `GLMNet.glmnet` function
           alpha            :: Real = model.alpha,
           weights          :: Vector{Float64} = ones(Float64, length(yTr)),
           intercept        :: Bool = true,
		   standardize  	:: Bool = true,
           penalty_factor   :: Vector{Float64} = ones(Float64, _getDim(𝐏Tr, vecRange)),
           constraints      :: Matrix{Float64} = [x for x in (-Inf, Inf), y in 1:_getDim(𝐏Tr, vecRange)],
           offsets          :: Union{Vector{Float64}, Nothing} = nothing,
           dfmax            :: Int = _getDim(𝐏Tr, vecRange),
           pmax             :: Int = min(dfmax*2+20, _getDim(𝐏Tr, vecRange)),
           nlambda          :: Int = 100,
           lambda_min_ratio :: Real = (length(yTr) < _getDim(𝐏Tr, vecRange) ? 1e-2 : 1e-4),
           lambda           :: Vector{Float64} = Float64[],
           tol              :: Real = 1e-5,
           maxit            :: Int = 1000000,
           algorithm        :: Symbol = :newtonraphson,
           # selection method
           λSelMeth :: Symbol = :sd1,
           # arguments for `GLMNet.glmnetcv` function
           nfolds   :: Int = min(10, div(size(yTr, 1), 3)),
           folds    :: Vector{Int} =
           begin
               n, r = divrem(size(yTr, 1), nfolds)
               shuffle!([repeat(1:nfolds, outer=n); 1:r])
           end,
           parallel ::Bool=true)

    ⌚=now() # get the time in ms
    ℳ=deepcopy(model) # output model

	# overwrite fields in `ℳ` if the user has passed them here as arguments,
	# otherwise use as arguments the values in the fields of `ℳ`, e.g., the default
	if alpha ≠ 1.0 ℳ.alpha = alpha else alpha = ℳ.alpha end

    # check w argument and get weights for input matrices
    (w=_getWeights(w, yTr, "fit ("*_modelStr(ℳ)*" model)")) == nothing && return

    # other checks
    𝐏Tr isa HermitianVector ? nObs=length(𝐏Tr) : nObs=size(𝐏Tr, 1)
    !_check_fit(ℳ, nObs, length(yTr), length(w), length(weights), "ENLR") && return

	# project data onto the tangent space or just copy the features if 𝐏Tr is a matrix
	X=_getFeat_fit!(ℳ, 𝐏Tr, meanISR, meanInit, tol, w, vecRange, true, verbose, ⏩)

    # convert labels in GLMNet format
    y = convert(Matrix{Float64}, [(yTr.==1) (yTr.==2)])

    # write some fields in output model struct
    ℳ.standardize = standardize
    ℳ.intercept   = intercept
    ℳ.featDim     = size(X, 2)
	ℳ.vecRange    = vecRange

    # collect the argumenst for `glmnet` function excluding the `lambda` argument
    fitArgs_λ = (alpha            = alpha,
                 weights          = weights,
                 standardize      = standardize,
                 intercept        = intercept,
                 penalty_factor   = penalty_factor,
                 constraints      = constraints,
                 offsets          = offsets,
                 dfmax            = dfmax,
                 pmax             = pmax,
                 nlambda          = nlambda,
                 lambda_min_ratio = lambda_min_ratio,
                 tol              = tol,
                 maxit            = maxit,
                 algorithm        = algorithm)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:path, :all)
        # fit the regularization path
        verbose && println("Fitting "*_modelStr(ℳ)*" reg. path...")
        ℳ.path = glmnet(X, y, Binomial();
                         lambda = lambda,
                         fitArgs_λ...) # glmnet Args but lambda
    end
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if fitType ∈(:best, :all)
        verbose && println("Fitting best "*_modelStr(ℳ)*" model...")
        ℳ.cvλ=glmnetcv(X, y, Binomial();
                        nfolds   = nfolds,
                        folds    = folds,
                        parallel = parallel,
                        lambda   = lambda,
                        fitArgs_λ...) # glmnet Args but lambda

        # Never consider the model with only the intercept (0 degrees of freedom)
        l, i=length(ℳ.cvλ.lambda), max(argmin(ℳ.cvλ.meanloss), 1+intercept)

        # if bestλsel==:sd1 select the highest model with mean loss withinh 1sd of the minimum
        # otherwise the model with the smallest mean loss.
        thr=ℳ.cvλ.meanloss[i]+ℳ.cvλ.stdloss[i]
        λSelMeth==:sd1 ? (while i<l ℳ.cvλ.meanloss[i+1]<=thr ? i+=1 : break end) : nothing

        # fit the best model (only for the optimal lambda)
        ℳ.best = glmnet(X, y, Binomial();
                         lambda  = [ℳ.cvλ.path.lambda[i]],
                         fitArgs_λ...) # glmnet Args but lambda
    end
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    return ℳ
end
####
function predict_TVA(model   :: ENLRmodel,
                 𝐏Te     :: Union{HermitianVector, Matrix{Float64}},
                 what    :: Symbol = :labels,
                 fitType :: Symbol = :best,
                 onWhich :: Int    = Int(fitType==:best);
			transfer   :: Union{Hermitian, Nothing} = nothing,
            verbose    :: Bool = true,
            ⏩        :: Bool = true)

    ⌚=now()

    # checks
    if !_whatIsValid(what, "predict ("*_modelStr(model)*")") return end
    if !_fitTypeIsValid(fitType, "predict ("*_modelStr(model)*")") return end
    if fitType==:best && model.best==nothing @error 📌*", predict function: the best model has not been fitted; run the `fit`function with keyword argument `fitType=:best` or `fitType=:all`"; return end
    if fitType==:path && model.path==nothing @error 📌*", predict function: the regularization path has not been fitted; run the `fit`function with keyword argument `fitType=:path` or `fitType=:all`"; return end
    if !_ENLRonWhichIsValid(model, fitType, onWhich, "predict ("*_modelStr(model)*")") return end

    # projection onto the tangent space
	X=_getFeat_Predict!(model, 𝐏Te, transfer, model.vecRange, true, verbose, ⏩)

    # prediction
    verbose && println("Predicting "*_ENLRonWhichStr(model, fitType, onWhich)*"...")
    if 		fitType==:best
        	path=model.best
        	onWhich=1
    elseif  fitType==:path
        	path=model.path
    end

    onWhich==0 ? π=GLMNet.predict(path, X) : π=GLMNet.predict(path, X, onWhich)

    k, l=size(π, 1), length(path.lambda)
    if     	what == :functions || what == :f
        	🃏=π
    elseif 	what == :labels || what == :l
        	onWhich==0 ? 🃏=[π[i, j]<0 ? 1 : 2 for i=1:k, j=1:l] : 🃏=[y<0 ? 1 : 2 for y ∈ π]
    elseif 	what == :probabilities || what == :p
        	onWhich==0 ? 🃏=[softmax([-π[i, j], 0]) for i=1:k, j=1:l] : 🃏=[softmax([-y, 0]) for y ∈ π]
    end

    verbose && println(defaultFont, "Done in ", now()-⌚,".")
    verbose && println(titleFont, "\nPredicted ",_what2Str(what),":", defaultFont)
    return 🃏
end



# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, M::ENLR)
    if M.path==nothing
        if M.best==nothing
            println(io, greyFont, "\n↯ ENLR GLMNet machine learning model")
            println(io, "⭒  ⭒    ⭒       ⭒          ⭒")
            println(io, ".metric : ", string(M.metric))
            println(io, ".alpha  : ", "$(round(M.alpha, digits=3))", defaultFont)
            println(io, "Unfitted model")
            return
        end
    end

    println(io, titleFont, "\n↯ GLMNet ENLR machine learning model")
    println(io, separatorFont, "  ", _modelStr(M))
    println(io, separatorFont, "⭒  ⭒    ⭒       ⭒          ⭒", defaultFont)
    println(io, "type    : PD Tangent Space model")
    println(io, "features: tangent vectors of length $(M.featDim)")
    println(io, "classes : 2")
    println(io, separatorFont, "Fields  : ")
	# # #
	println(io, greyFont, " Tangent Space Parametrization", defaultFont)
    println(io, separatorFont," .metric      ", defaultFont, string(M.metric))
	if M.meanISR == nothing
        println(io, greyFont, " .meanISR      not created")
    else
        n=size(M.meanISR, 1)
        println(io, separatorFont," .meanISR     ", defaultFont, "$(n)x$(n) Hermitian matrix")
    end
    println(io, separatorFont," .vecRange    ", defaultFont, "$(M.vecRange)")
    println(io, separatorFont," .featDim     ", defaultFont, "$(M.featDim)")
	# # #
	println(io, greyFont, " ENLR Parametrization", defaultFont)
    println(io, separatorFont," .alpha       ", defaultFont, "$(round(M.alpha, digits=3))")
    println(io, separatorFont," .intercept   ", defaultFont, string(M.intercept))
	println(io, separatorFont," .standardize ", defaultFont, string(M.standardize))


    if M.path==nothing
        println(io, greyFont," .path struct `GLMNetPath`:")
        println(io, "       not created ")
    else
        println(io, separatorFont," .path", defaultFont," struct `GLMNetPath`:")
        println(io, titleFont,"       .family, .a0, .betas, .null_dev, ")
        println(io, titleFont,"       .dev_ratio, .lambda, .npasses")
    end
    if M.cvλ==nothing
        println(io, greyFont," .cvλ  struct `GLMNetCrossValidation`:")
        println(io, "       not created ")
    else
        println(io, separatorFont," .cvλ", defaultFont,"  struct `GLMNetCrossValidation`:")
        println(io, titleFont,"       .path, .nfolds, .lambda, ")
        println(io, titleFont,"       .meanloss, stdloss")
    end
    if M.best==nothing
        println(io, greyFont," .best struct `GLMNetPath`:")
        println(io, "       not created ")
    else
        println(io, separatorFont," .best", defaultFont," struct `GLMNetPath`:")
        println(io, titleFont,"       .family, .a0, .betas, .null_dev, ")
        println(io, titleFont,"       .dev_ratio, .lambda, .npasses")
    end
end



end #module
