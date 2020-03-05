module FileSystem

# functions:

# getFilesInDir | read all files in a directory (optionally with given extensions)


export
    getFilesInDir


    # Get the complete path of all files in `dir` as a vector of strings
    # `ext` is an optional tuple of file extensions given strings.
    # If it is provided, only files with those extensions will be included
    # in the returned vector.
    # the extensions must be entered in lowercase
    ## Examples
    # S=getFilesInDir(@__DIR__)
    # S=getFilesInDir(@__DIR__; ext=(".txt", ))
    # S=getFilesInDir(@__DIR__; ext=(".txt", ".jl"))

    function getFilesInDir(dir::String; ext::Tuple=())
        if !isdir(dir) @error "Function `getFilesInDir`: input directory is incorrect!"
        else
            S=[]
            for (root, dirs, files) in walkdir(dir)
                if root==dir
                    for file in files
                        if ext==() || ( lowercase(string(splitext(file)[2])) ∈ ext )
                           push!(S, joinpath(root, file)) # complete path and file name
                        end
                    end
                end
            end
            isempty(S) && @warn "Function `getFilesInDir`: input directory does not contain any files"
            return Vector{String}(S)
        end
    end

end # module
