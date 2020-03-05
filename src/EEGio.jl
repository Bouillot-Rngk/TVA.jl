module EEGio
using NPZ, YAML, FileSystem, EEGpreprocessing, DSP

# ? ¤ CONTENT ¤ ? #

# STRUCTURES
# EEG | holds data and metadata of an EEG recording

# FUNCTIONS:
# loadNYdb | return a list of .npz files in a directory
# readNY   | read an EEG recording in NY (npz, yml) format
# readASCII (2 methods) | read one ASCII file or all ASCII files in a directory
# writeASCII            | write one abstractArray dara matrix in ASCII format

# max number of elements in an EEG matrix that can be handled by ICoN
const titleFont     = "\x1b[95m"
const separatorFont = "\x1b[35m"
const defaultFont   = "\x1b[0m"
const greyFont      = "\x1b[90m"

export
    EEG,
    loadNYdb,
    readNY,
    readASCII,
    writeASCII,
    loadNYdbCS

# `EEG` structure holding data and metadata information for an EEG recording
# An instance is created by the `readNY` function, which reads into files
# in NY (npz, yml) format.
# Fundamental fields can be accessed directly, for example, if `o` is an
# instance of the structure, the EEG data in in `o.X`.
# All metadata can be accessed in the dictionaries. For reading them
# use for example syntax `o.acquisition["ground"]`
struct EEG
    id            :: Dict{Any,Any} # `id` Dictionary of the .yml file
    # it includes keys:   "run", "other", "database", "subject", "session"
    acquisition   :: Dict{Any,Any} # `acquisition` Dictionary of the .yml file
    # it includes keys:   "sensors", "software", "ground", "reference",
    #                      "filter", "sensortype", "samplingrate", "hardware"
    documentation :: Dict{Any,Any} # `acquisition` Dictionary of the .yml file
    # it includes keys:   "doi", "repository", "description"
    formatversion :: String        # `formatversion` field of the .yml file

    # the following fields are what is useful in practice
    db            :: String        # name of the database to which this file belongs
    subject       :: Int           # serial number of the subject in database
    session       :: Int           # serial number of the session of this subject
    run           :: Int           # serial number of the run of this session
    sensors       :: Vector{String}# electrode leads on the scalp in standard 10-10 notation
    sr            :: Int           # sampling rate
    ne            :: Int           # number of electrodes (excluding reference and ground)
    ns            :: Int           # number of samples
    wl            :: Int           # window length: typically, the duration of the trials
    offset        :: Int           # each trial start at `stim` sample + offset
    nc            :: Int           # number of classes
    clabels       :: Vector{String} # class labels given as strings
    stim          :: Vector{Int}    # stimulations for each sample (0, 1, 2...). 0 means no stimulation
    cstim         :: Vector{Vector{Int}}  # stimulations for class 1, 2...
    y             :: Vector{Int}          # the vectors in `cstim` concatenated
    X             :: Matrix{T} where T<:Real # whole recording EEG data (ns x ne)
    trials        :: Union{Vector{Matrix{T}}, Nothing} where T<:Real # all trials in order ot `stims` (optional)
end




# Return a list of the complete path of all .npz files
# in directory `DBdir` for which a corresponding .yml file exist.
function loadNYdb(dbDir=AbstractString)
  # create a list of all .npz files found in dbDir (complete path)
  npzFiles=getFilesInDir(dbDir; ext=(".npz", ))

  # check if for each .npz file there is a corresponding .yml file
  missingYML=[i for i ∈ eachindex(npzFiles) if !isfile(splitext(npzFiles[i])[1]*".yml")]
  if !isempty(missingYML)
    @warn "the following .yml files have not been found:\n"
    for i ∈ missingYML println(splitext(npzFiles[i])[1]*".yml") end
    deleteat!(npzFiles, missingYML)
    println("\n $(length(npzFiles)) files have been retained.")
  end
  return npzFiles
end

####

function loadNYdbCS(BDir=String)
  # create a list of all .npz files found in dbDir to complete Cross Session TL (complete path)
  DirCS=BDir*"/Sujet1/Base1"
  npzFiles = loadNYdb(DirCS)
  return npzFiles
end

####


# Read EEG data in NY (npz, yml) format and create an `EEE` structure.
# The complete path of the file given by `filename`.
#  Either the .npz or the .yml file can be passed.
# If a 2-tuple is passed as `bandpass`, data is filtered in the bandpass.
# If a fractional or integer number is given as `resample`, the data is
# resamples, e.g., `resample=1//2` will downsample by half and `resample=3`
# will upsample by 3. NB: resampling is still experimental.
# If `getTrials` is true, the `trials` field of the `EEG` structure is filled.
# If `msg` is not empty, print `msg` on exit.
function readNYArtifact(filename  :: AbstractString;
                bandpass  :: Tuple=(),
                resample  :: Union{Rational, Int}=1,
                getTrials :: Bool=true,
                msg       :: String="")

  data = npzread(splitext(filename)[1]*".npz") # read data file
  info = YAML.load(open(splitext(filename)[1]*".yml")) # read info file

  sr      = info["acquisition"]["samplingrate"]
  stim    = data["stim"]                  # stimulations
  (ns, ne)= size(data["data"])            # of sample, # of electrodes)
  os      = info["stim"]["offset"]        # offset for trial starting sample
  wl      = info["stim"]["windowlength"]  # trial duration
  nc      = info["stim"]["nclasses"]      # of classes

  # band-pass the data if requested
  if isempty(bandpass)
    X=data["data"]
  else
    BPfilter = digitalfilter(Bandpass(first(bandpass)/(sr/2), last(bandpass)/(sr/2)), Butterworth(2))
    X        = filtfilt(BPfilter, data["data"])
  end

  # resample data if requested
  if resample≠1
    X         = resample(X, resample; stim=stim)
    (ns, ne)  = size(X)
    wl        = round(Int, wl*resample)
    os        = round(Int, os*resample)
    sr        = round(Int, sr*resample)
    wl        = round(Int, wl*resample)
  end

  # vectors of samples where the trials start for each class 1, 2,...

  gfp=[x⋅x for x ∈ eachrow(o.X)]

  #Calcul du gfp + tri dans un DataFrame pour conserver l'ordre
  lsgfp=log10.(gfp)
  Plots.plot(lsgfp)
  x = 1:length(lsgfp)
  data=DataFrame(X = x, GFP = lsgfp, Deriv = missing)
  datasorted = sort!(data, [:GFP, :X])
  Plots.plot(datasorted[!, :GFP])


  #Calcul de la derivée
  derivative = Vector{Float64}(undef,length(lsgfp))
  for x0 = 31:length(lsgfp)-30
    moy=Vector{Float64}(undef,29)
    for wl = 2:30
        moy[wl-1] = (datasorted[!, :GFP][x0+wl] - datasorted[!, :GFP][x0-wl])/(2*wl)
    end
    mean = Statistics.mean(moy)
    derivative[x0] = mean
  end
  for i=1:length(derivative)
    if derivative[i]<0.0000000000001 derivative[i] = 0 ; end
  end
  deriv = Statistics.mean(derivative)
  Scalederivative = derivative/deriv
  Plots.plot(Scalederivative)
  for i=length(derivative)-50:length(derivative)
    Scalederivative[i] = 30
  end
  datasorted.Deriv = Scalederivative
  dataArtifact = datasorted[1000:end,  :]

  dataArtifact = dataArtifact[dataArtifact.Deriv .> 15, :]
  Plots.plot(dataArtifact[!, :GFP])
  indexArtif = copy(dataArtifact)
  indexArtif = sort!(indexArtif, :X)
  Plots.plot(indexArtif[!, :GFP])

  stimArtifact = deleteat!(o.stim, indexArtif[!, :X])

  ns,ne = size(stimArtifact)

  cstim=[[i+os for i in eachindex(stimArtifact) if stimArtifact[i]==j && i+os+wl<=ns] for j=1:nc]

  getTrials ?  trials=[X[cstim[i][j]:cstim[i][j]+wl-1,:] for i=1:nc for j=1:length(cstim[i])] :
                trials=nothing


  if !isempty(msg) println(msg) end

  # this creates the `EEG` structure
  EEG(
     info["id"],
     info["acquisition"],
     info["documentation"],
     info["formatversion"],

     info["id"]["database"],
     info["id"]["subject"],
     info["id"]["session"],
     info["id"]["run"],
     info["acquisition"]["sensors"],
     sr,
     ne,
     ns,
     wl,
     os, # trials offset
     nc,
     collect(keys(info["stim"]["labels"])), # clabels
     stimArtifact,
     cstim,
     [i for i=1:nc for j=1:length(cstim[i])], # y: all labels
     X, # whole EEG recording
     trials # all trials, by class
  )

end

####
function readNY(filename  :: AbstractString;
                bandpass  :: Tuple=(),
                resample  :: Union{Rational, Int}=1,
                getTrials :: Bool=true,
                msg       :: String="")

  data = npzread(splitext(filename)[1]*".npz") # read data file
  info = YAML.load(open(splitext(filename)[1]*".yml")) # read info file

  sr      = info["acquisition"]["samplingrate"]
  stim    = data["stim"]                  # stimulations
  (ns, ne)= size(data["data"])            # of sample, # of electrodes)
  os      = info["stim"]["offset"]        # offset for trial starting sample
  wl      = info["stim"]["windowlength"]  # trial duration
  nc      = info["stim"]["nclasses"]      # of classes

  # band-pass the data if requested
  if isempty(bandpass)
    X=data["data"]
  else
    BPfilter = digitalfilter(Bandpass(first(bandpass)/(sr/2), last(bandpass)/(sr/2)), Butterworth(2))
    X        = filtfilt(BPfilter, data["data"])
  end

  # resample data if requested
  if resample≠1
    X         = resample(X, resample; stim=stim)
    (ns, ne)  = size(X)
    wl        = round(Int, wl*resample)
    os        = round(Int, os*resample)
    sr        = round(Int, sr*resample)
    wl        = round(Int, wl*resample)
  end

  # vectors of samples where the trials start for each class 1, 2,...

  cstim=[[i+os for i in eachindex(stim) if stim[i]==j && i+os+wl<=ns] for j=1:nc]

  getTrials ?  trials=[X[cstim[i][j]:cstim[i][j]+wl-1,:] for i=1:nc for j=1:length(cstim[i])] :
                trials=nothing


  if !isempty(msg) println(msg) end

  # this creates the `EEG` structure
  EEG(
     info["id"],
     info["acquisition"],
     info["documentation"],
     info["formatversion"],

     info["id"]["database"],
     info["id"]["subject"],
     info["id"]["session"],
     info["id"]["run"],
     info["acquisition"]["sensors"],
     sr,
     ne,
     ns,
     wl,
     os, # trials offset
     nc,
     collect(keys(info["stim"]["labels"])), # clabels
     stim,
     cstim,
     [i for i=1:nc for j=1:length(cstim[i])], # y: all labels
     X, # whole EEG recording
     trials # all trials, by class
  )

end


# read EEG data from a .txt file in LORETA format and put it in a matrix
# of dimension txn, where n=#electrodes and t=#samples.
# If optional keyword argument `msg` is not empty, print `msg` on exit.
function readASCII(fileName::AbstractString; msg::String="")
    if !isfile(fileName)
        @error "function `readASCII`: file not found" fileName
        return nothing
    end

    S=readlines(fileName) # read the lines of the file as a vector of strings
    t=length(S) # number of samples
    n=length(split(S[1])) # get the number of electrodes
    X=Matrix{Float64}(undef, t, n) # declare the X Matrix
    for j=1:t
        x=split(S[j]) # this get the n potentials from a string
        for i=1:n
            X[j, i]=parse(Float64, x[i])
        end
    end
    if !isempty(msg) println(msg) end
    return X
end

# Read several EEG data from .txt files in LORETA format given in `filenames`
# (a Vector of strings) and put them in a vector of matrices object.
# `skip` is an optional vector of serial numbers of files in `filenames` to skip.
# print: "read file "*[filenumber]*": "*[filename] after each file has been read.
readASCII(fileNames::Vector{String}, skip::Vector{Int}=[]) =
        [readASCII(fileNames[f]; msg="read file $f: "*basename(fileNames[f])) for f in eachindex(fileNames) if f ∉ skip]


# Write an EEG data matrix into a text ASCII file in LORETA tabular format
# (# of samples x # of electrodes).
# The data is written as `filename` which must be a complete path from root.
# If `filename` already exists, if `overwrite` is true the file will be
# overwritten, otherwise a warning in printed and nothing is done.``
# `SamplesRange` is a UnitRange delimiting the samples (rows of `X`) to be written.
# If optional keyword argument `msg` is not empty, print `msg` on exit
function writeASCII(X::Matrix{T}, fileName::String;
              samplesRange::UnitRange=1:size(X, 1),
              overwrite::Bool=false,
              msg::String="") where T <: Real

    if isfile(fileName) && !overwrite
        @error "writeASCII function: `filename` already exists. Use argument `overwrite` if you want to overwrite it."
    else
        io = open(fileName, "w")
        write(io, replace(chop(string(X[samplesRange, :]); head=1, tail=1), ";" =>"\r\n" ))
        close(io)
        if !isempty(msg) println(msg) end
    end
end


# overwrite the Base.show function to nicely print information
# about the sturcure in the REPL
# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, o::EEG)
    r, c=size(o.X)
    type=eltype(o.X)
    l=length(o.stim)
    println(io, titleFont, "∿ EEG Data type; $r x $c ")
    println(io, separatorFont, "∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽", greyFont)
    println(io, "NY format info:")
    println(io, "Dict: id, acquisition, documentation")
    println(io, "formatversion   : $(o.formatversion)")
    println(io, separatorFont, "∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽", defaultFont)
    println(io, "db (database)   : $(o.db)")
    println(io, "subject         : $(o.subject)")
    println(io, "session         : $(o.session)")
    println(io, "run             : $(o.run)")
    println(io, "sensors         : $(length(o.sensors))-Vector{String}")
    println(io, "sr(samp. rate)  : $(o.sr)")
    println(io, "ne(# electrodes): $(o.ne)")
    println(io, "ns(# samples)   : $(o.ns)")
    println(io, "wl(win. length) : $(o.wl)")
    println(io, "offset          : $(o.offset)")
    println(io, "nc(# classes)   : $(o.nc)")
    println(io, "clabels(c=class): $(length(o.clabels))-Vector{String}")
    println(io, "stim(ulations)  : $(length(o.stim))-Vector{Int}")
    println(io, "cstim(ulations) : $([length(o.cstim[i]) for i=1:length(o.cstim)])-Vectors{Int}")
    println(io, "y (all c labels): $(length(o.y))-Vector{Int}")
    println(io, "X (EEG data)    : $(r)x$(c)-Matrix{$(type)}")
    o.trials==nothing ? println("                : nothing") :
    println(io, "trials          : $(length(o.trials))-Matrix{$(type)}")
    r≠l && @warn "number of class labels in y does not match the data size in X" l r
end


end # module

# Example
# dir="C:\\Users\\congedom\\Documents\\My Data\\EEG data\\NTE 84 Norms"

# Gat all file names with complete path
# S=getFilesInDir(dir) # in FileSystem.jl

# Gat all file names with complete path with extension ".txt"
# S=getFilesInDir(@__DIR__; ext=(".txt", ))

# read one file of NTE database and put it in a Matrix object
# X=readASCII(S[1])

# read all files of NTE database and put them in a vector of matrix
# 𝐗=readASCII(S)
