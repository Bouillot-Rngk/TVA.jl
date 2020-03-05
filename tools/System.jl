module System

# functions:

# waste | free the memory for all objects passed as arguments

export
  waste,
  charlie

# free memory for all arguments passed as `args...`
# see https://github.com/JuliaCI/BenchmarkTools.jl/pull/22
function waste(args...)
  for a in args a=nothing end
  for i=1:4 GC.gc(true) end
end

# if b is true, print a warning with the `msg` and return true,
# otherwise return false. This is used within functions
# to make a check and if necessary print a message and return.
# Example: charlie(type ≠ :s && type ≠ :t, "my message") && return
charlie(b::Bool, msg::String) = b ? (@warn msg; return true) : return false

end
