using Weave
using Plots
using LinearAlgebra
using Statistics
using Random
using Images 
using Printf
function MakeResponse(doc::Vector{String}, src::Vector{String}, f, lines)

    global f

    # Report
    for i =  eachindex(doc)
        f = open(joinpath(pwd(),"doc", doc[i]))
        lines = cat(lines,readlines(f); dims=1)
        close(f)
    end 

    # Source Code 
    lines = cat(lines,"## Appendix: Source Code Listings"; dims=1)

    for i =  eachindex(src)
        lines = cat(lines,"##### "*src[i]; dims=1)
        lines = cat(lines,"```julia; eval = false"; dims=1)
        f = open(joinpath(pwd(),"src", src[i]))
        lines = cat(lines,readlines(f); dims=1)
        close(f)
        lines = cat(lines,"```"; dims=1)

    end 
    return f, lines
end

#--------------------------------------------------

# PROBLEM 1 RESPONSE
f = open(joinpath(pwd(),"doc","Problem1Response.jmd"))
lines = readlines(f)
lines = cat(lines,readlines(f); dims=1)
close(f)

#--------------------------------------------------

# PROBLEM RESPONSES
ProblemResponse = ["Problem2Response.jmd", "Problem3Response.jmd", "Problem4Response.jmd", "Problem5Response.jmd", "Problem6Response.jmd"]

#--------------------------------------------------

# PROBLEM
Problem = ["Problem1.jl", "Problem2.jl", "Problem3.jl", "Problem4.jl", "Problem5.jl"]

f, lines = MakeResponse(ProblemResponse, Problem, f, lines)

#-------------------------------------------------

# WRITE REPORT FILE
fout = open("doc//ProjectReport.jmd","w")
for x in lines
    println(fout, x)
end
close(fout)
weave("doc//ProjectReport.jmd")
