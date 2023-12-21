

ENV["JULIA_PKG_SERVER"]="https://mirrors.tuna.tsinghua.edu.cn/julia"
ENV["PYTHON"]="/usr/bin/python3"

using Pkg
Pkg.add("HDF5")
Pkg.add("StatsBase")
Pkg.add("Pandas")
Pkg.add("JLD2")
Pkg.add("PyPlot")
Pkg.add("Plots")
Pkg.add("PyCall")
Pkg.add("Dates")
Pkg.add("CSV")
Pkg.add("XLSX")
Pkg.add("DSP")
Pkg.add("DataFrames")
Pkg.add("ProgressMeter")
Pkg.add("IJulia")
Pkg.add("Flux")
Pkg.add("CUDA")
Pkg.add("MLDataUtils")
Pkg.add("Debugger")