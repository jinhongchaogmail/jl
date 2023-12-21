using Pkg
Pkg.build("IJulia")
using IJulia 

notebook(;dir="/home/jin/workspace/",detached=true)
