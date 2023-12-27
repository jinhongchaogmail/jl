#把jld2存储的数据表，转换为h5存储的数据表 并转换成float32，及上下翻转

using HDF5, JLD2

# 打开JLD2文件
jld = jldopen("DB.jld2", "r")

# 创建HDF5文件
h5 = h5open("DB.h5", "w")

# 将JLD2文件中的每个数据集复制到HDF5文件中
for name in keys(jld)
    data = jld[name]
    println(name,size(data))

    # 去掉第一列和第一行
    data = data[2:end, 2:end]

    # 上下翻转
    data = reverse(data, dims=1)
# 将数据转换为具体类型
data = convert(Array{Float32}, data)
    write(h5, name, data)
end

# 关闭文件
close(jld)
close(h5)