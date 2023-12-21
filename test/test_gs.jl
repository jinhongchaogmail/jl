using Flux # 导入Flux包
using CUDA # 导入CUDA包，如果你用的是AMD的ＧＰＥ，可以用AMDGPU包代替

W = rand(2, 5) # 随机生成一个2x5的权重矩阵
b = rand(2) # 随机生成一个2维的偏置向量
predict(x) = W * x .+ b # 定义一个预测函数，用线性变换和广播加法
loss(x, y) = sum((predict(x) - y).^2) # 定义一个损失函数，用平方误差和求和

x, y = rand(5), rand(2) # 生成一些随机的输入和输出数据
x, y = gpu(x), gpu(y) # 把数据转移到ＧＰＵ上
W, b = gpu(W), gpu(b) # 把模型参数转移到ＧＰＵ上

gs = gradient(() -> loss(x, y), Flux.params(W, b)) # 计算梯度，用params函数把模型参数打包
gs[W] # 查看权重矩阵的梯度
gs[b] # 查看偏置向量的梯度




# 获取第一个批次的数据
for (x, y)  in train_data
println("获取第一个批次的数据完成")

# 前向传播：使用模型计算预测值
y_pred = model(x)
println("前向传播完成")

# 计算损失
loss_value = loss(x, y)
println("损失计算完成: ", loss_value)

# 创建一个新的梯度计算上下文，并计算梯度
gs = gradient(() -> loss_value, Flux.params(model))
println("梯度计算完成")

# 使用优化器和计算出的梯度更新模型的参数
Flux.Optimise.update!(opt, Flux.params(model), gs)
println("模型参数更新完成")
end


function train!(loss, ps::Params, data, opt::AbstractOptimiser; cb = () -> ())
  
  itrsz = Base.IteratorSize(typeof(data))
  n = (itrsz == Base.HasLength()) || (itrsz == Base.HasShape{1}()) ? length(data) : 0
  @withprogress for (i, d) in enumerate(data)
    l, gs = withgradient(ps) do
      loss(batchmemaybe(d)...)
    end
    if !isfinite(l)
      throw(DomainError("Loss is $l on data item $i, stopping training"))
    end
    update!(opt, ps, gs)
    cb()

    @logprogress iszero(n) ? nothing : i / n
  end
end

function train!(loss, model, data, opt; cb = nothing)
    isnothing(cb) || error("""train! does not support callback functions.
                              For more control use a loop with `gradient` and `update!`.""")
     for (i,d) in enumerate(data)
      d_splat = d isa Tuple ? d : (d,)
      l, gs = Zygote.withgradient(m -> loss(m, d_splat...), model)
      if !isfinite(l)
        throw(DomainError("Loss is $l on data item $i, stopping training"))
      end
      opt, model = Optimisers.update!(opt, model, gs[1])
      
    end
  end