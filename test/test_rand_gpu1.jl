
using StatsBase,Plots,Flux,MLDataUtils,CUDA
using Debugger

bsz=10
yangben=1000
epochs=100
shownum=10


# LSTM 模型定义
model = Chain(
  LSTM(8, 32),
  Dense(32, 1),
  )|> gpu
# 损失函数定义
loss(x, y) = Flux.Losses.mse(model(x), y)
# 优化器定义
opt = ADAM()


cudalossnum=Int(epochs/shownum)
train_losses = ones(cudalossnum)|> gpu
val_losses = ones(cudalossnum)|> gpu


# 数据生成
train_x = rand(Float32,8, 10,yangben )
train_y = rand(Float32,1,10,yangben )
# x,y--->双集合 
(train_set, val_set) = splitobs(shuffleobs((train_x, train_y)), at = 0.8)

train_data = Flux.Data.DataLoader((copy(train_set[1]) |> gpu, copy(train_set[2]) |> gpu), batchsize=bsz)
val_data = Flux.Data.DataLoader((copy(val_set[1]) |> gpu, copy(val_set[2]) |> gpu), batchsize=bsz)

# 定义回调函数
callback() = Flux.reset!(model)

# 在训练中使用回调函数
@time for i in 1:100
#Flux.train!(loss, Flux.params(model), train_data, opt,cb=callback)
# 获取第一个批次的数据
for (x, y)  in train_data
  Flux.reset!(model)
  loss_value = loss(x, y)
  gs = gradient(() -> loss_value, Flux.params(model))
  Flux.Optimise.update!(opt, Flux.params(model), gs)
end

if i % shownum == 0
     is=Int(i/shownum)
  # 计算训练集的平均损失
  train_loss = mean(loss(x, y) for (x, y) in train_data)
  train_losses[is]=train_loss

  # 计算验证集的平均损失
  val_loss = mean(loss(x, y) for (x, y) in val_data)
  val_losses[is]=val_loss
  #@info "Epoch $i. 训练集 loss: $(train_losses[end]). 验证集 loss: $(val_losses[end])"
end

end

p = plot(train_losses, xlabel="Time", ylabel="Loss", labels=["train_losses"], title="train Loss over time 9", size=(800, 600))
    plot!(p, val_losses, labels=["val_losses"])
