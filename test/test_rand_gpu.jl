
using StatsBase,Plots,Flux,MLDataUtils

# LSTM 模型定义
model = Chain(
  LSTM(8, 32),
  Dense(32, 1),
  last)
# 损失函数定义
loss(x, y) = Flux.Losses.mse(model(x), y)
# 优化器定义
opt = ADAM()
train_losses = Float32[]
val_losses = Float32[]

# 数据生成
train_x = rand(Float32,8, 10,1000 )
train_y = rand(Float32,1000)
# x,y--->双集合 
(train_set, val_set) = splitobs(shuffleobs((train_x, train_y)), at = 0.8)
train_data=Flux.Data.DataLoader((copy(train_set[1]), copy(train_set[2])))
val_data=Flux.Data.DataLoader((copy(val_set[1]), copy(val_set[2])))

## 创建一个数组来存储损失值
loss_values = Float32[]
# 定义回调函数
callback() = Flux.reset!(model)
# 在训练中使用回调函数
@time for i in 1:100
Flux.train!(loss, Flux.params(model), train_data, opt,cb=callback)

if i % 10 == 0
  # 计算训练集的平均损失
  train_loss = mean(loss(x, y) for (x, y) in train_data)
  append!(train_losses, train_loss)

  # 计算验证集的平均损失
  val_loss = mean(loss(x, y) for (x, y) in val_data)
  append!(val_losses, val_loss)
  @info "Epoch $i. 训练集 loss: $(train_losses[end]). 验证集 loss: $(val_losses[end])"
end

end

p = plot(train_losses, xlabel="Time", ylabel="Loss", labels=["train_losses"], title="train Loss over time 9", size=(800, 600))
    plot!(p, val_losses, labels=["val_losses"])
