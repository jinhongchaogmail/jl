using Flux

# 数据生成
train_x = rand(Float32,1000, 8, 10)
train_y = rand(Float32,1000,1)

# 检查数据维度
println("train_x dimensions: ", size(train_x))
println("train_y dimensions: ", size(train_y))

# LSTM 模型定义
model = Chain(
  LSTM(8, 32),
  Dense(32, 1),
  softmax)

# 损失函数定义
loss(x, y) = Flux.Losses.mse(model(x), y)

# 优化器定义
opt = ADAM()

# 串行训练
@time begin
  for i in 1:size(train_x, 3)
    Flux.train!(loss, Flux.params(model), [(train_x[:,:,i], train_y)], opt)
  end
end

# 并行训练
data = [(train_x[:,:,i], train_y) for i in 1:size(train_x, 3)]
@time Flux.train!(loss, Flux.params(model), data, opt)