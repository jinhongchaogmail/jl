using Flux

# 定义输入数据的维度和输出标量的维度
input_dim = 10
output_dim = 1

# 定义LSTM层的隐藏单元数
hidden_dim = 20

# 定义LSTM模型，包括一个LSTM层和一个全连接层
model = Chain(
  Flux.LSTM(input_dim, hidden_dim), # 使用Flux.LSTM函数来创建LSTM层
  Dense(hidden_dim, output_dim),
  last # 添加一个last函数，只取最后一个时间步的输出
)

# 定义损失函数，这里使用均方误差
loss(x, y) = Flux.mse(model(x), y)

# 定义优化器，这里使用ADAM
opt = ADAM(0.01)

# 定义训练数据，这里假设有100个样本，每个样本有5个时间步，每个时间步有10个特征，输出是一个标量
# 直接生成三维数组，不需要进行转换
x = rand(Float32, input_dim, 100, 5) # 输入数据，维度为(10, 100, 5)
y = rand(Float32, output_dim, 100, 5) # 输出数据，维度为(1, 100, 5)

# 调整输入数据和输出数据的形状，让它们的最后一个维度是样本数，而时间步数是第二个维度
x = permutedims(x, (1, 3, 2)) # x的维度变为(10, 5, 100)
y = permutedims(y, (1, 3, 2)) # y的维度变为(1, 5, 100)

# 使用Flux.DataLoader来创建一个迭代器，指定批量大小为100，不打乱数据，保留最后一个不完整的批次
data = Flux.DataLoader(x, y, batchsize=100, shuffle=false, partial=true)

# 训练模型，在params前面加上Flux
Flux.train!(loss, Flux.params(model), data, opt)

