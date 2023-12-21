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
x = rand(Float32, input_dim, 5, 100) # 输入数据，维度为(10, 5, 100);
y = rand(Float32, output_dim, 5, 100) # 输出数据，维度为(1, 5, 100);

# 分解输入数据和输出数据成时间步的数组
x = Flux.unstack(x, 2); # x是一个长度为5的数组，每个元素是一个(10, 1, 100)的数组
y = Flux.unstack(y, 2); # y是一个长度为5的数组，每个元素是一个(1, 1, 100)的数组

# 打包输入数据和输出数据成一个迭代器
data = zip(x, y); # data是一个迭代器，每次迭代返回一个(x, y)的元组，其中x是一个(10, 1, 100)的数组，y是一个(1, 1, 100)的数组

# 训练模型
Flux.train!(loss, Flux.params(model), data, opt)

