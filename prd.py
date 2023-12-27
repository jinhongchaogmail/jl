import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
feature=9
w_size=10
def prep_data(feature=9.5, w_size=10):
    f = h5py.File("DB.h5", "r")
    ks = list(f.keys())
    X = []
    Y = []
    for i in range(1000):  # replace with range(len(ks)) to process all keys
        arr = np.array(f[ks[i]])
        # Reverse the array and select columns 2 to 9
        arr = arr.astype(np.float32)
        # Find indices where the 6th row is greater than feature
        indices = find_features(arr, feature, w_size)
        # Calculate mean and standard deviation
        mu = np.mean(arr, axis=1, keepdims=True)
        std_dev = np.std(arr, axis=1, keepdims=True)
        # Normalize the array
        arr = (arr - mu) / std_dev
        for i in indices:
            if i > w_size:
                d = arr[:, i-w_size:i]
                x = d[:, :-1]
                y = d[5, -1]#.reshape(-1, 1)
                X.append(x.T)
                Y.append(y.T)
    print(len(X))
    # Convert list of arrays to 3D array
    X3 = np.stack(X, axis=0)
    Y3 = np.stack(Y, axis=0)
    f.close()
    # Split and shuffle the data
    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y3, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test
def find_features(arr, feature, w_size):
    """
    从DB.jld2上按特征提取训练数据。默认上涨4个点之前10日数据。
    参数:
    - `feature`: 一个整数，指定要使用的特征数。
    - `w_size`: 时间窗口的大小。  
    返回:
    包含训练张量的x列表,和包含对比y的列表。
    """
    indices = []
    i = 0
    while i < len(arr[5, :]):
        if arr[5, i] > feature:
            indices.append(i)
            i += w_size  # 跳过w_size个数据
        else:
            i += 1
    return indices
def train_and_validate(model, train_data, val_data):
    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    for epoch in range(100):
        for X, y in train_data:
            # 前向传播
            pred = model(X)
            loss = loss_fn(pred, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 验证模型
    val_loss = 0
    with torch.no_grad():
        for X, y in val_data:
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

    # 返回验证损失的平均值
    return val_loss / len(val_data)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = prep_data(feature,w_size)
# 定义参数范围
lines = [2,4,8,16, 32, 64]
num_layers = [2, 3, 4]

# 记录最佳参数和最佳分数
best_line = None
best_num_layers = None
best_score = float('inf')



#批次大小
batch_size=32
#连接线数量
line=32
num_layer=3
# 创建DataLoader
train_data = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
val_data = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)

# 1. 定义模型
class MyModel(nn.Module):
    def __init__(self,line, num_layer):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(8, line, num_layers=3, batch_first=True)
        self.linear = nn.Linear(line, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out.squeeze(1)
# 穷举所有参数组合
for line in lines:
    for num_layer in num_layers:
        # 2. 创建模型实例
        model = MyModel(line, num_layer)
        score = train_and_validate(model, train_data, val_data)
        # 如果这个分数比之前的分数好，就更新最佳参数和最佳分数
        if score < best_score:
            best_line = line
            best_num_layers = num_layer
            best_score = score

print(f"Best line: {best_line}, best num_layers: {best_num_layers}, best score: {best_score}")
