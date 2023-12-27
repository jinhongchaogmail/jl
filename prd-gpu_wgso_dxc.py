import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import csv 
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
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
            # 将数据移动到GPU上
            X = X.to(device)
            y = y.to(device)

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
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

    # 返回验证损失的平均值
    return val_loss / len(val_data)
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

# 定义参数范围
lines = [6,8,12]
num_layers = [2, 3, 4]
batch_sizes=[1,2,3]
features=[9.6,9.8]
w_sizes=[60,70,80,90,100,110,120]
# 记录最佳参数和最佳分数
best_line = None
best_num_layers = None
best_batch_size=None
best_feature=None
best_w_size=None
best_score = float('inf')
import multiprocessing as mp
from multiprocessing import Pool
# 设置全局启动方法为'spawn'
mp.set_start_method('spawn', force=True)
def process_params(params):
    feature, w_size, batch_size, line, num_layer = params
    X_train, X_val, y_train, y_val = prep_data(feature, w_size)
    train_data = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)
    model = MyModel(line, num_layer).to(device)
    score = train_and_validate(model, train_data, val_data)
    return score, line, num_layer, batch_size, feature, w_size

if __name__ == '__main__':
    params = [(feature, w_size, batch_size, line, num_layer) 
              for feature in features
              for w_size in w_sizes
              for batch_size in batch_sizes
              for line in lines
              for num_layer in num_layers]

    with Pool(processes=4) as p:
        results = p.map(process_params, params)

    # 找到最佳分数和对应的参数
    best_score, best_line, best_num_layers, best_batch_size, best_feature, best_w_size = min(results)

    print(f"Best line: {best_line},\n best num_layers: {best_num_layers},\n best score: {best_score},\n best batch_size:{best_batch_size},\n best feature:{best_feature},\n best w_size:{best_w_size}")