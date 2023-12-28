# Date: 2021-08-31
#这个代码是用来跑optuna的，用来找最优参数的。如果在谷歌云盘上跑，需要先挂载云盘，然后把DB.h5和example.db放到云盘上，然后把下面的注释去掉。
"""#谷歌云盘时，需要先挂载云盘
!pip install optuna
!cp "/content/drive/MyDrive/DB.h5" .
!cp "/content/drive/MyDrive/example.db" .
"""
#region Imports
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import csv 
import optuna
from joblib import Parallel, delayed
import time
import shutil
#endregion
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device= 'cpu'
print(device)
torch.cuda.empty_cache()
def backup_db():
    while True:
        # 复制文件到Google Drive
        shutil.copy2('example.db', '/content/drive/MyDrive/example.db')
        
        # 暂停一小时
        time.sleep(60)
def prep_data(feature=9.5, w_size=10,n=1):
    with h5py.File("DB.h5", "r") as f:
        # 处理文件的代码
        ks = list(f.keys())
        X = []
        Y = []
        batch=len(ks)//3
        start=(n-1)*batch
        end=n*batch
        for i in range(start, min(end, len(ks))):  # replace with range(len(ks)) to process all keys
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
                    yield x.T, y.T
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
            i += w_size//3  # 跳过w_size个数据
        else:
            i += 1
    return indices
def train_and_validate(model, train_data, val_data):
    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
        # 更新学习率
        scheduler.step()
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

# 创建一个新的线程来运行备份操作默认备份到Google Drive
#backup_thread = threading.Thread(target=backup_db)
# 启动备份线程
#backup_thread.start()

# 定义优化的目标函数
def objective(trial):
    line = trial.suggest_int('line', 2, 16)
    num_layer = trial.suggest_int('num_layer', 2, 5)
    batch_size = trial.suggest_int('batch_size', 1, 512)
    feature = 9# trial.suggest_float('feature', 9,10)
    w_size = trial.suggest_int('w_size', 6, 31)
   
    data = list(prep_data(feature, w_size, 1))
    X3 = np.stack([x for x, _ in data], axis=0)
    Y3 = np.stack([y for _, y in data], axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X3, Y3, test_size=0.2, random_state=42)
    
    train_data = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)    
    model = MyModel(line, num_layer).to(device)
    start_time = time.time()
    score = train_and_validate(model, train_data, val_data)
    end_time = time.time()
           
    if score < 0.8:  # 如果loss低于阈值，进行1次验证
        print(f"正在进行第二次验证，score:{score},line: {line}, num_layers: {num_layer}, batch_size: {batch_size}, feature: {feature}, w_size: {w_size}")  
        data = list(prep_data(feature, w_size, 2))
        X3 = np.stack([x for x, _ in data], axis=0)
        Y3 = np.stack([y for _, y in data], axis=0)
        X_train, X_val, y_train, y_val = train_test_split(X3, Y3, test_size=0.2, random_state=42)
        
        train_data = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
        val_data = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)
        #这里是多次验证的数据获取
        total_score = 0
        n_validations = 5  # 验证次数
        for _ in range(n_validations):
            score = train_and_validate(model, train_data, val_data)
            total_score += score
        # 计算平均分数
        score = total_score / n_validations

        if score < 0.75:  # 如果loss低于阈值，进行2次验证
            print(f"正在进行第三次验证，score:{score},line: {line}, num_layers: {num_layer}, batch_size: {batch_size}, feature: {feature}, w_size: {w_size}")   
            data = list(prep_data(feature, w_size, 3))
            X3 = np.stack([x for x, _ in data], axis=0)
            Y3 = np.stack([y for _, y in data], axis=0)
            X_train, X_val, y_train, y_val = train_test_split(X3, Y3, test_size=0.2, random_state=42)
            
            train_data = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
            val_data = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)
            #这里是多次验证的数据获取
            total_score = 0
            n_validations = 5  # 验证次数
            for _ in range(n_validations):
                score = train_and_validate(model, train_data, val_data)
                total_score += score
            # 计算平均分数
            score = total_score / n_validations

    elapsed_time = end_time - start_time
    print(f"\033[31mElapsed time: {elapsed_time} seconds\033[0m")
    #print(f"score:{score},line: {line}, num_layers: {num_layer}, batch_size: {batch_size}, feature: {feature}, w_size: {w_size}")               
    return score            
def optimize(n_trials):
    # 使用optuna来优化超参数
    #study = optuna.create_study(direction='minimize')
    # 创建一个新的、持久化的 study
    study = optuna.create_study(study_name='my_study', storage='sqlite:///example.db', load_if_exists=True, direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value

# 使用joblib来并行运行多个试验
n_jobs = 4  # 你想要使用的进程数
results = Parallel(n_jobs=n_jobs, timeout=600)(
    delayed(optimize)(n_trials=500//n_jobs) for _ in range(n_jobs)
)
# 打印每个进程的最佳参数和最佳分数
for i, (best_params, best_value) in enumerate(results):
    print(f"Process {i}: Best parameters: {best_params}")
    print(f"Process {i}: Best score: {best_value}")