import logging
bay_logger = logging.getLogger('bay_logger')
bay_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs/bayesian.log')
bay_logger.addHandler(file_handler)

import optuna
import numpy as np
from transformer import eegtransformernet_1
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
data = np.load('data_processed/train_data.npy')
labels = np.load('data_processed/train_labels.npy')
labels = to_categorical(labels)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1)


# 定义目标函数
def objective_function(trial):
    # 超参数搜索范围
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    kern_length = trial.suggest_int("kern_length", 5, 101)
    heads = int(trial.suggest_discrete_uniform("heads", 4, 20, 4))
    F1 = int(trial.suggest_discrete_uniform("F1", 4, 20, 4))
    D = trial.suggest_int("D", 1, 5)
    F2 = trial.suggest_int("F2", 8, 64)
    
    # 创建模型
    model = eegtransformernet_1(nb_classes=4, dropoutRate=dropout_rate, kernLength=kern_length, heads=heads, Chans=32, Samples=512, F1=F1, D=D, F2=F2)
    
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=0, validation_data=(X_val, y_val))
    
    # 计算验证损失
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
    
    return val_loss

# 创建Optuna优化器
print('Creating optimizer')
study = optuna.create_study(direction="minimize")
print('Optimizer created')
# 搜索迭代次数
n_trials = 30

# 执行优化
print('Optimizing')
study.optimize(objective_function, n_trials=n_trials)
print('End')

# 输出最优超参数
best_params = study.best_params
best_value = study.best_value

print(f"最佳参数：{best_params}, 最佳损失：{best_value:.4f}")
bay_logger.info(f"最佳参数：{best_params}, 最佳损失：{best_value:.4f}")