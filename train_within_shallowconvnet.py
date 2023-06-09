import logging
shallowconvnet_logger = logging.getLogger('shallowconvnet_logger')
shallowconvnet_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs/shallowconvnet.log')
shallowconvnet_logger.addHandler(file_handler)

import numpy as np
#可选用的模型
from transformer import ShallowConvNet
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.utils import to_categorical
import evaluate
from sklearn.model_selection import KFold

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

cv_scores = []

data = np.load('data_processed/train_data.npy')
labels = np.load('data_processed/train_labels.npy')
t_data = np.load('data_processed/test_data.npy')
t_labels = np.load('data_processed/test_labels.npy')

data = np.concatenate((data, t_data), axis=0)
labels = np.concatenate((labels, t_labels), axis=0)
labels = to_categorical(labels)

kf = KFold(n_splits=10,shuffle=True,random_state=42)
i = 0
for train_index, test_index in kf.split(data):
    
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    #设置检查点, 保存最佳模型
    checkpoint = ModelCheckpoint(filepath=f'model/model_best_within_shallowconvnet_{i}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    #导入模型
    model = ShallowConvNet(nb_classes=4, Chans = 32, Samples = 512, dropoutRate = 0.5)
    model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.001),#学习率可调, 与下面的学习率自动更改要同步
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #设置学习率的自动变化
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,mode='auto',min_lr=1e-5)
    #训练模型
    hist = model.fit(train_data, train_labels, epochs=200,
                     batch_size=128,
                     validation_data=(test_data, test_labels),
                     callbacks=[reduce_lr, checkpoint, early_stop])
    #在提取最佳准确率
    cv_scores.append(max(list(hist.history['val_accuracy'])))


    model_best = ShallowConvNet(nb_classes=4, Chans = 32, Samples = 512, dropoutRate = 0.5)
    model_best.load_weights(f'model/model_best_within_shallowconvnet_{i}.h5')
    #取最后一个测试集进行测试绘图
    pred = model_best.predict(test_data)
    #绘制ROC曲线图
    evaluate.plot_ROC(test_labels, pred, labels=['Excited', 'Content', 'Angry', 'Sad'], colorlist=['navy', 'pink', 'cyan', 'darkviolet'], file=f'picture/roc_within_shallowconvnet_{i}.jpg',dpi=300)#添加file参数可以将图片保存
    #绘制混淆矩阵
    evaluate.plotHeatMap(np.argmax(test_labels,axis=1), np.argmax(pred,axis=1),ClassSet=['Excited', 'Content', 'Angry', 'Sad'], file=f'picture/hm_within_shallowconvnet_{i}.jpg',dpi=300)#添加file参数可以将图片保存
    shallowconvnet_logger.info(f"{i} Fold Training Epochs Accuracy: {list(hist.history['val_accuracy'])}")
    i = i + 1
#打印模型参数
model.summary()
#保存
print(cv_scores)
print('Accuracy: %.2f%% (+/- %.2f%%)' % (np.mean(cv_scores)*100, np.std(cv_scores)*100))
shallowconvnet_logger.info('Accuracy: %.2f%% (+/- %.2f%%)' % (np.mean(cv_scores)*100, np.std(cv_scores)*100))
print('shallowconvnet Train End.')
