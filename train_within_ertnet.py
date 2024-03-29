import logging
ertnet_logger = logging.getLogger('ertnet_logger')
ertnet_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs/ertnet.log')
ertnet_logger.addHandler(file_handler)

import numpy as np
from transformer import ertnet
from config import Model_Config_DEAP
import keras
from tensorflow.keras.utils import to_categorical
import utils.evaluate as evaluate
from sklearn.model_selection import KFold

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

cv_scores = []

# Load data
data = np.load('data_processed/train_data.npy')
labels = np.load('data_processed/train_labels.npy')
t_data = np.load('data_processed/test_data.npy')
t_labels = np.load('data_processed/test_labels.npy')

data = np.concatenate((data, t_data), axis=0)
labels = np.concatenate((labels, t_labels), axis=0)
labels = to_categorical(labels)

# Load Hyperparamters
params = Model_Config_DEAP

# 10-fold cross validation
kf = KFold(n_splits=params.K,shuffle=True,random_state=42)
i = 0
for train_index, test_index in kf.split(data):
    
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

    early_stop = EarlyStopping(monitor='val_loss', patience=params.es_patience, verbose=1)
    # Set model checkpoint callback
    checkpoint = ModelCheckpoint(filepath=f'model/model_best_within_t1_{i}.h5', 
                                 monitor='val_accuracy', 
                                 verbose=1, save_best_only=True, mode='max')
    # Load model
    model = ertnet(nb_classes=params.nb_classes, Chans=params.Chans, 
                   Samples=params.Samples, dropoutRate=params.dropoutRate, 
                   kernLength=params.kernLength, F1=params.F1, D=params.D, 
                   F2=params.F2)
    model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=params.lr_begin),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Set reduce learning rate callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=params.re_lr_factor, 
                                  patience=params.re_lr_patience, mode='auto', 
                                  min_lr=params.lr_end)
    # Fit model
    hist = model.fit(train_data, train_labels, epochs=params.max_epochs,
                     batch_size=params.batch_size,
                     validation_data=(test_data, test_labels),
                     callbacks=[reduce_lr, checkpoint, early_stop])
    # Get the best accuracy on the validation set
    cv_scores.append(max(list(hist.history['val_accuracy'])))

    model_best = ertnet(nb_classes=params.nb_classes, Chans=params.Chans,
                        Samples=params.Samples, dropoutRate=params.dropoutRate,
                        kernLength=params.kernLength, F1=params.F1, D=params.D,
                        F2=params.F2)
    model_best.load_weights(f'model/model_best_within_t1_{i}.h5')
    # Get the prediction on the test set
    pred = model_best.predict(test_data)
    # Plot ROC curve
    evaluate.plot_ROC(test_labels, pred, labels=['Excited', 'Content', 'Angry', 'Sad'], 
                      colorlist=['navy', 'pink', 'cyan', 'darkviolet'], 
                      file=f'picture/roc_within_t1_{i}.jpg',dpi=300)
    # Plot confusion matrix
    evaluate.plotHeatMap(np.argmax(test_labels,axis=1), np.argmax(pred,axis=1), 
                         ClassSet=['Excited', 'Content', 'Angry', 'Sad'], 
                         file=f'picture/hm_within_t1_{i}.jpg',dpi=300)
    ertnet_logger.info(f"{i} Fold Training Epochs Accuracy: {list(hist.history['val_accuracy'])}")
    i = i + 1
# Print model summary
model.summary()
# Save information to log file
print(cv_scores)
ertnet_logger.info(cv_scores)
print('Accuracy: %.2f%% (+/- %.2f%%)' % (np.mean(cv_scores)*100, np.std(cv_scores)*100))
ertnet_logger.info('Accuracy: %.2f%% (+/- %.2f%%)' % (np.mean(cv_scores)*100, np.std(cv_scores)*100))
print('ERTNet Train End.')
