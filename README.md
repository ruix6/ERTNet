# File Tree:
|-  README.md
|-  requirements.txt
|-  logs  # Path: store logs
|-  data_processed  # Path: store processed data
|-  model  # Path: store  trained model
|-  picture  # Path: store pictures in 10-fold cross validation
|-  bayesian_optimizier.py  # File: bayesian optimization for finding best hyperparameters
|-  train_within_*.py  # File: train model within 10-fold cross validation
|-  transformer.py  # File: model's code
|-  visualize.py  # File: visualize the signals
|-  evaluate.py  # File: evaluate the model contains accuracy, precision, recall, f1-score, confusion matrix, ROC curves
