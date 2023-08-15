class Model_Config_DEAP:
    K = 10

    nb_classes=4
    Chans=32
    Samples=512
    dropoutRate=0.5
    kernLength=64
    F1=8
    D=4
    F2=32
    dropoutType='Dropout'

    es_patience=50
    batch_size = 128
    max_epochs = 200
    lr_begin = 1e-3
    lr_end = 1e-5
    re_lr_patience = 10
    re_lr_factor = 0.1