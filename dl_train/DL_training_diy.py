# Python script to train a neural network using Keras library.

import os
import zipfile
import time
"""
chemin_absolu = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../tensorflow"))

# Ajouter ce chemin au path Python
os.path.append(chemin_absolu)

# Maintenant, tu peux importer ton module
import tensorflow
"""



start_time = time.time()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas
import numpy as np
import random
import sys


with tensorflow.device("/gpu:0"):


    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Conv1D
    from tensorflow.keras.layers import MaxPooling1D

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import ModelCheckpoint
    from datetime import datetime

    random.seed(datetime.now())

    # model_type
    # 1: both left and right sides of time series are padded
    # 2: only left side of time series is padded

    # Get command line parameters
    model_type = int(sys.argv[1])
    kk = int(sys.argv[2]) # index for NN
    print('Train a classifier of type {} with index {}'.format(model_type, kk))

    # model_type = 1
    # kk = 1

    # Set size of training library and length of time series
    (lib_size, ts_len) = (500000, 500) # Or (500000, 500) for the 500-classifier


    # keeps track of training metrics
    f1_name = 'training_results_{}_{}.txt'.format(kk, model_type)
    f2_name = 'training_results_{}_{}.csv'.format(kk, model_type)

    f_results= open(f1_name, "w")
    f_results2 = open(f2_name, "w")


    if model_type==1:
        pad_left = 225 if ts_len==500 else 725
        pad_right = 225 if ts_len==500 else 725

    if model_type==2:
        pad_left = 450 if ts_len==500 else 1450
        pad_right = 0
        

    # get zipfile of time series 
    print('Load in zip file containing training data')
    zf = zipfile.ZipFile('../training_data/output_full/ts_{}/combined/output_resids.zip'.format(ts_len))
    text_files = zf.infolist()
    sequences = list()


    print('Extract time series from zip file')
    tsid_vals = np.arange(1,lib_size+1)
    counter=0
    a=0
    for tsid in tsid_vals:
        df = pandas.read_csv(zf.open('output_resids/resids'+str(tsid)+'.csv'))
        values = df[['Residuals']].values
        sequences.append(values)
        counter+=1
        if counter>10000:
            a+=1
            print(a)
            counter=0
        

    sequences = np.array(sequences)
    print("extracted!")
    # Get target labels for each data sample
    df_targets = pandas.read_csv('../training_data/output_full/ts_{}/combined/labels.csv'.format(ts_len),
                            index_col='sequence_ID')

    # train/validation/test split denotations
    df_groups = pandas.read_csv('../training_data/output_full/ts_{}/combined/groups.csv'.format(ts_len),
                                index_col='sequence_ID')

    #Padding input sequences
    print("padding...")
    for i, tsid in enumerate(tsid_vals):
    # for i in range(lib_size):
        pad_length = int(pad_left*random.uniform(0, 1))
        for j in range(0,pad_length):     
            sequences[i,j] = 0

        pad_length = int(pad_right*random.uniform(0, 1))
        for j in range(ts_len - pad_length, ts_len):
            sequences[i,j] = 0
    print("padded")
    print("normalizing")
    # normalizing input time series by the average. 
    for i, tsid in enumerate(tsid_vals):
    # for i in range(lib_size):
        values_avg = 0.0
        count_avg = 0
        for j in range (0,ts_len):
            if sequences[i,j] != 0:
                values_avg = values_avg + abs(sequences[i,j])
                count_avg = count_avg + 1
        if count_avg != 0:
            values_avg = values_avg/count_avg
            for j in range (0,ts_len):
                if sequences[i,j] != 0:
                    sequences[i,j] = sequences[i,j]/values_avg
    print("normalized!")
    final_seq = sequences

    # apply train/test/validation labels
    train = [final_seq[i] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid]==1]
    validation = [final_seq[i] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid]==2]
    test = [final_seq[i] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid]==3]


    train_target = [df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid]==1]
    validation_target = [df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid]==2]
    test_target = [df_targets['class_label'].loc[tsid] for i, tsid in enumerate(tsid_vals) if df_groups['dataset_ID'].loc[tsid]==3]  
    print("splitted!")

    train = np.array(train)
    validation = np.array(validation)
    test = np.array(test)

    train_target = np.array(train_target)
    validation_target = np.array(validation_target)
    test_target = np.array(test_target)


    # hyperparameter settings
    CNN_layers = 1
    LSTM_layers = 1  
    pool_size_param = 2 #pourrait l'augmenter mais on ne va pas le faire
    learning_rate_param = 0.0005      #on pourrait l'augmenter
    batch_param = 250000 #1000
    dropout_percent = 0.10
    filters_param = 50  - 20 #on réduit le nombre de filtres
    mem_cells = 50 + 50 #pour contrebalancer la suppresion de l'autre couche LSTM
    mem_cells2 = 10
    kernel_size_param = 12 -3 #pour reduire le cout de computation
    epoch_param = 1000 #1500 antes
    initializer_param = 'lecun_normal'


    model = Sequential()

    # add layers

    model.add(Conv1D(filters=filters_param, kernel_size=kernel_size_param, activation='relu', padding='same',input_shape=(ts_len, 1),kernel_initializer = initializer_param))
    #pourrait utiliser Leakyrelu instead
    model.add(Dropout(dropout_percent))
    model.add(MaxPooling1D(pool_size=pool_size_param))


    model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer = initializer_param))
    model.add(Dropout(dropout_percent))

    #model.add(LSTM(mem_cells2,kernel_initializer = initializer_param)) on retire un LSTM car très couteux et on augmente memcelle de l'autre pour contrebalancer
    #model.add(Dropout(dropout_percent))
    model.add(Dense(4, activation='softmax',kernel_initializer = initializer_param))

    # name for output pickle file containing model info
    model_name = 'best_model_diy_{}_{}_length{}.pkl'.format(kk,model_type,ts_len)
    print("estamos aqui")
    # Set up optimiser
    adam = Adam(learning_rate=learning_rate_param)
    chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

    # Train model
    history = model.fit(train, train_target, epochs=epoch_param, batch_size=batch_param, callbacks=[chk], validation_data=(validation,validation_target))


    model = load_model(model_name)

    # generate test metrics
    from sklearn.metrics import accuracy_score
    test_preds = model.predict_classes(test)
    accuracy_score(test_target, test_preds)

    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import classification_report 

    print(classification_report(test_target, test_preds, digits=3))
    print(history.history['accuracy'])
    print(history.history['val_accuracy'])
    print(history.history['loss'])
    print(history.history['val_loss'])
    print("F1 score:",f1_score(test_target, test_preds, average='macro'))
    print("Precision: ",precision_score(test_target, test_preds, average="macro"))
    print("Recall: ",recall_score(test_target, test_preds, average="macro"))    
    print("Confusion matrix: \n",confusion_matrix(test_target, test_preds))

    f_results.write("Simulation %d macro f1: %f. macro avg precision %f.  macro avg recall %f. Kernel size %d.  Filters %d.  Batch size %d.  Epochs %d.  seq_len %d.  Set size %d.  Mem_cells %d. Mem_cells2 %d. dropout %f. CNN layers %d, LSTM layers %d, pool_size_param %d, learning_rate_param %f.  initializer %s. pad_left %d.  pad_right %d \r\n" % (kk, f1_score(test_target, test_preds, average='macro'),precision_score(test_target, test_preds, average='macro'),recall_score(test_target, test_preds, average='macro'),kernel_size_param,filters_param,batch_param,epoch_param,ts_len,lib_size,mem_cells,mem_cells2,dropout_percent,CNN_layers,LSTM_layers,pool_size_param,learning_rate_param,initializer_param,pad_left, pad_right))
    f_results.flush()

    f_results2.write("%d, %f, %f, %f, %d, %d, %d, %d, %d, %d, %d, %d, %f, %d, %d, %d, %f, %s, %d, %d \r\n" % (kk, f1_score(test_target, test_preds, average='macro'),precision_score(test_target, test_preds, average='macro'),recall_score(test_target, test_preds, average='macro'),kernel_size_param,filters_param,batch_param,epoch_param,ts_len,lib_size,mem_cells,mem_cells2,dropout_percent,CNN_layers,LSTM_layers,pool_size_param,learning_rate_param,initializer_param,pad_left, pad_right))
    f_results2.flush()

    f_results.close()
    f_results2.close()

    end_time = time.time()
    print("Temps d'exécution : ")
    print(end_time-start_time)


