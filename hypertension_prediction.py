'''
Script for predicting Hypertension from MIMIC data 
Author : srinivasan@cs.toronto.edu
'''
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


def fit_model(train_data, columns):
    train_X = train_data[columns]
    print(train_X.shape)
    train_Y = train_data['hypertension']
    model = LogisticRegression(penalty='l2', C=3.0, random_state=0, solver='lbfgs', multi_class='ovr', max_iter=10000)
    model.fit(train_X, train_Y)
    return model

def test_model(model, test_data, columns):
    test_X = test_data[columns]
    test_Y = test_data['hypertension']

    pred_Y = model.predict_proba(test_X)[:,1]
    pred_Y_lab = model.predict(test_X)
    
    fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
    plt.plot(fpr, tpr)
    plt.show()
    auc = roc_auc_score(test_Y, pred_Y)
    f1 = f1_score(test_Y, pred_Y_lab)
    p, r , th= precision_recall_curve(test_Y, pred_Y)
    plt.plot(p,r)
    plt.show()
    print(auc)
    print(f1)

def predict_hypertension():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    patients = pd.read_csv(os.path.join(mimicdir, 'hypertension_patients.gz'), compression='gzip')
    charts = pd.read_csv(os.path.join(mimicdir, 'hypertension_charts.gz'), compression='gzip')
    print(patients.loc[:,'hadm_id'])
    print(list(charts))
    print(list(patients))

    means = pd.DataFrame(charts.groupby(['hadm_id','itemid'],axis=0)['valuenum'].mean())
    maxes = pd.DataFrame(charts.groupby(['hadm_id','itemid'],axis=0)['valuenum'].max())
    mins = pd.DataFrame(charts.groupby(['hadm_id','itemid'],axis=0)['valuenum'].min())

    means = means.reset_index().pivot('hadm_id', 'itemid', 'valuenum')
    maxes = maxes.reset_index().pivot('hadm_id', 'itemid', 'valuenum')
    mins = mins.reset_index().pivot('hadm_id', 'itemid', 'valuenum')

    feature_map = {220045: 'heartrate', 220210: 'rbpm', 220277:'spo2', 220181:'bpmm'}

    means.rename(columns=feature_map, inplace=True)
    means = means.add_suffix('_mean')
    maxes.rename(columns=feature_map, inplace=True)
    maxes = maxes.add_suffix('_max')
    mins.rename(columns=feature_map, inplace=True)
    mins= mins.add_suffix('_min')

    data = pd.concat([mins, maxes, means],axis=1)
    data = data.merge(patients, how='left', left_on='hadm_id', right_on='hadm_id')
    data.dropna(axis=0, inplace=True)

    hr_columns = ['heartrate_min', 'heartrate_max', 'heartrate_mean']
    rbpm_columns = ['rbpm_min', 'rbpm_max', 'rbpm_mean']
    spo2_columns = ['spo2_min', 'spo2_max', 'spo2_mean']
    bpmm_columns = ['bpmm_min', 'bpmm_max', 'bpmm_mean']

    train_data = data[data['train']==1]
    test_data = data[data['train']==0]

    # Evaluate Heart Rate model
    model = fit_model(train_data, hr_columns)
    test_model(model, test_data, hr_columns)
    print(model.n_iter_)

    # Evaluate Respiratory BPM model
    model = fit_model(train_data, rbpm_columns)
    test_model(model, test_data, rbpm_columns)
    print(model.n_iter_)

    # Evaluate SpO2 Model
    model = fit_model(train_data, spo2_columns)
    test_model(model, test_data, spo2_columns)
    print(model.n_iter_)

    # Evaluate BPMM Model
    model = fit_model(train_data, bpmm_columns)
    test_model(model, test_data, bpmm_columns)
    print(model.n_iter_)

    # Evaluate using all the features
    model = fit_model(train_data, hr_columns+rbpm_columns+spo2_columns+bpmm_columns)
    test_model(model, test_data, hr_columns+rbpm_columns+spo2_columns+bpmm_columns)
    print(model.n_iter_)


def build_lstm_model():
    model = Sequential()
    model.add(LSTM(1, activation='tanh', input_shape=(None, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def predict_hypertension_lstm():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    patients = pd.read_csv(os.path.join(mimicdir, 'hypertension_patients.gz'), compression='gzip')
    charts = pd.read_csv(os.path.join(mimicdir, 'hypertension_charts.gz'), compression='gzip')
    data = charts.merge(patients, how='left', left_on='hadm_id', right_on='hadm_id')

    train_data = data[data['train'] == 1]
    test_data = data[data['train'] == 0]

    # Get the HeartRate column in train data
    heartrate_train = train_data[train_data['itemid']==220045]
    hrgroup_train = heartrate_train.groupby(['hadm_id','hypertension'], axis=0)
    keys = hrgroup_train.groups.keys()

    train_X = []
    train_Y = []

    for hadm_id, y in keys:
        trainx = np.array(hrgroup_train.get_group((hadm_id,y)).sort_values('charttime')['valuenum'])
        train_X.append([[i] for i in list(trainx)])
        train_Y.append(y)
    
    model = build_lstm_model()
    gen = batch_generator(train_X, train_Y)
    model.fit_generator(gen,steps_per_epoch=len(train_X), epochs=5)
    model.save('HeartRateLSTM.h5')

    # Test Data
    heartrate_test = test_data[train_data['itemid']==220045]
    hrgroup_test = heartrate_test.groupby(['hadm_id','hypertension'], axis=0)
    keys = hrgroup_test.groups.keys()

    test_X = []
    test_Y = []

    for hadm_id, y in keys:
        testx = np.array(hrgroup_test.get_group((hadm_id,y)).sort_values('charttime')['valuenum'])
        test_X.append([[i] for i in list(testx)])
        test_Y.append(y)
    
    gen = batch_generator(test_X, test_Y, train=False)
    pred_Y = model.predict_generator(gen, steps = len(test_X))

    fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
    plt.plot(fpr, tpr)
    plt.show()
    auc = roc_auc_score(test_Y, pred_Y)
    print(auc)

def batch_generator(train_X, train_Y, train=True):
    idx = 0
    while True:
        idx = (idx+1)%len(train_X)
        if train:
            yield (np.array([train_X[idx]]), np.array([train_Y[idx]]))
        else:
            yield (np.array([train_X[idx]]))
        idx+=1


if __name__ == '__main__':
    #predict_hypertension() -> Split the columns ###
    predict_hypertension_lstm()