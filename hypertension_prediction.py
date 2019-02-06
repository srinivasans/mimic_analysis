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
plt.style.use('ggplot')

mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")

# Dictionary to store the feature mean, std, min and max for test time normalization
feature_max = {}
feature_min = {}

'''
Min-Max Normalization of features
'''
def min_max_normalize(feature, feature_name, test=False):
    if not test:
        max_val = feature.max()
        min_val = feature.min()
        feature_max[feature_name] = max_val
        feature_min[feature_name] = min_val
    else:
        max_val = feature_max[feature_name]
        min_val = feature_min[feature_name]
    feature = (feature - min_val) / (max_val - min_val)
    return feature

'''
Fit a logistic regression model using feature columns specified for hypertension prediction
'''
def fit_model(train_data, columns):
    # Remove rows with NA in the selected columns
    clean_data = train_data[columns+['hypertension']].dropna(axis=0)
    train_X = clean_data[columns]
    print(train_X.shape)
    # Normalize Data
    for feature in columns:
        train_X[feature] = min_max_normalize(train_X[feature], feature, test=False)    
    train_Y = clean_data['hypertension']
    # Train Logistic regression model
    model = LogisticRegression(penalty='l2', C=3.0, random_state=0, solver='lbfgs', multi_class='ovr', max_iter=10000)
    model.fit(train_X, train_Y)
    return model

'''
Evaluate a logistic regression model using feature columns specified for hypertension prediction
'''
def test_model(model, test_data, columns, name):
    clean_data = test_data[columns+['hypertension']].dropna(axis=0)
    test_X = clean_data[columns]
    # Normalize Data
    for feature in columns:
        test_X[feature] = min_max_normalize(test_X[feature], feature, test=True)  

    test_Y = clean_data['hypertension']
    # Evaluate Logistic Regression model
    pred_Y = model.predict_proba(test_X)[:,1]
    pred_Y_lab = model.predict(test_X)

    # Plot ROC curve and calculate AUC
    fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Hypertension Prediction - ROC curve - %s'%name)
    plt.savefig('part3a_roc_%s.png'%name)
    
    auc = roc_auc_score(test_Y, pred_Y)
    f1 = f1_score(test_Y, pred_Y_lab)
    print("AUC Score : %f"%auc)
    print("F1-Score : %f"%f1)

'''
Part 3(a) - Function to predict hypertension from patient charts data
'''
def predict_hypertension():
    patients = pd.read_csv(os.path.join(mimicdir, 'hypertension_patients.gz'), compression='gzip')
    charts = pd.read_csv(os.path.join(mimicdir, 'hypertension_charts.gz'), compression='gzip')
    
    # Group the values by the admission ID and obtain aggregates of mean, min and max
    # Removing patients with less than 2 measurements
    charts_gt1 = charts.groupby(['hadm_id','itemid'],axis=0).filter(lambda x : len(x)>=2)
    means = pd.DataFrame(charts_gt1.groupby(['hadm_id','itemid'],axis=0)['valuenum'].mean())
    maxes = pd.DataFrame(charts_gt1.groupby(['hadm_id','itemid'],axis=0)['valuenum'].max())
    mins = pd.DataFrame(charts_gt1.groupby(['hadm_id','itemid'],axis=0)['valuenum'].min())

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

    # Merge the datasets (min, mean and max) with patient data to obtain target value
    data = pd.concat([mins, maxes, means],axis=1)
    data = data.merge(patients, how='left', left_on='hadm_id', right_on='hadm_id')

    hr_columns = ['heartrate_min', 'heartrate_max', 'heartrate_mean']
    rbpm_columns = ['rbpm_min', 'rbpm_max', 'rbpm_mean']
    spo2_columns = ['spo2_min', 'spo2_max', 'spo2_mean']
    bpmm_columns = ['bpmm_min', 'bpmm_max', 'bpmm_mean']

    train_data = data[data['train']==1]
    test_data = data[data['train']==0]

    # Evaluate Heart Rate model
    print("Prediction with Heart Rate feature")
    model = fit_model(train_data, hr_columns)
    test_model(model, test_data, hr_columns,'heartrate')
    print(model.n_iter_)

    # Evaluate Respiratory BPM model
    print("Prediction with Respiratory BPM feature")
    model = fit_model(train_data, rbpm_columns)
    test_model(model, test_data, rbpm_columns, 'rbpm')
    print(model.n_iter_)

    # Evaluate SpO2 Model
    print("Prediction with SpO2 feature")
    model = fit_model(train_data, spo2_columns)
    test_model(model, test_data, spo2_columns, 'spo2')
    print(model.n_iter_)

    # Evaluate BPMM Model
    print("Prediction with BPMM feature")
    model = fit_model(train_data, bpmm_columns)
    test_model(model, test_data, bpmm_columns, 'bpmm')
    print(model.n_iter_)

    # Evaluate using all the features
    print("Prediction with all the features combined")
    model = fit_model(train_data, hr_columns+rbpm_columns+spo2_columns+bpmm_columns)
    test_model(model, test_data, hr_columns+rbpm_columns+spo2_columns+bpmm_columns, 'all')
    print(model.n_iter_)


'''
Data Generator for LSTM
'''
def batch_generator(train_X, train_Y, train=True):
    idx = 0
    while True:
        idx = (idx+1)%len(train_X)
        if train:
            yield (np.array([train_X[idx]]), np.array([train_Y[idx]]))
        else:
            yield (np.array([train_X[idx]]))
        idx+=1

'''
Build computational graph for the model with an LSTM unit and one Dense Layer
'''
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(32, activation='tanh', input_shape=(None, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

'''
Train LSTM Model
'''
def fit_lstm_model(train, name):
    group_train = train.groupby(['hadm_id', 'hypertension'], axis=0)
    # Group by admission ID and remove rows with < 2 feature measurements
    group_train = group_train.filter(lambda x : len(x)>=2).groupby(['hadm_id', 'hypertension'], axis=0)
    keys = group_train.groups.keys()

    train_X = []
    train_Y = []

    # Create training data with each admission ID (patient) having list of measurement values sorted by time
    for hadm_id, y in keys:
        trainx = np.array(group_train.get_group((hadm_id,y)).sort_values('charttime')['valuenum'])
        train_X.append([[i] for i in list(trainx)])
        train_Y.append(y)
    
    # Build the computational graph and learn the model
    model = build_lstm_model()
    gen = batch_generator(train_X, train_Y)
    model.fit_generator(gen,steps_per_epoch=len(train_X), epochs=5)
    model.save('%s_LSTM.h5'%name)
    return model

'''
Test LSTM Model
'''
def test_lstm_model(test, model, name):
    group_test = test.groupby(['hadm_id', 'hypertension'], axis=0)
    # Group by admission ID and remove rows with < 2 feature measurements
    group_test = group_test.filter(lambda x : len(x)>=2).groupby(['hadm_id', 'hypertension'], axis=0)
    keys = group_test.groups.keys()

    test_X = []
    test_Y = []

    # Creating test data and sorting the values by time
    for hadm_id, y in keys:
        testx = np.array(group_test.get_group((hadm_id,y)).sort_values('charttime')['valuenum'])
        test_X.append([[i] for i in list(testx)])
        test_Y.append(y)
    
    # Predict outcomes for the test data
    gen = batch_generator(test_X, test_Y, train=False)
    pred_Y = model.predict_generator(gen, steps = len(test_X))
    pred_Y_lab = (pred_Y>0.5).astype(np.uint8)

    # Plot ROC and compute AUC and F1 score
    fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Hypertension (LSTM) - ROC curve - %s'%name)
    plt.savefig('part3b_roc_%s.png'%name)
    
    auc = roc_auc_score(test_Y, pred_Y)
    f1 = f1_score(test_Y, pred_Y_lab)
    print("AUC Score : %f"%auc)
    print("F1-Score : %f"%f1)


'''
Part 3(b) - Predict hypertension from the patient charts data
'''
def predict_hypertension_lstm():
    patients = pd.read_csv(os.path.join(mimicdir, 'hypertension_patients.gz'), compression='gzip')
    charts = pd.read_csv(os.path.join(mimicdir, 'hypertension_charts.gz'), compression='gzip')
    data = charts.merge(patients, how='left', left_on='hadm_id', right_on='hadm_id')

    train_data = data[data['train'] == 1]
    test_data = data[data['train'] == 0]

    #  List of features to be used individually for building the LSTM model
    feature_map = {220045: 'heartrate', 220210: 'rbpm', 220277:'spo2', 220181:'bpmm'}

    # Train and evaluate model on different features
    for feature in feature_map.keys():    
        print("Evaluating LSTM model with %s feature data"%feature_map[feature])
        train = train_data[train_data['itemid']==feature]
        # Train the model for each feature
        train['valuenum'] = min_max_normalize(train['valuenum'], feature, test=False)
        model = fit_lstm_model(train, feature_map[feature])
        
        # Evaluate the model for each feature 
        test = test_data[test_data['itemid']==feature]
        test['valuenum'] = min_max_normalize(test['valuenum'], feature, test=True)
        test_lstm_model(test,model,feature_map[feature])
    

if __name__ == '__main__':
    #predict_hypertension()
    predict_hypertension_lstm()