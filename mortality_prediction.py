'''
Script for Mortality prediction from ICU data 
Author : srinivasan@cs.toronto.edu
'''

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
plt.style.use('ggplot')

mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
# Feature columns used for training (used to extract only the relevant features when the dataset changes)
feature_cols = ['age', 'first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN', 'admType_URGENT',
    'heartrate_min', 'heartrate_max', 'heartrate_mean', 'sysbp_min', 
    'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 
    'meanbp_min', 'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max', 
    'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 
    'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin', 
    'bicarbonate', 'bilirubin', 'creatinine','chloride', 'glucose', 'hematocrit', 
    'hemoglobin', 'lactate', 'magnesium', 'phosphate', 'platelet', 'potassium', 'ptt', 
    'inr', 'pt', 'sodium', 'bun', 'wbc']

# Non-binary data columns to be normalized
normalize_features = ['age', 'heartrate_min','heartrate_max','heartrate_mean', 'sysbp_min', 
    'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max', 'diasbp_mean', 
    'meanbp_min', 'meanbp_max', 'meanbp_mean', 'resprate_min', 'resprate_max', 
    'resprate_mean', 'tempc_min', 'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 
    'spo2_mean', 'glucose_min', 'glucose_max', 'glucose_mean','aniongap', 'albumin', 
    'bicarbonate', 'bilirubin', 'creatinine','chloride', 'glucose', 'hematocrit', 
    'hemoglobin', 'lactate', 'magnesium', 'phosphate', 'platelet', 'potassium','ptt', 
    'inr', 'pt', 'sodium', 'bun', 'wbc']

# Dictionary to store the feature mean, std, min and max for test time normalization
feature_max = {}
feature_min = {}
feature_mean = {}
feature_std = {}

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
Mean-Standard Normalization of features ()
'''
def mean_std_normalize(feature, feature_name, test=False):
    if not test:
        mean = feature.mean()
        std = feature.std()
        feature_mean[feature_name] = mean
        feature_std[feature_name] = std
    else:
        mean = feature_mean[feature_name]
        std = feature_std[feature_name]

    feature = (feature - mean) / (std)
    return feature

'''
Normalize the data by min-max or mean-std 
'''
def normalize_data(data, features, type='min_max', test=False):
    for feature in features:
        if type=='min_max':
            data[feature] = min_max_normalize(data[feature], feature, test)
        elif type=='mean_std':
            data[feature] = mean_std_normalize(data[feature], feature, test)
    return data

'''
Part 2(a) -  Logistic regression model to predict in-ICU mortality
'''
def predict_mortality(ensemble=False):
    data = pd.read_csv(os.path.join(mimicdir, 'adult_icu.gz'), compression='gzip')
    
    # Split Training and Test data
    train_data = data.loc[data['train'] == 1]
    test_data = data.loc[data['train'] == 0]

    # Target value (Mortality)
    mortality_target = ['mort_icu']

    # Formatting training data input
    train_X = train_data.loc[:,feature_cols]
    train_X = normalize_data(train_X, normalize_features,type='min_max', test=False)
    train_Y = train_data.loc[:,mortality_target]

    print(train_X.shape)
    print(train_Y.shape)

    # Formatting testing data input
    test_X = test_data.loc[:,feature_cols]
    test_X = normalize_data(test_X, normalize_features,type='min_max', test=True)
    test_Y = test_data.loc[:,mortality_target]

    # Initialize Logistic Regression Model with L2 regularization
    classifier = LogisticRegression(penalty='l2', C=1.0, random_state=0, solver='lbfgs', multi_class='ovr')
    classifier.fit(train_X, train_Y)
    W = classifier.coef_
    influence_weights = W.argsort(axis=-1) # Not absolute value as negative coeffs affect the model but reduce the risk factor 
    
    if not ensemble:
        # Obtain the feature indices that have the max and min influence on the model
        min_influence = influence_weights[0][0:5]
        max_influence = influence_weights[0][-5:][::-1]
        zero_influence = np.abs(W).argsort(axis=-1)[0][0:5]

        # Print the features with Max, Min and Zero influence
        print("High Risk Features : ", np.array(feature_cols)[max_influence])
        print("Low Risk Features : ", np.array(feature_cols)[min_influence])
        print("Zero Effect Features : ", np.array(feature_cols)[zero_influence])

        # Predict mortality on the test datasets
        pred_Y = classifier.predict_proba(test_X)[:,1]
        pred_Y_50 = classifier.predict(test_X)

        # Accuracy Score at 0.5 threshold
        accuracy = accuracy_score(test_Y, pred_Y_50)

        # Plot the ROC curve
        fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Mortality Prediction - ROC curve - ICU variables')
        plt.savefig('part2a_roc.png')

        # Compute Area under the ROC curve
        auc = roc_auc_score(test_Y, pred_Y)
        print("Accuracy: ", accuracy)
        print("AUC Score: ", auc)

    return classifier

'''
Part 2(b) -  Logistic regression model to predict in-ICU mortality from clinical notes
'''
def predict_mortality_from_notes(ensemble=False):
    data = pd.read_csv(os.path.join(mimicdir, 'adult_notes.gz'), compression='gzip')
    
    # Split Training and Test data
    train_data = data.loc[data['train'] == 1]
    test_data = data.loc[data['train'] == 0]
    train_data.reset_index(inplace = True, drop=True)
    test_data.reset_index(inplace = True, drop=True)
    print(train_data.shape)

    # Remove stopwords and punctuations
    stop = set(stopwords.words('english'))
    for i in range(0, train_data.shape[0]):
        text = train_data.at[i,'chartext']
        try:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
        except:
            tokens = ['']
        clean_tokens = [k for k in tokens if k not in stop]
        train_data.at[i,'chartext'] = (' ').join(clean_tokens)

    # Learn TF-IDF scores on train data
    vectorizer = TfidfVectorizer()
    train_X = vectorizer.fit_transform(train_data.loc[:,'chartext'].values.astype('U'))
    train_Y = train_data.loc[:,'mort_icu']

    # Learn a Logistic regression model with L1 regularizer on the TF-IDF vectors
    classifier = LogisticRegression(penalty='l1', C=1.0, random_state=0, multi_class='ovr')
    classifier.fit(train_X, train_Y)
    W = classifier.coef_
    influence_weights = W.argsort(axis=-1)

    if not ensemble:
        # Obtain the words of highest, lowest risk factor and zero influence
        min_influence = influence_weights[0][0:5]
        max_influence = influence_weights[0][-5:][::-1]
        zero_influence = np.abs(W).argsort(axis=-1)[0][0:5]
        
        print("High Risk Words : ",np.array(vectorizer.get_feature_names())[max_influence])
        print("Low Risk Words : ",np.array(vectorizer.get_feature_names())[min_influence])
        print("Zero Effect Words : ",np.array(vectorizer.get_feature_names())[zero_influence])
        
        # Remove stop words and punctuation in 
        stop = set(stopwords.words('english'))
        for i in range(0, test_data.shape[0]):
            text = test_data.at[i,'chartext']
            try:
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(text)
            except:
                tokens = ['']
            clean_tokens = [k for k in tokens if k not in stop]
            test_data.at[i,'chartext'] = (' ').join(clean_tokens)

        # Vectorize test data
        test_X = vectorizer.transform(test_data.loc[:,'chartext'].values.astype('U'))
        test_Y = test_data.loc[:,'mort_icu']

        # Predict test probabilities
        pred_Y = classifier.predict_proba(test_X)[:,1]
        pred_Y_50 = classifier.predict(test_X)

        # Accuracy Score at 0.5 threshold
        accuracy = accuracy_score(test_Y, pred_Y_50)

        # Plot the ROC curve
        fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Mortality Prediction - ROC curve - Notes')
        plt.savefig('part2b_roc.png')

        # Compute Area under the ROC curve
        auc = roc_auc_score(test_Y, pred_Y)
        print("Accuracy: ", accuracy)
        print("AUC Score: ", auc)

    return classifier, vectorizer

'''
Part 2(c) -  Ensemble model to predict in-ICU mortality from vitals and clinical notes
'''
def predict_mortality_from_vitals_and_notes():  
    vitals = pd.read_csv(os.path.join(mimicdir, 'adult_icu.gz'), compression='gzip')
    notes = pd.read_csv(os.path.join(mimicdir, 'adult_notes.gz'), compression='gzip')
    
    # Merging vitals and notes data by 
    data = vitals.merge(notes, how='left', left_on='icustay_id', right_on='icustay_id')
    
    # Split Training and Test data
    test_data = data.loc[data['train_x'] == 0]
    test_data = test_data.reset_index()
    mortality_target_vitals = ['mort_icu_x']
    mortality_target_notes = ['mort_icu_y']

    # Train and obtain the trained vitals and notes classifiers
    vitals_classifier = predict_mortality(ensemble=True)
    notes_classifier, vectorizer = predict_mortality_from_notes(ensemble=True)

    # Normalize the vitals test dataset
    vitals_test_X = test_data[feature_cols]
    vitals_test_X = normalize_data(vitals_test_X, normalize_features,type='min_max', test=True)
    vitals_test_Y = test_data[mortality_target_vitals]

    # Remove stop words and punctuations from notes test dataset
    stop = set(stopwords.words('english'))
    for i in range(0, test_data.shape[0]):
        text = test_data.at[i,'chartext']
        try:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
        except:
            tokens = ['']
        clean_tokens = [k for k in tokens if k not in stop]
        test_data.at[i,'chartext'] = (' ').join(clean_tokens)

    # Obtain the TF-IDF vector
    notes_test_X = vectorizer.transform(test_data.loc[:,'chartext'].values.astype('U'))
    notes_test_Y = test_data[mortality_target_notes]

    # This is an assert step to check if the target mortality values from vitals and notes data are the same
    checkeq = True
    if vitals_test_Y.shape[0] == notes_test_Y.shape[0]:
        for i in range(vitals_test_Y.shape[0]):
            if vitals_test_Y.at[i,'mort_icu_x']!=notes_test_Y.at[i,'mort_icu_y']:
                checkeq = True
    else:
        checkeq = False

    if not checkeq:
        print("Error: The predictions between notes and vitals data are not the same!")

    # Predict the mortality probabilities from vitals and notes data
    pred_Y_vitals = vitals_classifier.predict_proba(vitals_test_X)[:,1]
    pred_Y_notes = notes_classifier.predict_proba(notes_test_X)[:,1]

    # Ensemble model by combining the outputs of two models by averaging the probability outputs
    pred_Y = 0.5*(pred_Y_vitals+pred_Y_notes)
    test_Y = vitals_test_Y

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(test_Y, pred_Y, pos_label=1)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve - Ensemble (ICU Variables + Notes)')
    plt.savefig('part2c_roc.png')

    # Compute Area under the ROC curve
    auc = roc_auc_score(test_Y, pred_Y)
    print("AUC Score: ", auc)

if __name__ == '__main__':
    # Part 2(a)
    predict_mortality()
    # Part 2(b)
    predict_mortality_from_notes()
    # Part 2(c)
    predict_mortality_from_vitals_and_notes()