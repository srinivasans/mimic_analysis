'''
Script for predicting Hypertension from MIMIC data 
Author : srinivasan@cs.toronto.edu
'''
import os
import numpy as np
import pandas as pd

def predict_hypertension():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    patients = pd.read_csv(os.path.join(mimicdir, 'hypertension_patients.gz'), compression='gzip')
    charts = pd.read_csv(os.path.join(mimicdir, 'hypertension_charts.gz'), compression='gzip')
    print(patients.loc[:,'hadm_id'])
    print(list(charts))
    print(list(patients))

if __name__ == '__main__':
    predict_hypertension()