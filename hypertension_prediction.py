'''
Script for predicting Hypertension from MIMIC data 
Author : srinivasan@cs.toronto.edu
'''

def predict_hypertension():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    data = pd.read_csv(os.path.join(mimicdir, 'adult_icu.gz'), compression='gzip')
    print(list(data))

if __name__ == '__main__':
    predict_hypertension()