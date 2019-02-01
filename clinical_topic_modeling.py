'''
Script for predicting Hypertension from MIMIC data 
Author : srinivasan@cs.toronto.edu
'''

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from pprint import pprint


def model_topics_from_notes():
    mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")
    data = pd.read_csv(os.path.join(mimicdir, 'adult_notes.gz'), compression='gzip')

    stop = set(stopwords.words('english'))
    token_list = []
    for i in range(0, data.shape[0]):
        text = data.at[i,'chartext']
        try:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
        except:
            tokens = ['']
        clean_tokens = [k for k in tokens if k not in stop]
        #print(clean_tokens)
        token_list.append(clean_tokens)
        data.at[i,'chartext'] = (' ').join(clean_tokens)

    id2word = corpora.Dictionary(token_list)
    corpus = [id2word.doc2bow(note) for note in token_list]

    for num_topics in [20,50,100]:
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=20,
                                            alpha='auto',
                                            per_word_topics=True)
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        coherence_model_lda = CoherenceModel(model=lda_model, texts=token_list, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        


if __name__ == '__main__':
    model_topics_from_notes()