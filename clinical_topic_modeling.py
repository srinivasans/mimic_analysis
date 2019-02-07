'''
Script for topic modeling from adult ICU clinical notes using LDA (Part 4)
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
plt.style.use('ggplot')

mimicdir = os.path.expanduser("~/Coursework/ML4Health/Assignment")

'''
Topic modeling of Adult ICU clinical notes using Latent Dirchlet Allocation
'''
def model_topics_from_notes():
    data = pd.read_csv(os.path.join(mimicdir, 'adult_notes.gz'), compression='gzip')

    # Pre-processing : Removing stop words and punctuations from the text
    stop = set(stopwords.words('english'))
    token_list = []
    for i in range(0, data.shape[0]):
        text = str(data.at[i,'chartext'])
        text = text.lower()
        try:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
        except:
            tokens = ['']
        clean_tokens = [k for k in tokens if k not in stop]
        token_list.append(clean_tokens)
        data.at[i,'chartext'] = (' ').join(clean_tokens)

    # Creating a dictionary of words from the token list
    id2word = corpora.Dictionary(token_list)
    # Creating the corpus with bag of words from clinical notes
    corpus = [id2word.doc2bow(note) for note in token_list]

    # Run LDA for Number of Topics = 20, 50 and 100
    coherence_scores = []
    num_topics_list = [20,50,100]
    for num_topics in num_topics_list:
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            passes=1,
                                            alpha='auto',
                                            per_word_topics=True)
        
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_scores.append(coherence_lda)
        print('Coherence Score (Number of Topics : %s): '%num_topics, coherence_lda)
    
    # Obtain the number of topics with best coherence score
    besttopicidx = np.argmax(coherence_scores)
    best_num_topics = num_topics_list[besttopicidx]
    best_coherence_score = coherence_scores[besttopicidx]
    print("Number of topics with best coherence score : %d, : %f coherence"%(best_num_topics, best_coherence_score))

    # Plotting the number of topics vs coherence scores
    plt.figure()
    plt.plot(num_topics_list, coherence_scores)
    plt.xlabel('Number of Topics')
    plt.ylabel('UMass Coherence Scores')
    plt.title('Num Topics vs Coherence Score')
    plt.savefig('part4a_lda.png')

        # Build LDA model with num topics having best coherence score
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=best_num_topics, 
                                            random_state=100,
                                            passes=1,
                                            alpha='auto',
                                            per_word_topics=True)

    related_topics = {}
    # Look for related words in the topic of the search keywords using the best coherence model
    search_keywords = ['respiratory','vomiting', 'urine', 'pulse']
    for keyword in search_keywords:
        topic_scores = lda_model.get_term_topics(id2word.doc2bow([keyword])[0][0], minimum_probability=0)
        max_score = 0
        max_topic = 0
        for score in topic_scores:
            if score[1] > max_score:
                max_score = score[1]
                max_topic = score[0]
        print(max_topic)
        topic_ids = lda_model.get_topic_terms(max_topic, topn=50)
        topics = [id2word[topic_id[0]] for topic_id in topic_ids]
        related_topics[keyword] = topics
        print("Related words in Topic for \"%s\" : %s"%(keyword, topics))

if __name__ == '__main__':
    # Part 4(a), 4(b)
    model_topics_from_notes()