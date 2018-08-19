# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import HashingVectorizer
import re
#import os
import pickle
#cur_dir = os.path.dirname("__file__")



pickle_in = open("pkl_objects/stopwords.pickle","rb")

stop = pickle.load(pickle_in)


#stop = pickle.load(open('stopwords.pkl', 'rb'))

def tokenizer(text):
       text = re.sub('<[^>]*>', '', text)
       emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                              text.lower())
       text = re.sub('[\W]+', ' ', text.lower()) \
                      + ' '.join(emoticons).replace('-', '')
       tokenized = [w for w in text.split() if w not in stop]
       return tokenized
vect = HashingVectorizer(decode_error='ignore',
                            n_features=2**21,
                            preprocessor=None,
                            tokenizer=tokenizer)

'''

'''