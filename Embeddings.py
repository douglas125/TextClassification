import os
import zipfile
import numpy as np
from tqdm import tqdm

import requests
import math


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import scipy.stats

"""
NILC word embeddings

Reference:

@article{DBLP:journals/corr/abs-1708-06025,
  author    = {Nathan Hartmann and
               Erick R. Fonseca and
               Christopher Shulby and
               Marcos Vin{\'{\i}}cius Treviso and
               Jessica Rodrigues and
               Sandra M. Alu{\'{\i}}sio},
  title     = {Portuguese Word Embeddings: Evaluating on Word Analogies and Natural
               Language Tasks},
  journal   = {CoRR},
  volume    = {abs/1708.06025},
  year      = {2017},
  url       = {http://arxiv.org/abs/1708.06025},
  archivePrefix = {arXiv},
  eprint    = {1708.06025},
  timestamp = {Mon, 13 Aug 2018 16:49:05 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1708-06025},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

import re
def splitWithPunctuation(text):
    return re.findall(r"[\w']+|[.,!?:;\"]", text)

def _downloadFile(url, filename):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0)); 
    block_size = 1024
    wrote = 0 
    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")  


class WordEmbeddingBR:
    def downloadNILCEmbeddings(mode='50'):
        """
        Downloads some of the NILC embeddings to the 'embedding' folder
        """
        avModes = ['50', '100', '300', '600', '1000']
        assert mode in avModes, 'Mode has to be one of the available embedding sizes: {}'.format(avModes)
        
        baseURL = 'http://143.107.183.175:22980/download.php?file=embeddings'
        print('Downloading NILC word embeddings. More available at http://nilc.icmc.usp.br/embeddings')        
        NILCfiles = {
                        'glove{}'.format(mode) : baseURL + '/glove/glove_s{}.zip'.format(mode),
                        'cbow{}_wang2vec'.format(mode) :  baseURL +'/wang2vec/cbow_s{}.zip'.format(mode),
                        'cbow{}_fasttext'.format(mode) :  baseURL +'/fasttext/cbow_s{}.zip'.format(mode),
                        'skip{}_word2vec'.format(mode) :  baseURL +'/word2vec/skip_s{}.zip'.format(mode)
                    }
        
        if not os.path.exists('embedding/'):
            os.makedirs('embedding/')
            
        for key in NILCfiles:
            destFile = 'embedding/'+key+'.zip'
            if not os.path.exists(destFile):
                print('Downloading {} from {}'.format(key, NILCfiles[key]) )
                _downloadFile(NILCfiles[key], destFile)
            else:
                print('{} exists. Skipping.'.format(key))
        print('Done!')
            
    def getAvailableEmbeddings():
        """
        Retrieves available embeddings from 'embedding' folder
        """
        
        embs = next(os.walk('embedding/'))[2]
        embs = [x.replace('.zip','') for x in embs]
        return embs
    
    def __init__(self, sourceFile = 'cbow50_wang2vec'):
        """
        Initializes word embedding using desired file
        """
        availableEmbs = WordEmbeddingBR.getAvailableEmbeddings()
        assert sourceFile in availableEmbs, "Embedding {} not available. Options are: {}".format(sourceFile, availableEmbs)

        sourceFile += '.zip'
        
        print('Reading embedding file: {}'.format(sourceFile))
        self.wordEmbDict = {}
        self.embDim = 0
        with zipfile.ZipFile(os.path.join('embedding', sourceFile), "r") as z:
            embFile = z.namelist()[0]
            with z.open(embFile, "r") as f: #, encoding="utf8"
                  for line in tqdm(f):
                        line=line.decode("utf-8")
                        lineSplit = line.split(' ')
                        if len(lineSplit) > 2:
                            vec = [float(lineSplit[k+1]) for k in range(len(lineSplit)-1)]
                            self.wordEmbDict[lineSplit[0]] = np.array(vec)
                            self.embDim = len(vec)
        
        self.sourceFile = sourceFile

    def encodeWord(self, word):
        """
        Returns embedding of a word in the dictionary. Zeroes if none is found
        """
        return self.wordEmbDict.get(word, np.zeros(self.embDim))
    
    def wordFromEmbedding(self, embeddingVector, topN = 10):
        """
        Returns word that matches 'embeddingVector' best, using cosine similarity
        
        topN - how many matches to return
        """
        invqnorm = 1.0 / (np.linalg.norm(embeddingVector)+1e-4)
        query = invqnorm * embeddingVector

        def cosdist(q, wEmb):
            invNorm = 1.0 / (np.linalg.norm(wEmb)+1e-4)
            return np.dot(q, wEmb*invNorm )

        distances = [ {'word': x, 'd' : cosdist(query, self.wordEmbDict[x]) } for x in self.wordEmbDict]
        distances = sorted(distances, key=lambda k: 1-k['d']) 
        ans = [ {x['word'] : x['d']} for x in distances ]
        return ans[0:topN]
    
    def getSentenceVector(self, x):
        wordArray = splitWithPunctuation(x)
        ans = np.zeros( (self.maxlen, self.embDim) )
        for i,w in enumerate(wordArray):
            if len(w) > 2:
                ans[i] = self.encodeWord(w.lower())
                ans[i] /= (np.linalg.norm(ans[i])+1e-4)
        return np.sum(ans,axis=0)
        
    def TestBaselineClassifiers(self, X_test, y_test, classifiers):
        ans = {}
        vectTexts_test = [self.getSentenceVector(x).reshape((-1,)) for x in X_test]
        for c in classifiers:
            ans[c] = classifiers[c].best_estimator_.score(vectTexts_test, y_test)
        return ans
    
    def TrainBaselineClassifiers(self, X_train, y_train, n_iter=10):
        """
        Tests a set of baseline classifiers using cross-validation on X_train, y_train. 
        Returns scikit learn CrossValidation objects
        
        X_train - strings containing the texts to be classified
        y_train - desired labels
        """
        ans={}
        
        self.maxlen = max([len(x) for x in X_train])
        vectTexts_train = [self.getSentenceVector(x).reshape((-1,)) for x in X_train]
        
        print('Fitting Support Vector Machine...')
        svmParams = { #'verbose' : [1],
             'gamma': scipy.stats.uniform(0.0015,0.515),#[0.1,0.01,0.02,0.04,0.08],  
             'C' : scipy.stats.uniform(0.01,50),#[0.1,10,15, 20,25,40], 
             'shrinking'    :[True, False]}
        sksvc = SVC(verbose=1, gamma=0.1, tol=1e-5, C=2, kernel='rbf')
        svmRSCV = RandomizedSearchCV(sksvc, svmParams, verbose=1, return_train_score=True, n_iter=2*n_iter) #, n_jobs=-1)
        svmRSCV.fit(vectTexts_train, y_train)
        
        ans['SVM'] = svmRSCV
        
        print('Fitting Gradient Boosted Tree...')        
        gbParams = { #'verbose' : [1],
             'learning_rate': scipy.stats.uniform(0.005,0.85),  
             'n_estimators' : scipy.stats.randint(50, 851), 
             'max_depth'    : scipy.stats.randint(2, 20)}
        gbc = GradientBoostingClassifier(verbose=1, learning_rate=0.1, n_estimators=320, max_depth=6)
        gbRSCV = RandomizedSearchCV(gbc, gbParams, verbose=1, return_train_score=True, n_iter=n_iter) #, n_jobs=-1)
        gbRSCV.fit(vectTexts_train, y_train)
        ans['GradientBoostingClassifier'] = gbRSCV
        
        #self.baselineClassifiers = ans
        
        return ans