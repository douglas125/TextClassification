import os
import zipfile
import numpy as np
from tqdm import tqdm

import requests
import math

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
    def downloadNILCEmbeddings():
        """
        Downloads some of the NILC embeddings to the 'embedding' folder
        """
        print('Downloading NILC word embeddings. More available at http://nilc.icmc.usp.br/embeddings')        
        NILCfiles = {
                        'glove50' : 'http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s50.zip',
                        'cbow50_wang2vec' : 'http://143.107.183.175:22980/download.php?file=embeddings/wang2vec/cbow_s50.zip',
                        'cbow50_fasttext' : 'http://143.107.183.175:22980/download.php?file=embeddings/fasttext/cbow_s50.zip',
                        'skip50_word2vec' : 'http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s50.zip'
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