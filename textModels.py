import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

##############
#Dictionaries
##############

#for all dicts, add 1 to result and reserve 0 to not found
from sklearn.feature_extraction.text import CountVectorizer
import string

allchars = string.printable
allchars = [x for x in allchars] + ['PAD']
allchars = { allchars[i]:i for i in range(len(allchars)) if allchars[i] not in [' ','\n']}

def extractVocabulary(textSet, maxWords = 3000):
    #extracts vocabulary from a list of texts
    #preprocessing to remove accents and uppercase should be done before
    
    countVec = CountVectorizer(max_features=maxWords, lowercase=False, strip_accents=None)
    countVec.fit(textSet)
    
    #append punctuation
    vocab = countVec.vocabulary_
    n = len(vocab)
    for x in string.punctuation:
        vocab[x]=n
        n += 1
    
    return vocab

def sentence2code(sentence, vocabulary, embClass = None):
    """
    Converts a sentence to char embedding codes, word embedding codes and embeddings
    
    vocabulary - dictionary that maps words to integers
    sentence - list of words in the sentence, usually from preProcessing.clean_text().split(' ')
    embClass - a class that implements method encodeWord and has property embDim (embedding dimension)
    """
    
    assert type(sentence) == list, 'sentence should be a list of words'
    
    #sentence
    sentCode = [vocabulary.get(w,-1)+1 for w in sentence]
    
    #characters
    sent_len = len(sentence)
    maxwlen = np.max([len(x) for x in sentence])
    charCodes = np.zeros( (sent_len, maxwlen) ) + allchars['PAD']
    
    for i in range(sent_len):
        charEnc = [allchars.get(cc, -1)+1 for cc in sentence[i]]
        charCodes[i, 0:len(charEnc)] = charEnc
    
    wordEmbeddings = None
    if embClass is not None:
        wordEmbeddings = np.zeros ((sent_len, embClass.embDim))
        for i in range(sent_len):
            wordEmbeddings[i] = embClass.encodeWord(sentence[i])
        
    return np.array(sentCode), charCodes.astype(int), wordEmbeddings


##############
#Keras models
##############
from sklearn import preprocessing

#change LSTM to CuDNNLSTM
from keras.layers import CuDNNLSTM, LSTM
from keras.layers import Reshape, Dot, Softmax, Flatten, BatchNormalization, Dropout
from keras.layers import Input, Embedding, Conv2D, Lambda, Concatenate, Bidirectional, TimeDistributed, Dense
from keras.models import Model, load_model
from keras import backend as K

def createCharEncoder(charDictSize, embSize, nFiltersNGram=16, filterSize = 5):
    """
    Creates a character encoder. Receives the integer code of the character.
    
    charDictSize - Length of dictionary of characters
    embSize - Embedding size
    """
    inp = Input((None, ))
    
    embedded = Embedding(charDictSize, embSize)(inp)
        
    embedded = Lambda(lambda x: K.expand_dims(x))(embedded)
    ngram = Conv2D(nFiltersNGram, kernel_size = (5,1), padding='same', activation='relu')(embedded)
    ngram = Conv2D(1, kernel_size = (filterSize,1), padding='same', activation=None)(ngram)
    
    ngram = Lambda(lambda x: K.squeeze(x, axis=3))(ngram)
    ngram = Bidirectional(LSTM(embSize//2))(ngram)
    
    output = ngram
    
    model = Model(inputs=[inp], outputs=[output], name='CharEncoder')
    return model

def createDocEncoder(dictSize, embSize, nFiltersWordGram = 10, filterSize = 5, embDim = None):
    """
    Creates a document encoder. Receives the integer code of the words.
    
    dictSize - Length of word dictionary
    embSize - Embedding size
    """
    inp = Input((None, ))
    
    embedded = Embedding(dictSize, embSize)(inp)
    
    #combine learned and pretrained embeddings
    if embDim is not None:
        preTrainedEmb = Input((None, embDim))
        embedded = Concatenate()([embedded, preTrainedEmb])

    
    embedded = Lambda(lambda x: K.expand_dims(x))(embedded)
    ngram = Conv2D(nFiltersWordGram, kernel_size = (filterSize,1), padding='same', activation='relu')(embedded)
    ngram = Conv2D(1, kernel_size = (filterSize,1), padding='same', activation=None)(ngram)
    
    ngram = Lambda(lambda x: K.squeeze(x, axis=3))(ngram)
    
    output = ngram
    
    if embDim is None:
        model = Model(inputs=[inp], outputs=[output], name='WordEncoder')
    else:
        model = Model(inputs=[inp, preTrainedEmb], outputs=[output], name='WordEncoderWithPreEmb')
    return model  


def createAttLayer( val_dim = 60, key_dim = 41, query_dim = 30, nHeads = 3, projActivation=None ):
    """
    Multi head attention layer - returns an attention layer.
    Remember that Keras hides the batch_size dimension. It is mentioned for completeness
    
    inputs:
    
    values (batch_size, sequence_length, val_dim) - over which avg sum is computed. Can be equal to keys
    keys   (batch_size, sequence_length, key_dim) - used in search
    query  (batch_size, query_dim)
    nHeads - number of attention heads to use (i.e., get information from how many distinct points in input?)
    projActivation - Activation of the projection layer
    
    outputs:
    wAvg       (batch_size, nHeads, val_dim) - a total of nHeads weighted averages of values along sequence_length axis.
    attWeights (batch_size, nHeads, val_dim) - a total of nHeads weights used to average values.
    
    
    """
    vals = Input( (None, val_dim) )
    keys = Input( (None, key_dim) )
    query = Input( (query_dim,) )
    
    #project query
    q = Dense(nHeads*key_dim, activation = projActivation)(query)
    q = Reshape( (nHeads,key_dim) )(q)
    
    #compute attention
    attScores = Dot([2,2])([q,keys])
    attScores = BatchNormalization(axis=1)(attScores)
    attWeights = Softmax(name='attW')(attScores)
    
    wAvg = Dot([2,1])([attWeights, vals])
    model = Model(inputs=[vals, keys, query], outputs=[wAvg, attWeights], name='AttLayer_{}h'.format(nHeads))
    
    return model

def createBiDirAttModel(charDictSize, dictSize, nHeads = 3,
                        charEmbSize=16, nFiltersNGram=16, charfilterSize = 5, #character params
                        wordEmbSize=128, nFiltersWordGram = 10, wordfilterSize = 5, preTrainedEmbDim = None,
                        modelType = 'classifier', nClasses = 3): #word params
    
    inputChars = Input((None, None))
    cFeatLayer = createCharEncoder(charDictSize, charEmbSize, nFiltersNGram, charfilterSize)
    charFeats = TimeDistributed(cFeatLayer)(inputChars)
    
    inputWords = Input((None, ))
    if preTrainedEmbDim is not None:
        preTrainedEmb = Input((None, preTrainedEmbDim))
        wordEncoded = createDocEncoder(dictSize, wordEmbSize, nFiltersWordGram, 
                                       wordfilterSize, preTrainedEmbDim)([inputWords, preTrainedEmb])
    else:
        wordEncoded = createDocEncoder(dictSize, wordEmbSize, nFiltersWordGram, 
                                       wordfilterSize, preTrainedEmbDim)(inputWords)
        
    combinedEncodings = Concatenate()([wordEncoded, charFeats])
    enc_memory = Bidirectional(LSTM(wordEmbSize//2, return_sequences=True))(combinedEncodings)

    if modelType == 'classifier':
        q = LSTM(wordEmbSize, return_sequences=False)(enc_memory)
        
        attLayer = createAttLayer( val_dim = wordEmbSize, key_dim = wordEmbSize, query_dim = wordEmbSize, 
                                   nHeads = nHeads, projActivation=None )
        
        output, attWeights = attLayer([enc_memory, enc_memory, q])
        #output, attWeights = attLayer([combinedEncodings, enc_memory, q])
        
        output=Flatten()(output)
        
        output = Dense(wordEmbSize//2, activation = 'relu')(output)
        #output = Dropout(0.1)(output)
        output = Dense(wordEmbSize//4, activation = 'relu')(output)
        output = Dense(nClasses, activation = 'softmax', name='output')(output)

    
    if preTrainedEmbDim is None:
        model = Model(inputs=[inputWords, inputChars], outputs=[output, attWeights], name='BiAttEnc')
    else:
        model = Model(inputs=[inputWords, inputChars, preTrainedEmb], outputs=[output, attWeights], name='BiAttEncWithPretrainedEmb')
        
        
        
    return model

##############################
#Scikit learn compatible model
##############################
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
def step_decay(epoch):
    initial_lrate = 0.001                
    drop = 0.6
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    
    if (lrate < 5e-6):
        lrate = 5e-6
      
    print('Changing learning rate to {}'.format(lrate))
    return lrate

class BiDirAttModelClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, nClasses, charDictSize=len(allchars)+2, dictSize = 10000, preTrainedEmbeddings = None, nHeads = 3,
                 modelFileName='model-text.h5', patience = 25, epochs = 100, val_split = 0.15,
                 charEmbSize=16, nFiltersNGram=8, charfilterSize = 4, #character params
                 wordEmbSize=128, nFiltersWordGram = 8, wordfilterSize = 4): #word params
        """
        Initializes classifier
        
        charDictSize - size of character dictionary
        dictSize - word dictionary size (for trainable embeddings)
        
        val_split - validation split (20% for validation default)
        """
        self.nClasses = nClasses
        
        self.modelFileName = modelFileName
        self.patience = patience
        self.epochs = epochs
        self.val_split = val_split
        
        self.charDictSize = charDictSize
        self.dictSize = dictSize
        self.charEmbSize=charEmbSize
        self.nFiltersNGram=nFiltersNGram
        self.charfilterSize=charfilterSize
        self.wordEmbSize=wordEmbSize
        self.nFiltersWordGram = nFiltersWordGram 
        self.wordfilterSize = wordfilterSize
        
        self.nHeads = nHeads
        
        self.preTrainedEmbDim = None
        self.preTrainedEmbeddings = None
        if preTrainedEmbeddings is not None:
            self.preTrainedEmbDim = preTrainedEmbeddings.embDim
            self.preTrainedEmbeddings = preTrainedEmbeddings
        
        self.initialized = False
    
    def _initModel(self):
        #if self.initialized:
        #    return
        
        self.model_ = createBiDirAttModel(self.charDictSize, self.dictSize, nHeads = self.nHeads,
                                          charEmbSize=self.charEmbSize, nFiltersNGram=self.nFiltersNGram, 
                                          charfilterSize = self.charfilterSize, #character params
                                          wordEmbSize=self.wordEmbSize, nFiltersWordGram = self.nFiltersWordGram, 
                                          wordfilterSize = self.wordfilterSize, preTrainedEmbDim = self.preTrainedEmbDim,
                                          modelType = 'classifier', nClasses = self.nClasses)
        
        inputChars = Input((None, None))
        inputWords = Input((None, ))
        
        if self.preTrainedEmbeddings is not None:
            preTrainedEmb = Input((None, self.preTrainedEmbDim))
            output = self.model_([inputWords, inputChars, preTrainedEmb])
            output = output[0]
            self.trainModel_ = Model(inputs=[inputWords, inputChars, preTrainedEmb], outputs=output, name='BiAttEnc_train')
        else:
            output = self.model_([inputWords, inputChars])
            output = output[0]
            self.trainModel_ = Model(inputs=[inputWords, inputChars], outputs=output, name='BiAttEnc_train')
            
        self.trainModel_.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
        
        #note that for sparse the target y has to have shape (batch_size, 1) <-the 1 matters
        #self.model_.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
        
        #self.initialized = True
    
    def preprocess(self, X, y):
        X_wCodes = []
        X_cCodes = []
        X_wEmbs = []
        for s in X:
            wCodes, cCodes, wEmbs = sentence2code(s.split(' '), self.vocab_, self.preTrainedEmbeddings)
            X_wCodes.append(wCodes)
            X_cCodes.append(cCodes)
            X_wEmbs.append(wEmbs)
        
        #self.temp = X_wEmbs
        
        max_sentLen = max([cc.shape[0] for cc in X_cCodes])
        max_wordLen = max([cc.shape[1] for cc in X_cCodes])
        print('Maximum sentence length: {}. Maximum number of chars in a word: {}'.format(max_sentLen, max_wordLen))
        
        X_wCodes_transf = np.zeros( (len(X), max_sentLen), dtype=int )
        X_cCodes_transf = np.zeros( (len(X), max_sentLen, max_wordLen), dtype=int )
        if self.preTrainedEmbeddings is not None:
            X_wEmbs_transf = np.zeros( (len(X), max_sentLen, self.preTrainedEmbeddings.embDim) )
        else:
            X_wEmbs_transf = None        
            
        for i in range(len(X)):
            s_len = X_wCodes[i].shape[0]
            w_len = X_cCodes[i].shape[1]
            X_wCodes_transf[i, 0:s_len] =  X_wCodes[i]
            X_cCodes_transf[i, 0:s_len, 0:w_len] = X_cCodes[i]
            if self.preTrainedEmbeddings is not None:
                X_wEmbs_transf[i, 0:X_wEmbs[i].shape[0], 0:X_wEmbs[i].shape[1]] = X_wEmbs[i]
        
        return X_wCodes_transf,X_cCodes_transf,X_wEmbs_transf, np.expand_dims(self.lblEncoder_.transform(y).astype(int),1)
    
    def fit(self, X, y):
        assert len(X) == len(y), 'X and y must have the same length'

        self.vocab_ = extractVocabulary(X, maxWords=self.dictSize)
        self.dictSize = len(self.vocab_)+1
        self.lblEncoder_ = preprocessing.LabelEncoder()
        self.y_transf_train = self.lblEncoder_.fit(y)
        
        self._initModel()
        
        self.X_wCodes_train,self.X_cCodes_train,self.X_wEmbs_train, self.y_transf_train = self.preprocess(X,y)
            
        checkpointer = ModelCheckpoint(self.modelFileName, verbose=1, save_best_only=True, monitor='val_sparse_categorical_accuracy')    
        lrate = LearningRateScheduler(step_decay)
        earlystopper = EarlyStopping(patience=self.patience, #restore_best_weights=True,  
                                     verbose=1, monitor='val_sparse_categorical_accuracy')
        
        if self.preTrainedEmbeddings is not None:
            self.results_ = self.trainModel_.fit(x=[self.X_wCodes_train, self.X_cCodes_train, self.X_wEmbs_train], 
                                            y=self.y_transf_train, epochs=self.epochs,
                                            verbose=1, validation_split=self.val_split, 
                                            batch_size=128, callbacks=[checkpointer, earlystopper, lrate])
        else:
            self.results_ = self.trainModel_.fit(x=[self.X_wCodes_train, self.X_cCodes_train], 
                                            y=self.y_transf_train, epochs=self.epochs,
                                            verbose=1, validation_split=self.val_split, 
                                            batch_size=128, callbacks=[checkpointer, earlystopper, lrate])
        
        #load back best weights    
        self.trainModel_.load_weights(self.modelFileName)    
        
        # Return the classifier
        return self
    
    def predictWithAttention(self, X):
        X_wCodes_p,X_cCodes_p,X_wEmbs_p, y_transf_train = self.preprocess(X,[])
        
        if self.preTrainedEmbeddings is not None:
            ans = self.model_.predict([X_wCodes_p,X_cCodes_p,X_wEmbs_p])
        else:
            ans = self.model_.predict([X_wCodes_p,X_cCodes_p])
            
        y = ans[0]
        attw = ans[1]
        y = np.argmax(y, axis=1)
        y = self.lblEncoder_.inverse_transform(y)
        
        return y, attw
    
    def predict(self, X):
        X_wCodes_p,X_cCodes_p,X_wEmbs_p, y_transf_train = self.preprocess(X,[])
        
        if self.preTrainedEmbeddings is not None:
            y = self.trainModel_.predict([X_wCodes_p,X_cCodes_p,X_wEmbs_p])
        else:
            y = self.trainModel_.predict([X_wCodes_p,X_cCodes_p])
        
        y = np.argmax(y, axis=1)
        y = self.lblEncoder_.inverse_transform(y)
        # Check is fit had been called
        #check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)

        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return y #self.y_[closest]