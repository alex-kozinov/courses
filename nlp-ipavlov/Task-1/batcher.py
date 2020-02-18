import numpy as np
from numpy.random import permutation
from collections import Counter

UNK_TOKEN = '<UNK>'


def read_corpus(filepath):
    """ Read text from the specified text file. Than clean it and form the corpus
        Params:
            filepath (string): file with text
        Return:
            corpus_words (list of strings): list of clear words processed file
    """
    corpus = None
    with open(filepath, 'r') as f:
        text = f.read()
        corpus = [w.strip(' ,.!?').lower() for w in text.split()]
    return corpus


def build_vocabulary(corpus, vocabulary_size):
    """ Leave only vocabulary_size words and replace not common with UNK_TOKEN (this token also counted)
        Params:
            corpus (list of strings): corpus of document
            vocabulary_size (int): the size of vocabulary
        Return:
            corpus_words (list of strings): list of words with length equals to corpus length
    """
    
    c = Counter(corpus)
    if len(c) == vocabulary_size:
        most_common_words = c.most_common(vocabulary_size)
    else:
        most_common_words = c.most_common(vocabulary_size - 1)  #  Have to use UNK token
        
    vocabulary = set()
    for word, cnt in most_common_words:
        vocabulary.add(word)
        
    corpus = list(map(lambda w: w if w in vocabulary else UNK_TOKEN, corpus))
    return corpus

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of strings): corpus of document
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    corpus_words = list(set(corpus))  #  get only distinct values
    corpus_words = sorted(corpus_words)  #  sort all words
    num_corpus_words = len(corpus_words)

    return corpus_words, num_corpus_words


class SkipGramBatcherBase(object):
    def __init__(self, corpus, window_size=5, batch_size=2, vocabulary_size=None):
        """ initialize skip-gram batcher
            Params:
                corpus (list of strings): corpus of document
                window_size (int): range of window around central word
                batch_size (int): size of each butch.
                vocabulary_size (int?): number of distinct words. Corpus has to have at least vocabulary_size words
        """
        self._n = len(corpus)
        self._window_size = window_size
        self._batch_size = batch_size
        
        dist_words  = len(Counter(corpus))
        if vocabulary_size is None:
            vocabulary_size = dist_words
        assert vocabulary_size <= dist_words
        self._voc_size = vocabulary_size
        
        self._corpus = build_vocabulary(corpus, vocabulary_size)
        self._word2index = {}
        self._index2word, dist_words = distinct_words(self._corpus)
        assert dist_words == vocabulary_size, 'dist_num = {}'.format(dist_num)
        for ind, word in enumerate(self._index2word):
            self._word2index[word] = ind
    
    def __len__(self):
        return self._n - 2*self._window_size

    def index_to_word(self, idx):
        assert idx < self._voc_size
        return self._index2word[idx]
    
    def word_to_index(self, word):
        assert word in self._word2index.keys()
        return self._word2index[word]
    
    def index_to_onehot(self, n):
        assert n < self._voc_size
        onehot = np.zeros(self._voc_size)
        onehot[n] = 1
        return onehot
    
    def onehot_to_index(self, v):
        assert len(v) == self._voc_size
        return np.argmax(v)
    
    def word_to_onehot(self, word):
        return self.index_to_onehot(self.word_to_index(word))
    
    def onehot_to_word(self, v):
        return self.index_to_word(self.onehot_to_index(word))
    
    def _get_word_indices(self, pos):
        """ Get indices of neighbour words and central word
            Params:
                pos (int): word by its position in corpus

            Return:
                central (int): number of central word
                neighbours (np.array): numbers of words in window [ind - window_size, ind - 1] and [ind + 1,  ind + window_size]
        """
        assert pos >= self._window_size
        assert pos + self._window_size < self._n
        
        central = self.word_to_index(self._corpus[pos])
        neighbours = []
        
        window_lift_bound = pos - self._window_size
        window_right_bound = pos + self._window_size
        for i in range(window_lift_bound, window_right_bound + 1):
            if i == pos:
                continue
            neighbours.append(self.word_to_index(self._corpus[i]))

        return central, np.array(neighbours)

    def batch_to_words(self, batch, batch_type='indices'):
        """ Translate batch to human read view.
            Params:
               batch (np.array): some dimential array
               batch_type (string): type of values in batch, possible value is 'indices', 'onehot'
            Return:
               words_batch (list): list with length as length of batch
        """
        assert batch_type in ['indices', 'onehot'], 'You can only use indices and onehot types'
        transform_function = self.index_to_word if batch_type == 'indices' else self.onehot_to_word
        
        words_batch = []
        for el in batch:
            if not len(el.shape):
                words_batch.append(transform_function(el))
            else:
                words_batch.append(self.batch_to_words(el, batch_type))
        return words_batch

    def __iter__(self):
        self._batchs_positions = list(permutation(range(self._window_size, self._n - self._window_size)))
        return self
