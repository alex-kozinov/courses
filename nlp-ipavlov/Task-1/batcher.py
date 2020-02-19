import numpy as np
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
        
        self._corpus_indices = []
        for word in self._corpus:
            self._corpus_indices.append(self._word2index[word])
        self._corpus_indices = np.array(self._corpus_indices)
        self._neighbors_indent = np.zeros((batch_size, 2*window_size), dtype='int64')
        self._neighbors_indent += np.hstack([np.arange(-window_size, 0), np.arange(0, window_size)+1])

    def __len__(self):
        return self._n - 2*self._window_size

    def index_to_word(self, idx):
        assert idx < self._voc_size
        return self._index2word[idx]
    
    def word_to_index(self, word):
        assert word in self._word2index.keys()
        return self._word2index[word]
    
    def indices_to_onehot(self, idxs):
        idxs = idxs.flatten()
        n = len(idxs)
        one_hots = np.zeros((n, self._voc_size))
        one_hots[np.arange(n), idxs] = 1
        return one_hots
    
    def onehot_to_index(self, v):
        assert len(v) == self._voc_size
        return np.argmax(v)
    
    def word_to_onehot(self, word):
        return self.index_to_onehot(self.word_to_index(word))
    
    def onehot_to_word(self, v):
        return self.index_to_word(self.onehot_to_index(word))
    
    def _get_next_batch(self):
        """ Return next batch with order specified in self._batchs_positions
            Return:
                centrals (np.array()): batch of central words indices with shape (1, batch_size)
                neighbours (np.array()): batch with indices of neighbour words. The size is (batch_size, 2*window_size)
        """
        if self._batch_start + self._batch_size > len(self._batchs_positions):
            return None, None
        
        central_positions = self._batchs_positions[np.arange(self._batch_start, self._batch_start + self._batch_size)]  # size(batch_size, 1)
        neighbours_positions = self._neighbors_indent + central_positions.reshape(-1, 1)  # size(batch_size, 2*window_size)
        assert central_positions.shape == (self._batch_size, ), str(central_positions.shape)
        assert neighbours_positions.shape == (self._batch_size, 2*self._window_size), str(neighbours_positions.shape)

        centrals = self._corpus_indices[central_positions]  # size(batch_size, )
        neighbours = self._corpus_indices[neighbours_positions]  # size(batch_size, 2*window_size)
        assert centrals.shape == (self._batch_size, )
        assert neighbours.shape == (self._batch_size, 2*self._window_size)

        self._batch_start += self._batch_size
        return centrals, neighbours

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
        self._batchs_positions = np.random.permutation(np.arange(self._window_size, self._n - self._window_size))
        self._batch_start = 0
        return self
