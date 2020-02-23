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
            corpus_words (np.array of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    corpus_words = list(set(corpus))  #  get only distinct values
    corpus_words = sorted(corpus_words)  #  sort all words
    num_corpus_words = len(corpus_words)

    return np.array(corpus_words), num_corpus_words


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
        self._word2token = {}
        self._token2word, dist_words = distinct_words(self._corpus)
        assert dist_words == vocabulary_size, 'dist_num = {}'.format(dist_num)
        for ind, word in enumerate(self._token2word):
            self._word2token[word] = ind
        
        self._corpus_tokens = []
        for word in self._corpus:
            self._corpus_tokens.append(self._word2token[word])
        self._corpus_tokens = np.array(self._corpus_tokens)
        self._neighbors_indent = np.zeros((batch_size, 2*window_size), dtype='int64')
        self._neighbors_indent += np.hstack([np.arange(-window_size, 0), np.arange(0, window_size)+1])

    def __len__(self):
        return int((self._n - 2*self._window_size) / self._batch_size)

    def tokens_to_words(self, tokens):
        """ Get corresponding words to tokens
            Params:
                tokens (np.array): array with tokens
            Return:
                words (list (maybe of lists)): array with words
        """ 
        return self._token2word[tokens].tolist()
        
    def words_to_tokens(self, words):
        """ Get corresponding words to tokens
            Params:
                words (list): list with words which need to translate
            Return:
                tokens (np.array): array with tokens
        """
        tokens = []
        for word in words:
            assert word in self._word2token.keys()
            tokens.append(self._word2token[word])

        return np.array(tokens)

    def _get_next_batch(self):
        """ Return next batch with order specified in self._batchs_positions
            Return:
                centrals (np.array()): batch of central words tokens with shape (batch_size, )
                neighbours (np.array()): batch with tokens of neighbour words. The size is (batch_size, 2*window_size)
        """
        if self._batch_start + self._batch_size > len(self._batchs_positions):
            return None, None
        
        central_positions = self._batchs_positions[np.arange(self._batch_start, self._batch_start + self._batch_size)]  # size(batch_size, 1)
        neighbours_positions = self._neighbors_indent + central_positions.reshape(-1, 1)  # size(batch_size, 2*window_size)
        assert central_positions.shape == (self._batch_size, ), str(central_positions.shape)
        assert neighbours_positions.shape == (self._batch_size, 2*self._window_size), str(neighbours_positions.shape)

        centrals = self._corpus_tokens[central_positions]  # size(batch_size, )
        neighbours = self._corpus_tokens[neighbours_positions]  # size(batch_size, 2*window_size)
        assert centrals.shape == (self._batch_size, )
        assert neighbours.shape == (self._batch_size, 2*self._window_size)

        self._batch_start += self._batch_size
        return centrals, neighbours

    def __iter__(self):
        self._batchs_positions = np.random.permutation(np.arange(self._window_size, self._n - self._window_size))
        self._batch_start = 0
        return self
