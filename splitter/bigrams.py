# author: Mark Lahvinovich <vedmark2012@gmail.com>

from math import log10

from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader

"""
A splitter for texts with no spaces using algorithm
described by Peter Norvig in his book "Beautiful Data".
"""


def memo(f):
    """
    :param f: function that accepts arguments to compute result may
        return result from table if function result is estimated earlier.
    """
    table = {}

    def f_memo(*args):
        if len(table) > 1000000:
            table.clear()
        if args not in table:
            table[args] = f(*args)
        return table[args]

    f_memo.memo = table
    return f_memo


class FrequenceVocabulary:
    """
    Vocabulary that contains words frequency estimated from
    words count in files specified.
    """

    def __init__(self, miss_f):
        """
        Construct new vocabulary with function that computes word probability
        for words which absent in vocabulary. Example usage:

            >>> miss_f = lambda key, N: 10. / (N * 10 ** len(key))

        :param miss_f: function for estimating probability of missing words.
        """
        self.vocab = FreqDist()
        self._miss_f = miss_f

    def load_vocab(self, root='.', files='.*'):
        """
        Load new vocabulary.

        :param root: the root directory for the corpus.
        :param files: A list or regexp specifying the files in this corpus.
        """
        voc = PlaintextCorpusReader(root, files)
        for word in voc.words():
            self.vocab[word.lower()] += 1

    def p(self, key):
        """
        :param key: word to compute it's probability
        :return: A probability distribution computed for key.
        """
        return 1. * self.vocab[key] / self.vocab.N() if key in self.vocab.keys() else self._miss_f(key, self.vocab.N())


class FrequenceBigramsVocabulary(FrequenceVocabulary):
    """
    Vocabulary for words collocations.
    """

    def __init__(self, miss_f):
        """
        Construct new bigram vocabulary with function that computes word
        probability for words which absent in vocabulary. Example usage:

            >>> miss_f = lambda key, N: 10. / (N * 10 ** len(key))

        :param miss_f: function for estimating probability of missing words.
        """
        FrequenceVocabulary.__init__(self, miss_f)

    def load_vocab(self, root='.', files='.*'):
        """
        Load new vocabulary.

        :param root: the root directory for the corpus.
        :param files: A list or regexp specifying the files in this corpus.
        """
        voc = PlaintextCorpusReader(root, files)
        for line in voc.sents():
            for word in self._get_collocations(line):
                self.vocab[word] += 1

    @staticmethod
    def _get_collocations(line):
        for i in range(1, len(line)):
            yield line[i - 1] + ' ' + line[i]


class BigramSplitter:
    """
    Splitter for texts from plaintext documents.
    """

    def __init__(self, max_len=20):
        """
        Constructs new BigramSplitter for words with maximum length
        specified by max_len
        :param max_len: maximum possible length of words in training samples.
        """
        self.vocab = FrequenceVocabulary(lambda key, N: 10. / (N * 10 ** len(key)))
        self.collocations = FrequenceBigramsVocabulary(lambda key, N: 1. / N)
        self.max_len = max_len

    def load_corpus(self, root, files):
        """
        Load corpus for detecting word patterns.
        :param root: the root directory for the corpus.
        :param files: A list or regexp specifying the files in this corpus.
        """
        self.vocab.load_vocab(root, files)
        self.collocations.load_vocab(root, files)

    @memo
    def split(self, text):
        """
        :param text: string to tokenize into separate words.
        :return: list of separated words.
        """
        return self._run_split(text)[1]

    @memo
    def _run_split(self, text, prev='<S>'):
        if not text:
            return .0, []
        return max([self._combine(log10(self._prob(first.lower(), prev.lower())), first, *self._run_split(rem, first))
                    for first, rem in self._splits(text)])

    def _splits(self, text):
        return [(text[:i+1], text[i+1:]) for i in range(min(len(text), self.max_len))]

    def _prob(self, word, prev_word):
        phrase = prev_word + ' ' + word
        if self.vocab.vocab[prev_word] and self.collocations.vocab[phrase]:
            return self.collocations.vocab[phrase] / float(self.vocab.vocab[prev_word])
        else:
            return self.vocab.p(word)

    @staticmethod
    def _combine(p_first, first, p_rem=0, rem=list()):
        return p_first + p_rem, [first] + rem
