import sys
import os
import re
from collections import defaultdict
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(r'\w+')
cached_stop_words = stopwords.words("english")


class MyCorpus(object):
    """a memory-friendly iterator"""
    def __init__(self, dirname):
        self.dirname = dirname
        word_freq = self.count_word_freq()
        self.filter_list = self.get_low_freq_words(word_freq, threshold=5)

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fpath = os.path.join(self.dirname, fname)
            if not os.path.isfile(fpath) or fname[0] == '.':
                continue
            try:
                with open(fpath) as fp:
                    words = word_tokenizer.tokenize(fp.read().lower())
                    words = [word for word in words if word not in cached_stop_words + self.filter_list]
                    yield words
            except Exception as e:
                print e
                sys.exit()
            else:
                fp.close()

    def __len__(self):
        return sum(1 for fname in os.listdir(self.dirname) if os.path.isfile(os.path.join(self.dirname, fname)) and not fname[0] == '.')

    def count_word_freq(self):
        word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
        for fname in os.listdir(self.dirname):
            fpath = os.path.join(self.dirname, fname)
            if not os.path.isfile(fpath) or fname[0] == '.':
                continue
            try:
                with open(fpath) as fp:
                    words = word_tokenizer.tokenize(fp.read().lower())
                    words = [word for word in words if word not in cached_stop_words]
                    # word frequency
                    for each in words:
                        word_freq[each] += 1

            except Exception as e:
                print e
                sys.exit()
            else:
                fp.close()
        return word_freq

    def get_low_freq_words(self, word_freq, threshold=5):
        return [word for word, freq in word_freq.iteritems() if freq < threshold]


if __name__ == '__main__':
    try:
        corpus_dir = sys.argv[1]
        save_file = sys.argv[2]
    except:
        corpus_dir = "../corpus/20news-bydate/20news-t20-d100/"
        save_file = "lda.mod"

    corpus = MyCorpus(corpus_dir)
    dictionary = corpora.Dictionary(corpus)
    lda_corpus = [dictionary.doc2bow(doc) for doc in corpus]

    lda = LdaModel(lda_corpus, num_topics=20, id2word = dictionary)
    lda.save(save_file)
    # print lda.print_topics(num_topics=20, num_words=10)

    # lda = LdaModel.load(save_file)
    topic_word_dist = lda.print_topics(num_topics=20, num_words=10)
    for each in topic_word_dist:
        print "topic %s)" % each[0]
        print each[1]

    print
    print
    print

    for i in range(len(lda_corpus)):
        topic_prob = lda.get_document_topics(lda_corpus[i], minimum_probability=1e-5)
        topic_prob = sorted(topic_prob, key=lambda d:d[1], reverse=True)
        print "doc %s)" % i
        print " + ".join(["topic %s)*%s" % (topic_prob[j][0], topic_prob[j][1]) for j in range(min(10, len(topic_prob)))])

