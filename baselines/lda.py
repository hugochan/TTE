import sys
import os
import re
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords

word_prog = re.compile('[A-Za-z]+')
cached_stop_words = stopwords.words("english")


class MyCorpus(object):
    """a memory-friendly iterator"""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fpath = os.path.join(self.dirname, fname)
            if not os.path.isfile(fpath) or fname[0] == '.':
                continue
            try:
                with open(fpath) as fp:
                    words = word_prog.findall(fp.read().lower())
                    words = [word for word in words if word not in cached_stop_words]
                    yield words
            except Exception as e:
                print e
                sys.exit()
            else:
                fp.close()

    def __len__(self):
        return sum(1 for fname in os.listdir(self.dirname) if os.path.isfile(os.path.join(self.dirname, fname)) and not fname[0] == '.')

corpus_dir = "linux/corpus/20news-bydate/tiny/"
corpus = MyCorpus(corpus_dir)
dictionary = corpora.Dictionary(corpus)
lda_corpus = [dictionary.doc2bow(doc) for doc in corpus]

lda = LdaModel(lda_corpus, num_topics=10, id2word = dictionary)
print(lda.print_topics(num_topics=10, num_words=5))
