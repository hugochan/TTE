'''
Created on Sep, 2016

@author: hugo
'''
import os
import sys
import math
from collections import defaultdict
import chardet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(r'\w+')
cached_stop_words = stopwords.words("english")


def save_word_word_network(word_word_freq_file, word_word_freq, filter_list=[]):
    try:
        with open(word_word_freq_file, 'w') as fp:
            for word1, val in word_word_freq.iteritems():
                if word1 in filter_list:
                    continue
                for word2, freq in val.iteritems():
                    if word2 in filter_list:
                        continue
                    fp.write("%s %s %s\n"%(word1, word2, freq))
                    fp.write("%s %s %s\n"%(word2, word1, freq)) # bidirectional
    except Exception as e:
        print e
        sys.exit()
    else:
        fp.close()

def save_word_doc_network(word_doc_freq_file, word_doc_freq, filter_list=[]):
    try:
        with open(word_doc_freq_file, 'w') as fp:
            for word, val in word_doc_freq.iteritems():
                if word in filter_list:
                    continue
                for doc, freq in val.iteritems():
                    fp.write("%s %s %s\n"%(word, doc, freq))
    except Exception as e:
        print e
        sys.exit()
    else:
        fp.close()

def load_data(corpus_path, context_win_size=5):
    half_win_size = int(math.ceil((context_win_size - 1.0) / 2))
    word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
    word_doc_freq = defaultdict(dict) # count the number of times a word appears in a doc
    word_word_freq = defaultdict(dict) # count the number of times a word appears in another word's context
    files = (filename for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)))
    for filename in files:
        if filename[0] == '.':
            continue
        try:
            with open(os.path.join(corpus_path, filename), 'r') as fp:
                text = fp.read().lower()
                try:
                    sentences = sent_tokenize(text)
                except:
                    sentences = sent_tokenize(text.decode(chardet.detect(text)['encoding']))
                for sent in sentences:
                    words = word_tokenizer.tokenize(sent)
                    words = [word for word in words if word not in cached_stop_words]

                    for i in range(len(words)):
                        # word-doc frequency
                        try:
                            word_doc_freq[words[i]][filename] += 1
                        except:
                            word_doc_freq[words[i]][filename] = 1
                        # word frequency
                        word_freq[words[i]] += 1
                        # word-word frequency
                        for j in range(i + 1, min(i + half_win_size + 1, len(words))):
                            if words[i] == words[j]:
                                continue
                            elif words[i] < words[j]:
                                a = words[i]
                                b = words[j]
                            else:
                                a = words[j]
                                b = words[i]
                            try:
                                word_word_freq[a][b] += 1
                            except:
                                word_word_freq[a][b] = 1
        except Exception as e:
            print e
            sys.exit()
        else:
            fp.close()
    return word_freq, word_word_freq, word_doc_freq

def get_low_freq_words(word_freq, threshold=5):
    return [word for word, freq in word_freq.iteritems() if freq < threshold]


if __name__ == "__main__":
    try:
        corpus_path = sys.argv[1]
        word_word_freq_file = sys.argv[2]
        word_doc_freq_file = sys.argv[3]
    except:
        corpus_path = "../corpus/20news-bydate/20news-t20-d100/"
        word_word_freq_file = "word-word.txt"
        word_doc_freq_file = "word-doc.txt"

    word_freq, word_word_freq, word_doc_freq = load_data(corpus_path, context_win_size=5)
    filter_list = get_low_freq_words(word_freq, threshold=5)
    save_word_word_network(word_word_freq_file, word_word_freq, filter_list)
    save_word_doc_network(word_doc_freq_file, word_doc_freq, filter_list)
