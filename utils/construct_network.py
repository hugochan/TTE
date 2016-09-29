'''
Created on Sep, 2016

@author: hugo
'''
import re
import os
import sys
from collections import defaultdict
from nltk.corpus import stopwords


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
            fp.close()
    except Exception as e:
        print e
        sys.exit()

def save_word_doc_network(word_doc_freq_file, word_doc_freq, filter_list=[]):
    try:
        with open(word_doc_freq_file, 'w') as fp:
            for word, val in word_doc_freq.iteritems():
                if word in filter_list:
                    continue
                for doc, freq in val.iteritems():
                    fp.write("%s %s %s\n"%(word, doc, freq))
            fp.close()
    except Exception as e:
        print e
        sys.exit()

def load_data(corpus_path, context_win_size=2):
    cached_stop_words = stopwords.words("english")
    word_prog = re.compile('[A-Za-z]+')
    word_freq = defaultdict(int) # count the number of times a word appears in a corpus
    word_doc_freq = defaultdict(dict) # count the number of times a word appears in a doc
    word_word_freq = defaultdict(dict)
    # doc_set = []
    files = (filename for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)))
    for filename in files:
        if filename[0] == '.':
            continue
        with open(os.path.join(corpus_path, filename), 'r') as fp:
            words = word_prog.findall(fp.read().lower())
            words = [word for word in words if word not in cached_stop_words]

            for i in range(len(words)):
                # word-doc frequency
                try:
                    word_doc_freq[words[i]][filename] += 1
                except:
                    word_doc_freq[words[i]][filename] = 1
                # word frequency
                try:
                    word_freq[words[i]] += 1
                except:
                    word_freq[words[i]] = 1
                # word-word frequency
                for j in range(i + 1, min(i + context_win_size + 1, len(words))):
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
        corpus_path = "linux/corpus/20news-bydate/20news-t20-d100/"
        word_word_freq_file = "word-word.txt"
        word_doc_freq_file = "word-doc.txt"

    word_freq, word_word_freq, word_doc_freq = load_data(corpus_path)
    filter_list = get_low_freq_words(word_freq, 2)
    save_word_word_network(word_word_freq_file, word_word_freq, filter_list)
    save_word_doc_network(word_doc_freq_file, word_doc_freq, filter_list)
