from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

class BagOfWords(object):
    def __init__(self, text=None, values=None):
        self.values = {}
        self.n = 0
        if text is not None:
            self.string_to_bag_of_words(text)
        elif values is not None:
            self.values = values.copy()

    def __str__(self):
        return self.values.__str__()

    def __len__(self):
        return self.values.__len__()

    def __iter__(self):
        for key in self.values:
            yield key, self.values[key]

    def intersection(self, other):
        intersection= {}
        intersected_keys = self.values.keys() & other.values.keys()
        for key in intersected_keys:
            intersection[key] = min(self.values[key], other.values[key])
        return BagOfWords(values=intersection)

    def union(self, other):
        union = self.values
        for key in other.values:
            if key in union:
                union[key] += other.values[key]
            else:
                union[key] = other.values[key]
        return BagOfWords(values=union)

    def string_to_bag_of_words(self, text):
        words = word_tokenize(text)
        english_stopwords = set(stopwords.words("english"))
        stemmer = PorterStemmer()

        for word in words:
            word = word.lower()
            word = word.strip(string.punctuation)
            if word and word not in english_stopwords:
                word = stemmer.stem(word)
                if word not in self.values:
                    self.values[word] = 1
                else:
                    self.values[word] += 1
