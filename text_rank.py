from common import Graph
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import string
import numpy as np

def extract_keywords(text):
    tokens = word_tokenize(text)
    tokens = pos_tag(tokens)
    keyword_list = []
    for token, tag in tokens:
        token = token.lower()
        token = token.strip(string.punctuation)
        if token and token not in string.punctuation \
                and token not in stopwords.words("english")\
                and tag in ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]:
            keyword_list.append(token)

    return keyword_list


def coocurrent_terms(terms, window=2):
    term_edges = []
    for i, word in enumerate(terms):
        for j in range(i + 1, i + window):
            if j >= len(terms):
                break
            edge = [word, terms[j]]
            if edge not in term_edges:
                term_edges.append(edge)
    return term_edges


with open("text.txt", encoding="UTF-8") as f:
    keywords = extract_keywords(f.read())
    dic = {}
    for word in keywords:
        if word not in dic:
            dic[word] = 1
        else:
            dic[word] += 1
    print(sorted(dic.items(), key=lambda t: t[1], reverse = True))

    edges = coocurrent_terms(keywords, 2)
    g = Graph(edges, undirected=True)
    scores = g.page_rank()

    print([(node, value) for node, value in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
