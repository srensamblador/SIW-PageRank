from common import Graph
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np

def extract_keywords(text):
    tokens = word_tokenize(text)
    for token in tokens:
        token.lower()
        token.strip(string.punctuation)
        tokens = [token for token in tokens if token not in string.punctuation \
            and token not in stopwords.words("english")]

    return tokens

def coocurrent_terms(keywords, coocurrent_size=2):
    edges = []
    for i, word in enumerate(keywords):
        for j in range(i+1, i+coocurrent_size):
            if j >= len(keywords):
                break
            edge = [word, keywords[j]]
            if edge not in edges:
                edges.append(edge)
    return edges


with open("text.txt") as f:
    keywords = extract_keywords(f.read())
    edges = coocurrent_terms(keywords, 4)
    print(edges)
    keyword_list = list(set(keywords))
    n = len(keyword_list)
    g = np.zeros((n,n))
    for w1, w2 in edges:
        i, j = keyword_list.index(w1), keyword_list.index(w2)
        g[i][j] = 1
    g = g + g.T - np.diag(g.diagonal())
    norm = np.sum(g, axis=0)
    g_norm = np.divide(g, norm, where=norm != 0)
    g = g_norm

    pr = np.ones(n)

    previous_pr = 0
    for _ in range(1000):
        pr = (1-0.85)/n + 0.85 * np.dot(g, pr)
        if abs(previous_pr - sum(pr)) < 1e-8:
            break

    scores = {}
    for index, word in enumerate(keyword_list):
        scores[word] = pr[index]

    print([(node, value) for node, value in sorted(scores.items(), key=lambda x: x[1], reverse=True)])
