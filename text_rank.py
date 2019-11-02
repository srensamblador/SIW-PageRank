from common import Graph, WeightedGraph
from similitude.BagOfWords import BagOfWords
from similitude.Coefs import coef_cosine

from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
import string
import numpy as np
import argparse

class TextRank:
    def extract_keywords(self, text):
        tokens = word_tokenize(text)
        tokens = pos_tag(tokens)  # For syntax filtering, see comment further down

        keyword_list = []
        for token, tag in tokens:
            token = token.lower()
            token = token.strip(string.punctuation)
            if token and token not in string.punctuation \
                    and token not in stopwords.words("english") \
                    and tag in ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]:  
                    # Only nouns and adjectives are added to the graph, Mihalcea and Tarau's paper
                    # claims they gave the best results
                    keyword_list.append(token)

        return keyword_list

    def coocurrent_terms(self, terms, window=2):
        '''
            Co-ocurrence is used to establish edges among vertices. Each term is a vertex
            and if two vertices appear within the specified window they are connected by an
            edge
        '''     
        term_edges = []
        for i, word in enumerate(terms):
            for j in range(i + 1, i + window):
                if j >= len(terms):
                    break
                edge = [word, terms[j]]
                if edge not in term_edges:
                    term_edges.append(edge)
        return term_edges

    def extract_sentences(self, text):
        # Using nltk sentence tokenizer to divide the text in sentences
        sentences = sent_tokenize(text)
        processed_sentences = []
        # Basic cleanup
        for sentence in sentences:
            sentence = " ".join(sentence.splitlines())
            sentence.strip()
            processed_sentences.append(sentence)
        return processed_sentences

    def get_similarity_edges(self, sentences):
        """
            Using cosine coefficient as the similarity value to calculate the weights of the graph
        """
        sent_edges = []
        for i, sent in enumerate(sentences):
            bag1 = BagOfWords(sent)
            for j in range(len(sentences)):
                if (j == i):
                    continue
                bag2 = BagOfWords(sentences[j]) # Bag of Words from the third assignment
                cosine_sim = coef_cosine(bag1, bag2)
                edge = [(sent, sentences[j]), cosine_sim] # [(node A, node B), cosine_sim(a,b)]
                if edge not in sent_edges:
                    sent_edges.append(edge)
        return sent_edges

def get_top_keywords(text, num_words = 10, window=2):
    tr = TextRank()
    keywords = tr.extract_keywords(text)
    edges = tr.coocurrent_terms(keywords, window)
    g = Graph(edges, undirected=True) # Using an undirected graph
    scores = g.page_rank()

    sorted_result = [(node, value) for node, value in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_result[:min(len(sorted_result), num_words)]

def get_top_sentences(text, num_sentences = 10):
    tr = TextRank()
    sentences = tr.extract_sentences(text)
    sent_edges = tr.get_similarity_edges(sentences)
    g = WeightedGraph(sent_edges)
    scores = g.page_rank()

    sorted_result = [(node, value) for node, value in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return sorted_result[:min(len(sorted_result), num_sentences)]

def parse_args():
   parser = argparse.ArgumentParser(description="Text Rank implementation")
   parser.add_argument("file", help="Text file")
   parser.add_argument("-k", "--keywords", default=10, type=int, help="Number of top keywords to extract")
   parser.add_argument("-s", "--sentences", default=10, type=int, help="Number of sentences to extract")
   parser.add_argument("-w", "--window", default=2, type=int, help="Co-ocurrence window for keyword text-ranking") 
   args = parser.parse_args()
   return args

def main(args):
    with open(args.file, encoding="UTF-8") as f:
        text = f.read()
        
        # Top Keywords
        print("Extracting top %d keywords"%(args.keywords))
        keywords = get_top_keywords(text, args.keywords, args.window)
        for word, score in keywords:
            print("- %s: %f"%(word, score))

        # Top Sentences
        print("Extracting top %d sentences"%(args.sentences))
        sentences = get_top_sentences(text, args.sentences)
        for sentence, score in sentences:
            print("- %s: %f"%(sentence, score))
        


main(parse_args())