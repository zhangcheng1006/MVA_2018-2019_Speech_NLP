################################################################################################
# This python file implements the OoV module and can assign a similar token to an unseen word. #
################################################################################################

import numpy as np
import pandas as pd
import re

# Noramlize digits by replacing them with #
DIGITS = re.compile("[0-9]", re.UNICODE)

def Damerau_Levenshtein_distance(word, token):
    '''Define the Damerau-Levenshein distance between two strings.
       Implemented by dynamic programming.
    -----------------------------------
        Input:
            word: the first string
            token: the second string
    -----------------------------------
        Return:
            the Damerau-Levenshein distance between the two strings
    '''
    l1 = len(word)
    l2 = len(token)
    dp = np.zeros((l1+1, l2+1))
    for i in range(l1+1):
        dp[i, 0] = i
    for j in range(l2+1):
        dp[0, j] = j
    for i in range(1, l1+1):
        for j in range(1, l2+1):
            if word[i-1] == token[j-1]:
                cost = 0
            else:
                cost = 1
            dp[i, j] = min([dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost]) # deletion, insertion and substitution
            # transpose
            if i > 1 and j > 1 and word[i-1] == token[j-2] and word[i-2] == token[j-1]:
                dp[i, j] = min([dp[i, j], dp[i-2, j-2] + cost])
    return dp[l1, l2]



class OoV(object):
    '''The class OoV defines the out of vocabulary module.
        --- self.grammer: it is related to a specific grammar because the out-of-vocabulary words are related to the already seen words in the grammar.
        --- self.vocab, self.embeddings: it is related to a specific word embedding bacause the similarity relies on the word embeddings.
    '''
    def __init__(self, grammer):
        '''Init the OoV module.
        ---------------------------
            Input:
                grammer: an object of Grammer class
        ---------------------------
            initialize the grammer, and vocab and embeddings by None
        '''
        self.grammer = grammer
        self.vocab = None
        self.embeddings = None
        
    def get_embeddings(self, filename):
        '''Get the word embeddings from a pkl file
        -------------------------------
            Input:
                filename: the pickle file containing the word embeddings
        -------------------------------
            Updates self.vocab and self.embeddings from the input file
        '''
        words, embeddings = pd.read_pickle(filename)
        vocab = {}
        for i, w in enumerate(words):
            vocab[w] = i
        self.vocab = vocab
        self.embeddings = embeddings

    def get_bigram(self, filename):
        '''Get the Bigram transition matrix.
        ------------------------------
            Input:
                filename: the file containing raw training sentences without tags
        ------------------------------
            Updates self.bigram
        '''
        all_tokens = list(self.grammer.token_count.keys())
        l = len(all_tokens)
        self.token2id = {}
        self.id2token = {}
        for i, key in enumerate(all_tokens):
            self.token2id[key] = i
            self.id2token[i] = key
        self.bigram = np.zeros((l+1, l+1))
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                words = line.strip().split(' ')
                self.bigram[l, self.token2id[words[0]]] += 1
                for i in range(len(words) - 1):
                    self.bigram[self.token2id[words[i]], self.token2id[words[i+1]]] += 1
                self.bigram[self.token2id[words[-1]], l] += 1
        self.bigram = self.bigram / np.sum(self.bigram, axis=1)

    def case_normalizer(self, word):
        '''Normalize the word case if it is not in the embedding vocabulary
        ------------------------------
            Input:
                word: the word to normalize
        ------------------------------
            Return:
                the case normalized word
        '''
        w = word
        lower = (self.vocab.get(w.lower(), 1e12), w.lower())
        upper = (self.vocab.get(w.upper(), 1e12), w.upper())
        title = (self.vocab.get(w.title(), 1e12), w.title())
        results = [lower, upper, title]
        results.sort()
        index, w = results[0]
        if index != 1e12:
            return w
        return word

    def normalize(self, word):
        '''The whole normalizer including the case of digits.
        ------------------------------
            Input:
                word: the word to normalize
        ------------------------------
            Return:
                the normalized word. We can not ensure that the normalize word will be in the embedding vocabulary.
        '''
        if word not in self.vocab:
            word = DIGITS.sub("#", word)
        if word not in self.vocab:
            word = self.case_normalizer(word)
        return word
    
    def Embedding_similarity(self, word, candidate):
        '''Compute cosine similarity between two words
        -------------------------------
            Input: 
                word: the first word
                candidate: the second word
        -------------------------------
            Return:
                similarity between two words' embeddings
        '''
        word_vec = self.embeddings[self.vocab[word]]
        candidate_vec = self.embeddings[self.vocab[candidate]]
        similarity = np.dot(np.reshape(word_vec, (1, -1)), np.reshape(candidate_vec, (-1, 1))) / (np.linalg.norm(word_vec) * np.linalg.norm(candidate_vec))
        return similarity
    
    def generate_candidates(self, word, k=2):
        '''Generate candidates which are within distance k to the unseen word
        --------------------------------
            Input:
                word: the unseen word
                k: the tolerance of distance, defaut: 2
        --------------------------------
            Return:
                a list of candidates
        '''
        candidates = []
        for token in self.token2id.keys():
            if Damerau_Levenshtein_distance(word, token) <= k:
                candidates.append(token)
        return candidates

    def get_bigram_proba(self, prev_word, next_word, candidates):
        '''Get Bigram probability of each candidates
        ------------------------------
            Input:
                prev_word: the previous word of the unseen word in the sentence
                next_word: the next word of the unseen word in the sentence
                candidates: the list of canidates to be computed
        ------------------------------
            Return:
                a list of probabilities corresponding to each condidate
        '''
        l = self.bigram.shape[0] - 1
        lefts = []
        rights = []
        if prev_word is None :
            for candidate in candidates:
                lefts.append(self.bigram[l, self.token2id[candidate]])
        else:
            if prev_word in self.token2id.keys():
                for candidate in candidates:
                    lefts.append(self.bigram[self.token2id[prev_word], self.token2id[candidate]])
            else:
                lefts = [1.] * len(candidates)
        if next_word is None:
            for candidate in candidates:
                rights.append(self.bigram[self.token2id[candidate], l])
        else:
            if next_word in self.token2id.keys():
                for candidate in candidates:
                    rights.append(self.bigram[self.token2id[candidate], self.token2id[next_word]])
            else:
                rights = [1.] * len(candidates)
        
        probas = np.multiply(lefts, rights)
        if sum(probas) == len(candidates):
            for i, candidate in enumerate(candidates):
                probas[i] = self.grammer.token_count[candidate]
            probas = probas / sum(probas)
        return probas

    def assign_similar_token(self, word, prev_word, next_word, k=2, lamda=1000):
        '''Assign an unique similar token to the unseen word
        --------------------------------
            Input:
                word: the unseen word
                prev_word: the previous word
                next_word: the next word
                k: the tolerance of distance
                lamda: the hyperparameter to adjust the importance of two aimilarities
        --------------------------------
            Return:
                the unique similar token
        '''
        w = self.normalize(word)
        if w in self.vocab.keys():
            all_tokens = list(self.token2id.keys())
            simis = np.zeros(len(all_tokens))
            for _, token in enumerate(all_tokens):
                t = self.normalize(token)
                if t in self.vocab.keys():
                    simis[self.token2id[token]] = self.Embedding_similarity(w, t)
            
            candidates = self.generate_candidates(word, k)
            if len(candidates) == 0:
                return self.id2token[np.argmax(simis)]
            else:
                bigram_probas = self.get_bigram_proba(prev_word, next_word, candidates)
                for i, candidate in enumerate(candidates):
                    idx = self.token2id[candidate]
                    simis[idx] += lamda * bigram_probas[i]
                return self.id2token[np.argmax(simis)]
        else:
            candidates = self.generate_candidates(word, k)
            if len(candidates) == 0:
                candidates = list(self.token2id.keys())
            bigram_probas = self.get_bigram_proba(prev_word, next_word, candidates)
            return candidates[np.argmax(bigram_probas)]
    
    
    # def assign_similar_token(self, word, prev_word, next_word, k=2, lamda=0.01):
    #     '''The naive implementation of assign similar token. You can try this one to compare with the other.
    #     --------------------------------
    #         Input:
    #             word: the unseen word
    #             prev_word: the previous word
    #             next_word: the next word
    #             k: the tolerance of distance
    #             lamda: the hyperparameter to adjust the importance of two aimilarities
    #     --------------------------------
    #         Return:
    #             the unique similar token
    #     '''
    #     candidates = self.generate_candidates(word, k)
    #     word = self.normalize(word)
    #     if len(candidates) == 0 and word in self.vocab.keys():
    #         candidates = list(self.token2id.keys())
    #         simis = np.zeros(len(candidates))
    #         for i, candidate in enumerate(candidates):
    #             candidate = self.normalize(candidate)
    #             if candidate in self.vocab.keys():
    #                 simis[i] = self.Embedding_similarity(word, candidate)
    #         return candidates[np.argmax(simis)]
        
    #     while len(candidates) == 0:
    #         candidates = self.generate_candidates(word, k+1)
    #         k += 1

    #     bigram_probas = self.get_bigram_proba(prev_word, next_word, candidates)

    #     if word not in self.vocab.keys():
    #         return candidates[np.argmax(bigram_probas)]
    #     for i, candidate in enumerate(candidates):
    #         candidate = self.normalize(candidate)
    #         if candidate in self.vocab.keys():
    #             simi = self.Embedding_similarity(word, candidate)
    #             bigram_probas[i] += lamda * simi
    #     return candidates[np.argmax(bigram_probas)]

