#################################################################################################
# This python file implements the CYK parser module and can get the parse result of a sentence. #
#################################################################################################

import numpy as np
from OOV import *
import collections

def build_tree(back, words, low, high, tag, tags_dict):
    '''Build the tree by back tracking the results of CYK algorithm
        Implemented by recursive function
    -------------------------------------
        Input:
            back: the resultant table of CYK algorithm
            words: the list of words in the sentence
            low: the lowest index to consider
            high: the highest index to consider
            tag: the tag to consider
            tags_dict: the dictionary mapping tags to id, so that we can find the corresponding index of the tag in the back table
    -------------------------------------
        Return:
            a string of the parsing result between the low and high indexes, under the bracket format
    '''
    if back[low, high, tags_dict[tag]] is None:
        s = '(' + tag + ' ' + words[low] + ')'
        return s
    else:
        k, B, C = back[low, high, tags_dict[tag]]
        left = build_tree(back, words, low, k, B, tags_dict)
        right = ''
        if C is not None:
            right = build_tree(back, words, k, high, C, tags_dict)
        if right != '':
            return '(' + tag + ' ' + left + ' ' + right + ')'
        return '(' + tag + ' ' + left + ')'       


def PCYK(sentence, grammer, oov):
    '''Probabilistic CYK algorithm
    --------------------------------
        Input:
            sentence: the sentence to parse
            grammer: the PCFG object
            oov: the out of vocabulary object
    --------------------------------
        Return:
            the string of parsing result, under bracket format
    '''
    words = sentence.strip().split(' ')
    # print(words)
    tags_dict = grammer.tags2id
    token_tag_prob = grammer.token_tag_prob
    token_tags_dict = grammer.token_tags_dict
    lhs_rhs_prob = grammer.lhs_rhs_prob
    lhs_rhss_dict = grammer.lhs_rhss_dict
   
    table = np.zeros((len(words), len(words)+1, len(tags_dict.keys())))
    back = [[[None for i in range(len(tags_dict.keys()))] for j in range(len(words)+1)] for l in range(len(words))]
    back = np.array(back)

    for j in range(1, len(words)+1):
        word = words[j-1]
        if word not in token_tags_dict.keys():
            print("original word: " + word)
            if j == 1:
                prev_word = None
            else:
                prev_word = words[j-2]
            if j == len(words):
                next_word = None
            else:
                next_word = words[j]
            word = oov.assign_similar_token(word, prev_word, next_word)
            print("similar word: " + word)
        tags = token_tags_dict[word]
        for tag in tags:
            table[j-1, j, tags_dict[tag]] = token_tag_prob[(word, tag)]
        
        for i in range(j-1, -1, -1):
            for lhs, rhss in lhs_rhss_dict.items():
                for rhs in rhss:
                    if type(rhs) is tuple:
                        B = rhs[0]
                        C = rhs[1]
                        for k in range(i+1, j):
                            if table[i, k, tags_dict[B]] > 0 and table[k, j, tags_dict[C]] > 0:
                                if table[i, j, tags_dict[lhs]] < lhs_rhs_prob[(lhs, rhs)] * table[i, k, tags_dict[B]] * table[k, j, tags_dict[C]]:
                                    table[i, j, tags_dict[lhs]] = lhs_rhs_prob[(lhs, rhs)] * table[i, k, tags_dict[B]] * table[k, j, tags_dict[C]]
                                    back[i, j, tags_dict[lhs]] = (k, B, C)
                    else:
                        if table[i, j, tags_dict[rhs]] > 0:
                            if table[i, j, tags_dict[lhs]] < lhs_rhs_prob[(lhs, rhs)] * table[i, j, tags_dict[rhs]]:
                                table[i, j, tags_dict[lhs]] = lhs_rhs_prob[(lhs, rhs)] * table[i, j, tags_dict[rhs]]
                                back[i, j, tags_dict[lhs]] = (j, rhs, None)
        
    return build_tree(back, words, 0, len(words), 'SENT', tags_dict), table[0, len(words), tags_dict['SENT']]

