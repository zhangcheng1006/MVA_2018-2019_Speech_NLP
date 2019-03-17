######################################################################################
# This python file implements the PCFG and can create the pcfg from a training file. #
######################################################################################

from nltk.tree import Tree
import re

class Grammer(object):
    '''The class Grammer defines the Probabilistic Context-Free Grammar (PCFG), made of.
        --- left-hand-side tags -> right-hand-side tags with its probability, by self.lhs_rhs_prob
        --- tag -> token with the probability of that the token is generating from the tag, by self.token_tag_prob
    '''
    def __init__(self):
        '''Init fields of the class.
            They are all empty. Only after self.create_pcfg() is called, they get PCFG from the training set.
        '''
        self.lhs_count = {}
        self.lhs_rhs_count = {}
        self.lhs_rhss_dict = {}
        self.token_count = {}
        self.token_tag_count = {}
        self.token_tags_dict = {}
        self.tags = []
        self.tags2id = {}
        self.id2tags = {}
        self.lhs_rhs_prob = {}
        self.token_tag_prob = {}

    def __clean_tags(self):
        '''To create two dictionaries: 
            --- self.tags2id, which maps tag to an unique id
            --- self.id2tag, the inverse map
        '''
        self.tags = list(set(self.tags))
        for i, tag in enumerate(self.tags):
            self.tags2id[tag] = i
            self.id2tags[i] = tag

    def __cross_tree(self, t):
        '''Traverse the tree produced from a training sentence, in order to store rules.
        ------------------------------
            Input:
                t: nltk Tree object
        ------------------------------
            Do not return anything, but updates the dictionary of rules
        '''
        lhs = t.label()
        self.tags.append(lhs)
        if len(t) == 1:
            if type(t[0]) == str:
                token = str(t[0])
                self.__storeLexicon(token, lhs)
            else:
                self.tags.append(t[0].label())
                self.__storeRule(lhs, t[0].label())
                self.__cross_tree(t[0])
        else:
            assert(len(t) == 2)
            rhs = tuple([t[0].label(), t[1].label()])
            self.tags.append(rhs[0])
            self.tags.append(rhs[1])
            self.__storeRule(lhs, rhs)
            self.__cross_tree(t[0])
            self.__cross_tree(t[1])
            

    def __storeLexicon(self, token, lhs):
        '''To store a rule tag -> token
        --------------------------------
            Input:
                token: the token in this rule
                lhs: the tag which generates this token
        --------------------------------
            Do not return anything, but stores the rule by updating the dictionary
        '''
        if token in self.token_count.keys():
            self.token_count[token] += 1.
            if (token, lhs) in self.token_tag_count.keys():
                self.token_tag_count[(token, lhs)] += 1.
            else:
                self.token_tag_count[(token, lhs)] = 1.
                self.token_tags_dict[token].append(lhs)
        else:
            self.token_count[token] = 1.
            self.token_tags_dict[token] = [lhs]
            self.token_tag_count[(token, lhs)] = 1.

    def __storeRule(self, lhs, rhs):
        '''To store a rule lhs -> rhs
        -------------------------------
            Input:
                lhs: the left-hand-side tag
                rhs: the right-hand-side tag
        -------------------------------
            Do not return anything, but stores the rule by updating the dictionary
        '''
        if lhs in self.lhs_count.keys():
            self.lhs_count[lhs] += 1.
            if (lhs, rhs) in self.lhs_rhs_count.keys():
                self.lhs_rhs_count[(lhs, rhs)] += 1.
            else:
                self.lhs_rhs_count[(lhs, rhs)] = 1.
                self.lhs_rhss_dict[lhs].append(rhs)
        else:
            self.lhs_count[lhs] = 1.
            self.lhs_rhss_dict[lhs] = [rhs]
            self.lhs_rhs_count[(lhs, rhs)] = 1.

    def create_pcfg(self, filename):
        '''Create the PCFG
        ------------------------------
            Input:
                filename: the path to the training file
        ------------------------------
            Do not return anything, but stores all rules in the Grammar object
        '''
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                t = Tree.fromstring(line.strip())
                t.chomsky_normal_form()
                self.__cross_tree(t[0])
                
        # get tag2id and id2tag dictionary
        self.__clean_tags()
        
        # get lhs to rhs probability
        for rule, value in self.lhs_rhs_count.items():
            lhs = rule[0]
            prob = value / self.lhs_count[lhs]
            self.lhs_rhs_prob[rule] = prob
            # print(lhs, " -> ", rule[1], ": ", prob)

        # get token to tag probability
        for lexicon, value in self.token_tag_count.items():
            token = lexicon[0]
            prob = value / self.token_count[token]
            self.token_tag_prob[lexicon] = prob
            # print(token, ", ", lexicon[1], ": ", prob)

