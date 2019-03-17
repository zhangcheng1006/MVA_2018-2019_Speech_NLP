#############################################################################################################################
# This python file contains the main program, it will process the data, create the grammar and oov, get the parsed results. #
#############################################################################################################################

import os  
from Grammer import *
from OOV import OoV
from split_data import split
from parser import *
import sys

def preprocess(raw_file, processed_file):
    '''Preprocess the data, to remove functional labels
    --------------------------------
        Input:
            raw_file: the path to the file containing the raw data
            processed_file: the file path to store the processed results
    --------------------------------
        Do not return anything, but write the results to processed_file
    '''
    out = open(processed_file, 'w')
    with open(raw_file, 'r') as f:
        for i, line in enumerate(f):
            line = re.sub(r'-[A-Z].{0,3}[A-Z]\s',' ', line)
            line = re.sub(r'-OBJ::OBJ##OBJ/OBJ', '', line)
            line = re.sub(r'-DE_OBJ', '', line)
            line = re.sub(r'-AFF.DEMSUJ', '', line)
            out.write(line)
    out.close()

def flat_print(t):
    '''Print a tree in the correct string format
    -----------------------------------
        Input:
            t: the nltk Tree object
    -----------------------------------
        Return:
            the correct string
    '''
    tag = t.label()
    if len(t) == 1:
        if type(t[0]) == str:
            return '(' + tag + ' ' + str(t[0]) + ')'
        else:
            return '(' + tag + ' ' + flat_print(t[0]) + ')'
    else:
        s = []
        for i in range(len(t)):
            s.append(flat_print(t[i]))
        return '(' + tag + ' ' + ' '.join(s) + ')'

def main(test_file_path, train_file_path='./data/train', train_sent_file='./data/train_sent', output_file='evaluation_data2.parser_output'):
    '''main function to do the parsing task
    -------------------------------------
        Input:
            test_file_path: the path to the test file containing sentences
            train_file_path: the path to the training file. 
            train_sent_file: the path to the training sentence file
            output_file: the path to the output file containing the parsing results on test file
    -------------------------------------
        Do not return anything, but write the results to the output file
    '''
    if not os.path.exists('./processed_data'):
        preprocess("./raw_data", './processed_data')

    if not os.path.exists(train_file_path):
        split('train', 'train_sent', 'dev', 'dev_res', 'test', 'test_res', 'processed_data')

    grammer = Grammer()
    grammer.create_pcfg(train_file_path)

    oov = OoV(grammer)
    oov.get_embeddings('./embedding/polyglot-fr.pkl')
    oov.get_bigram(train_sent_file)

    pred = open(output_file, 'w')
    dev = open(test_file_path, 'r').read().splitlines()
    for i, line in enumerate(dev):
        sent = line.strip()
        s, p = PCYK(sent, grammer, oov)
        s = '( ' + s + ')'
        t = Tree.fromstring(s)
        t.un_chomsky_normal_form(unaryChar='_')
        res = flat_print(t)
        pred.write(res + '\n')
    pred.close()


if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        print("Error: please give the test file path!")
    if len(args) != 2 and len(args) != 5:
        print("Error: do not have correct number of arguments (expected 1 or 4)!")
    if len(args) == 2:
        main(args[1])
    if len(args) == 5:
        main(args[1], args[2], args[3], args[4])
