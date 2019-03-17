###################################################################################################################
# This python file splits data to training set (2479 lines), validation set (309 lines) and test set (311 lines). #
###################################################################################################################

import numpy as np

np.random.seed(2019)

def split(train_filename, train_sent, dev_filename, dev_res, test_filename, test_res, raw_filename):
    '''function to split data, the split results will be stored under data folder
    ---------------------------------------------
        Input:
            train_filename: the path to the output training file
            train_sent: the path to the output training sentence file
            dev_filename: the path to the output validation file
            dev_res: the path to the output validation result file
            test_filename: the path to the output test file
            test_res: the path to the output test result file
            raw_filename: the path to the file containing raw data
    ---------------------------------------------
        Do not return anything, but write results to files
    '''
    train_f = open('./data/'+train_filename, "w")
    train_sent = open('./data/'+train_sent, "w")
    dev_f = open('./data/'+dev_filename, "w")
    test_f = open('./data/'+test_filename, "w")
    dev_res = open('./data/'+dev_res, "w")
    test_res = open('./data/'+test_res, "w")

    raw_lines = []
    with open(raw_filename, 'r') as f:
        for _, line in enumerate(f):
            if not line.endswith('\n'):
                line += '\n'
            raw_lines.append(line)

    total_num = len(raw_lines)
    indexes = np.array(range(total_num))
    np.random.shuffle(indexes)
    print(total_num)
    train_num = int(total_num * 0.8)
    train_indexes = indexes[0:train_num]
    print(train_num)
    dev_num = int(total_num * 0.1)
    dev_indexes = indexes[train_num:train_num+dev_num]
    print(dev_num)
    test_num = total_num - train_num - dev_num
    test_indexes = indexes[train_num+dev_num:]
    print(test_num)

    for i in train_indexes:
        line = raw_lines[i]
        train_f.write('{}'.format(line))

        words = []
        for j, c in enumerate(line):
            if c == ')' and line[j-1] != ")":
                end = j-1
                start = end
                while line[start] != " ":
                    start -= 1
                words.append(line[start+1:end+1])
        sentence = ' '.join(words)
        train_sent.write('{}\n'.format(sentence))
    train_f.close()
    train_sent.close()

    for i in dev_indexes:
        line = raw_lines[i]
        words = []
        for j, c in enumerate(line):
            if c == ')' and line[j-1] != ")":
                end = j-1
                start = end
                while line[start] != " ":
                    start -= 1
                words.append(line[start+1:end+1])
        sentence = ' '.join(words)
        dev_f.write('{}\n'.format(sentence))
        dev_res.write('{}'.format(line))
    dev_f.close()
    dev_res.close()

    for i in test_indexes:
        line = raw_lines[i]
        words = []
        for j, c in enumerate(line):
            if c == ')' and line[j-1] != ")":
                end = j-1
                start = end
                while line[start] != " ":
                    start -= 1
                words.append(line[start+1:end+1])
        sentence = ' '.join(words)
        test_f.write('{}\n'.format(sentence))
        test_res.write('{}'.format(line))
    test_f.close()
    test_res.close()

