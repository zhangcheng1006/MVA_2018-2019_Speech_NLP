# The second TP concerning PCFG and CYK parser.

## Run the system

You can run the system by './run.sh $test_file_path' in the command line. The test file path is necessary. In our system, $test_file_path = ./data/test. You can also take your own test file. There are also three other arguments corresponding to the training file, training sentences file and the results output file. However, the other three arguments are not necessary. You can just leave them blank and it will take the default arguments.  The default training file is './data/train', the default training sentence file is './data/train_sent' and the default output file is 'evaluation_data2.parser_output'. You can also provide other file paths as arguments, in this case, you need to provide all four files.

After the results are written in the output file, you can evaluate the performance by the EVALB module. (Here we evaluate the performance by the pre-calculated results 'evaluation_data.parser_output')

You just need to do: 

- 'cd EVALB'
- 'make'
- './evalb -p sample/sample.prm ../data/test_res ../evaluation_data.parser_output'


## More details

The system contains five python files:
- split_data.py: splits data to training set, validation set and test set. they will be stored under 'data' folder, so you need to first create this folder.
- Grammer.py: defines the PCFG grammar.
- OOV.py: defines the Out-of-Vocabulary module.
- parser.py: implements the CYK parser.
- main.py: the main program to do the parsing task.

In main.py, we first process the raw data, so you need to have a file names 'raw_data' which contains the the 'SEQUOIA treebank v6.0' dataset.

Then it will create a 'processed_data' which eliminates the functional labels. Then it splits the processed data to training set, validation set and test set under 'data' folder.

Then it calls the Grammer module to create PCFG. 

Then it builds the OoV module according to the built grammer. Specifically, it takes the 'polyglot-fr.pkl' file under 'embedding' folder to read word embeddings.

Finally, it takes the test file and write the parsing results to a file.
