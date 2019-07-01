# Learning-Python-Through-NLP
N-Gram-based language model, dependency parser, authorship attribution with Keras

This project, which I worked on in June and July 2017, provided an invaluable introduction into many crucial concepts in NLP and Python. 
Before this point, I had had three years of programming experience in Java, Computational Linguistic theory, and university-level math courses. This project was originally submitted in the course *Statistical Natural Language Processing*, taught by Dr. Cagri Coltekin
 
 Much of the theory on which the implementations were based can be found in the book *Speech and Language Processing*, by Daniel Jurafsky and James H. Martin
 
 # An explanation of each file
1. In `ngram_language_model.py`, I develop various NGram-based Language Models that incorporate:
 
 - Maximum Likelihood Estimation (MLE) 
 - additive smoothing (both Laplace smoothing and with an optimized α parameter)
 - perplexity scores
 - Simple back-off N-grams
 - Use of `NLTK` for word and sentence tokenization 

 The language model are then trained to successfully perform authorship attribution on the works of eight famous authors who have multiple works in the public domain

2. In `dependency_parser.py`, I implement a dependency parser using Universal Dependencies(UD) Version 2 treebanks for training and testing. The steps taken to build the parser were:

 - Feature extraction: get word forms and POS tags for terms on buffer and stack, have pre-made oracle predict labels for upcoming actions
 - Classification: train deep NN incorporating embeddings and GRU in Keras to predict next parser action label on a given UD entry
 - Parsing: a function that top predictions from NN to output the one best dependency tree
 - Training, Tuning hyperparameters on development set, and calculating final performance on test set

3. In `deep_tweets_embedding.py`, I participate in Germeval Task 2017: A Shared Task on Aspect-based Sentiment in Social Media Customer Feedback. It focuses on labelling customer tweets regarding Deutsche Bahn, the national German train operator

 After data extraction and cleanup, I prepare an embedding with a Gated Recurrent Unit in Keras running on Tensorflow. The network makes use of softmax activation and the Adam optimizer to achieve 86% accuracy - Not bad for my first attempt at a challenging shared task in twitter sentiment analysis!
