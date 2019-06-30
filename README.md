# Learning-Python-Through-NLP
N-Gram-based language model , dependency parser, authorship attribution with Keras

This project, which I worked on in June and July 2017, provided an invaluable introduction into many crucial concepts in NLP and Python. 
Before this point, I had had three years of programming experience in Java, Computational Linguistic theory, and university-level math courses. This project was orininally submitted in the course *Statistical Natural Language Processing*, taught by Dr. Cagri Coltekin
 
 Much of the theory on which the implementations were based can be found in the book *Speech and Language Processing*, by Daniel Jurafsky and James H. Martin
 
 # An explanation of each file
1. In `ngram_language_model.py`, I develop various NGram-based Language Models that incorporate:
 
- Maximum Likelihood Estimation (MLE) 
- additive smoothing (both Laplace smoothing and with an optimized α parameter)
- perplexity scores
- Simple back-off N-grams
- Use of NLTK for word and sentence tokenization 

The language model are then trained to successfully perform authorship attribution on the works of eight famous authors from Project Gutenberg
