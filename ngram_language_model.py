

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

 #sources: https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python
import nltk
import sys
from itertools import islice, izip
from collections import Counter
import math
import numpy

# !!!We also need to include the end-of-sentence marker
# </s> (but not the beginning-of-sentence marker <s>) in the total count of word tokens
# N.

def tokenize(story): #substitute filename!!!!!!
	sent_word_list = []
	word_list = []
	sent_text = nltk.sent_tokenize(story)
   	for sentence in sent_text:
   		tokenized_text = nltk.word_tokenize(sentence)
   		for word in tokenized_text:
   			word_list.append(word)
   		sent_word_list.append(word_list)
   		word_list= []
	return sent_word_list
   		

class NGram:
	def __init__(self,n):
		self.n = n
		self.current_ngrams  = []
		self.cur_ngrams_counter = Counter()
		self.cur_lessgrams_counter = Counter()
		self.type_counter = Counter()
		self.best_alpha = .1


	def add_pad(self, sequence):
		start_pad = ['<s>']*((self.n)-1)
		end_pad = ['</s>']*((self.n)-1)
		pad_seq = start_pad + sequence + end_pad
		return pad_seq

	def update(self, sequence):
		
		pad_seq = self.add_pad(sequence)

		if self.n > 1:
			new_ngrams = zip(*[pad_seq[i:] for i in range(self.n)])

			new_lessgrams = zip(*[pad_seq[i:] for i in range(self.n-1)])
			new_types = sequence
			self.type_counter.update(new_types)
		else:
			new_ngrams = sequence
			new_lessgrams = []
			
		self.cur_ngrams_counter.update(new_ngrams)
		self.cur_lessgrams_counter.update(new_lessgrams)

		return self.cur_ngrams_counter
		
  		

  	def prob_mle(self, sequence):
  		prob = 0
  		overall_prob = 0
  		pad_seq = self.add_pad(sequence)
  		seq_ngrams = zip(*[pad_seq[i:] for i in range(self.n)])


  		if self.n > 0:
  			for w in seq_ngrams:
  				if self.n < 2: 
  					prob = numpy.log2(float(self.cur_ngrams_counter[w[0]])/float(sum(self.cur_ngrams_counter.values())))
  					overall_prob += prob
  				else: 
  					one_less = w[:self.n-1]
  					prob = numpy.log2(float(self.cur_ngrams_counter[w])/float(self.cur_lessgrams_counter[one_less]))
  		 			overall_prob += prob
  		
  		return overall_prob



	def prob_add(self, sequence, alpha=1):
		prob = 0
  		overall_prob = 0
  		pad_seq = self.add_pad(sequence)
  		seq_ngrams = zip(*[pad_seq[i:] for i in range(self.n)])

  		if self.n > 0:
  			for w in seq_ngrams:
  				if self.n < 2:
  					prob = float(self.cur_ngrams_counter[w[0]]+alpha)/float(sum(self.cur_ngrams_counter.values())+ (alpha*(len(self.cur_ngrams_counter))))
  					overall_prob *= prob
  				else:
  					one_less = w[:self.n-1]
  					prob = float(self.cur_ngrams_counter[w]+alpha)/float((self.cur_lessgrams_counter[one_less])+(alpha*(len(self.type_counter))))
					overall_prob *= prob

		return overall_prob

					

	
	def perplexity(self, sentences, alpha=1):
		sent_word_list = tokenize(sentences)
		calculated_mle = 0
		n = 0

		for sent in sent_word_list:
			calculated_mle += self.prob_add(sent, alpha)
			for word in sent:
				n +=1
			n +=1 #add an end of sentence marker for each sentence

			# log_perplexity = pow(calculated_mle, -1/float(n))
			#perplexity = 2 ** log_perplexity #is this right? test without log
			#print(perplexity)
			# print(log_perplexity)
			return (2 ** (calculated_mle/ -n))


	def estimate_alpha(self, sentences):
		prev_val = self.perplexity(sentences, alpha=0)

		for tuner in numpy.linspace(0.0, 1.0, 11): #change step?
			test_per = self.perplexity(sentences, alpha=tuner)
			if test_per < prev_val:
				prev_val = test_per

		self.best_alpha = tuner
		return tuner

	def training(self, letter):
		#assuming training files are in current directory
		files_list_c = []
		files_list_d = []
		cond = dict = {'c': files_list_c, 'd': files_list_d}
		ext = ".txt"
		files_1 = "c"
		files_2 = "d"
		calc_alpha = False
		alpha = 1
		
		files_list_c.extend([files_1+"0"+str(i)+ext for i in range(1,11)])
		files_list_c.extend([files_1+str(i)+ext for i in range(11,25)])
		files_list_d.extend([files_2+"0"+str(i)+ext for i in range(1,11)])
		files_list_d.extend([files_2+str(i)+ext for i in range(11,22)])
		
		#maybe a try statement for file reading? check github
		for file in cond[letter]:
			print file
			with open(file) as f:
				lines = f.read()
				
				sent_word_list = tokenize(lines)
        		for sent in sent_word_list:
        			self.update(sent)
        		if file == (letter + "04.txt"):
        			self.estimate_alpha(lines)

class BackoffNGram:
	
	#backoff n-grams do not need an n-value! that's the whole point
	#I should have used instances of NGrams within this class, but was unfamiliar with python when i wrote it
	def __init__(self,n):
		self.n = n #always given value of 3 during instantiation
		self.current_ngrams  = []
		self.cur_ngrams_counter = Counter()
		self.cur_lessgrams_counter = Counter()
		self.cur_less2grams_counter = Counter()
		self.type_counter = Counter()
		self.unique_tri = Counter()
  		self.unique_bi = Counter()
  		self.lamd = 0
  		self.beta = 0
  		self.best_alpha = 1 #for brevity, it was always 1 in for the NGrams
  
	def update(self, sequence):
		pad_seq = self.add_pad(sequence)

		new_ngrams = zip(*[pad_seq[i:] for i in range(self.n)])
		new_lessgrams = zip(*[pad_seq[i:] for i in range(self.n-1)])
		new_types = sequence
		self.type_counter.update(new_types)

		self.cur_ngrams_counter.update(new_ngrams)
		self.cur_lessgrams_counter.update(new_lessgrams)
		self.cur_less2grams_counter.update(sequence)

		return self.cur_ngrams_counter

	def update_unique_ngms(self):
		for trigram in self.cur_ngrams_counter:
			if self.cur_ngrams_counter[trigram] == 1:
				self.unique_tri.update(trigram) #used counter instead of set so i could check

		self.lamd = float(sum(self.unique_tri.values())/sum(self.cur_ngrams_counter.values()))


		for bigram in self.cur_lessgrams_counter:
			if self.cur_lessgrams_counter[bigram] == 1:
				self.unique_bi.update(bigram)

		self.beta = float(sum(self.unique_bi.values())/sum(self.cur_lessgrams_counter.values()))
	

	def prob(self,sequence):
		pad_seq = self.add_pad(sequence)
		overall_prob = 0

		#GENERALIZE LATER

  		seq_ngrams = zip(*[pad_seq[i:] for i in range(self.n)])
  		seq_lessgrams = zip(*[pad_seq[i:] for i in range(self.n-1)])
  		seq_less2grams = sequence


  		for a in seq_ngrams:
  			if self.cur_ngrams_counter[a] > 0:
  				mle_prob = float((self.cur_ngrams_counter[a]))/float((self.cur_lessgrams_counter[a[:2]]))
  				fin_prob = numpy.log2(float((1-self.lamd)* mle_prob))
  				overall_prob += fin_prob
  				#!!!Is this the right was to multiply/add?
  				

  			elif self.cur_lessgrams_counter[a[:2]] > 0:
  				mle_prob = float(self.cur_lessgrams_counter[a[:2]])/float(self.cur_less2grams_counter[a[:1][0]])
  				#divide by zero error if we haven't filled in lamda or beta
  				fin_prob = numpy.log2(float(self.lamd*(1-self.beta)* mle_prob))
  				overall_prob += fin_prob
  				
  				
  			else:
  				add_prob = float(self.cur_less2grams_counter[a[:1][0]]+self.alpha)/float(sum(self.cur_ngrams_counter.values())+(self.alpha*(len(self.cur_ngrams_counter))))
  				#do we multiply here?
  				fin_prob = numpy.log2(float(self.lamd * self.beta * add_prob))

  				overall_prob += fin_prob

  				
  		return overall_prob

	def perplexity(self,sentences): 
		sent_word_list = tokenize(sentences)
		calculated_mle = 0
		n = 0

		for sent in sent_word_list:
			calculated_mle += self.prob_add(sent)
			for word in sent:
				n +=1
			n +=1 

		return (2 ** (calculated_mle/ -n))

	def add_pad(self, sequence):
		start_pad = ['<s>']*((self.n)-1)
		end_pad = ['</s>']*((self.n)-1)
		pad_seq = start_pad + sequence + end_pad
		return pad_seq



	def training(self, letter):
		#assuming training files are in current directory
		files_list_c = []
		files_list_d = []
		cond = dict = {'c': files_list_c, 'd': files_list_d}
		ext = ".txt"
		files_1 = "c"
		files_2 = "d"
		calc_alpha = False

		files_list_c.extend([files_1+"0"+str(i)+ext for i in range(1,11)])
		files_list_c.extend([files_1+str(i)+ext for i in range(11,25)])
		files_list_d.extend([files_2+"0"+str(i)+ext for i in range(1,11)])
		files_list_d.extend([files_2+str(i)+ext for i in range(11,22)])
		
		#maybe a try statement for file reading? check github
		for file in cond[letter]:
			print file
			with open(file) as f:
				
				lines = f.read()
				
				sent_word_list = tokenize(lines)
        		for sent in sent_word_list:
        			self.update(sent)
        		
def main():


	files_list_t = []
	uni_c = NGram(1)
	uni_c.training("c")
	uni_d = NGram(1)
	uni_d.training("d")
	bi_c = NGram(2)
	bi_c.training("c")
	bi_d = NGram(2)
	bi_d.training("d")
	tri_c = NGram(3)
	tri_c.training("c")
	tri_d = NGram(3)
	tri_d.training("d")
	back_c = BackoffNGram(3)
	back_c.training("c")
	back_d = BackoffNGram(3)
	back_d.training("d")

	#tabulate is probably a much better idea here
	#from tabulate import tabulate
	headers = ("c-1g-l c-1g-a d-1g-l d-1g-a c-2g-l c-2g-a d-2g-l d-2g-a c-3g-l c-3g-a d-3g-l d-3g-a c-backoff d-backoff")
	print(headers)
	scores_for_file = []

	ext = ".txt"
	
	
		#does this get everything?
	files_list_t.extend(["t0"+str(i)+ext for i in range(1,6)])
	files_list_t.extend(["c00"+ext])
	files_list_t.extend(["d00"+ext])

	for file in files_list_t:
			with open(file) as f:
				lines = f.read()
				
				
         		out = str(file + ",")
         		#specify size of number
         		num = "{0:.3f}"

        		end_str += num.format(uni_c.perplexity(lines)) + ","
        		end_str += num.format(uni_c.perplexity(lines, uni_c.best_alpha)) + ","
        		end_str += num.format(uni_d.perplexity(lines)) + ","
        		end_str += num.format(uni_d.perplexity(lines, uni_d.best_alpha)) + ","
        		end_str += num.format(bi_c.perplexity(lines)) + ","
        		end_str += num.format(bi_c.perplexity(lines, bi_c.best_alpha)) + ","
        		end_str += num.format(bi_d.perplexity(lines)) + ","
        		end_str += num.format(bi_d.perplexity(lines, bi_d.best_alpha)) + ","
        		end_str += num.format(tri_c.perplexity(lines)) + ","
        		end_str += num.format(tri_c.perplexity(lines, tri_c.best_alpha)) + ","
        		end_str += num.format(tri_d.perplexity(lines)) + ","
        		end_str += num.format(tri_d.perplexity(lines, tri_d.best_alpha)) + ","
        		end_str += num.format(back_c.perplexity(lines)) + ","
        		end_str += num.format(back_d.perplexity(lines)) + ","
				
        		print(end_str)


		

	


if __name__ == "__main__":
    main()



# Authors of the documents
# file author
# ------- --------
# c*.txt Wilkie Collins
# d*.txt  Charles Dickens
# t01.txt  Charles Dickens
# t02.txt  Charles Dickens
# t03.txt  Wilkie Collins 
# t04.txt  Charles Dickens
# t05.txt  Wilkie Collins
# t06.txt  Mark Twain
# t07.txt  Lewis Carroll
# t08.txt  Jane Austen