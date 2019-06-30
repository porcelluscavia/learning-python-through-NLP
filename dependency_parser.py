#!/usr/bin/env python
# -*- coding: utf-8 -*-



import conllu
import copy
import transition
import numpy as np
from sklearn import datasets, linear_model
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.feature_extraction import DictVectorizer
from keras.layers import Dense, Embedding, Dropout, Activation
from keras.layers import SimpleRNN
# from keras.layers import LSTM
from keras.layers import GRU
from keras.models import Sequential
import sys
from collections import namedtuple


#from Daniel De Kok's implementation "train.py" in the second Deep Learning Assignment from his SS 2017 Class
class Numberer:
    def __init__(self):
        self.v2n = dict()
        self.n2v = list()
        self.start_idx = 1

    def number(self, value, add_if_absent=True):
        n = self.v2n.get(value)

        if n is None:
            if add_if_absent:
                n = len(self.n2v) + self.start_idx
                self.v2n[value] = n
                self.n2v.append(value)
            else:   
                return 0
        return n

    def value(self, number):
        return self.n2v[number - 1]

    def max_number(self):
        return len(self.n2v) + 1

#straight from the transition.py class
def test_parser(parsed, gold_standard):

    Token = namedtuple(
        'Token', "tid, form lemma pos xpos feats head deprel deps misc children")

    def read_conllu(fname=None, fp=sys.stdin, mark_children=False):
        if fname is not None:
            fp = open(fname, 'r', encoding='utf-8')

        treebank = []
        sent_start = True
        for line in fp:
            if line.startswith('#'):
                continue
            line = line.strip()

            if len(line) == 0 and not sent_start:
                if mark_children:
                    for tok in sent:
                        if tok.head is not None:
                            hd = sent[tok.head]
                            hd.children.append(tok.tid)
                treebank.append(sent)
                sent_start = True
                continue

            if mark_children: chi = []
            else: chi = None

            if sent_start:
                sent = [Token(
                    0, "_", "root", "_", "_", "_", None, "_", "_", "_", chi)]
                sent_start = False

            (tid, form, lemma, pos, xpos, feats, head, deprel, deps, misc) = \
                    line.strip().split('\t')
            if "-" in tid:
                continue
            sent.append(Token(int(tid),
                              form,
                              lemma,
                              pos,
                              xpos,
                              feats,
                              int(head),
                              deprel.split(":")[0],
                              deps,
                              misc,
                              chi))
        return treebank

    conllu.save(parsed, "..\\parsed")
    conllu.save(gold_standard, "..\\gold_standard")
    out = read_conllu("..\\parsed")
    gs  = read_conllu("..\\gold_standard")
    # out = read_conllu(parsed)
    # gs  = read_conllu(gold_standard)

    if len(out) != len(gs):
        print("The number of sentences differ!")
        sys.exit(-1)


    arcs_lmatch_w = 0
    arcs_umatch_w = 0
    arcs_total = 0
    for i in range(len(out)):
        sent_out = out[i]
        sent_gs = gs[i]

        if len(sent_out) != len(sent_gs):
            print("The number of words differ in sentence {}".format(i))
            sys.exit(-1)

        arcs_lmatch_sent = 0
        arcs_umatch_sent = 0
        ntokens = len(sent_out) - 1
        for j in range(1,len(sent_out)):
            if sent_out[j].head == sent_gs[j].head:
                arcs_umatch_sent += 1
                if sent_out[j].deprel == sent_gs[j].deprel:
                    arcs_lmatch_sent += 1
        arcs_total += ntokens
        arcs_lmatch_w += arcs_lmatch_sent
        arcs_umatch_w += arcs_umatch_sent


    print("UAS: {:.2f}\tLAS: {:.2f}".format(
        100 * arcs_umatch_w / arcs_total,
        100 * arcs_lmatch_w / arcs_total)
    )


def format_as_sents(conllu_path):
    sents = list(conllu.load(conllu_path))
    # conllu.save(sents, temp)
    # assert sents == list(conllu.load(temp))
    return sents
    # is it more efficient to write them tos a file or save all as a list?
    # pickle the list?
    # return sents and use it directly?


def get_things(sents,proj=False, lazy=True, verbose=True):
    everything = []

    for s in sents:
        o = transition.Oracle(s, proj, lazy)
        c = transition.Config(s)

        feat_list = []
        tup_list = []
       
        while not c.is_terminal():
            act, arg = o.predict(c)
            # print(o.predict(c))
            # print(c.stack_nth(1))
            # print(c.stack)
            # print(c.input)
            feat_list = features_to_list(s,c)
            tup_list.append("{}\t{}".format(act, arg))
            tup_list.append(feat_list)
            #create two-tuples as specified by assignment
            two_tup = tuple(tup_list)
            everything.append(two_tup)
            feat_list = []
            tup_list = []
            if verbose: print("{}\t{}".format(act, arg))
            assert c.doable(act)
            getattr(c, act)(arg)

    return everything

def features_to_list(s,c):
    feat_list = []

    # add features and pos tags from stack
    for x in range(3, 0, -1):
            try:                    
                if c.stack_nth(x)== 0:
                    feat_list.append("ROOT")
                    feat_list.append("root")
                else:
                    #do we need stacknth
                    feat_list.append(s.upostag[c.stack_nth(x)])
                    feat_list.append(s.form[c.stack_nth(x)])

            except IndexError:
                feat_list.append("")    
                feat_list.append("")

    # add features and pos tags from buffer
    for y in range(1, 4, 1):
        try:    
            feat_list.append(s.upostag[c.input_nth(y)])
            feat_list.append(s.form[c.input_nth(y)])

        except IndexError:
            feat_list.append("")
            feat_list.append("")
    return feat_list


def train_classifier(formpos_num, labels_num, train_list,val_list):
    #ADD VALIDATION ST
    labels_train_numbered = []
    feat_train_numbered = []
    all_feats_numbered_tr =[]
    labels_test_numbered = []
    feat_test_numbered = []
    all_feats_numbered_tst =[]


    for two_tup in train_list:
        label = two_tup[0]
        numbed_label = labels_num.number(label, True)
        labels_train_numbered.append(numbed_label)

        for form_or_pos in two_tup[1]:
            numbed_feature = formpos_num.number(form_or_pos, True)
            feat_train_numbered.append(numbed_feature)

        all_feats_numbered_tr.append(feat_train_numbered)
        feat_train_numbered = []

    # assert len(labels_train_numbered) == len(all_feats_numbered_tr) 

    for two_tup in val_list:
        label = two_tup[0]
        numbed_label = labels_num.number(label, False)
        labels_test_numbered.append(numbed_label)

        for form_or_pos in two_tup[1]:
            numbed_feature = formpos_num.number(form_or_pos, False)
            feat_test_numbered.append(numbed_feature)

        all_feats_numbered_tst.append(feat_test_numbered)
        feat_test_numbered = [] 

    all_feats_numbered_tst = np.asarray(all_feats_numbered_tst)
    labels_test_numbered = np.asarray(labels_test_numbered)
    all_feats_numbered_tr = np.asarray(all_feats_numbered_tr)
    labels_train_numbered = np.asarray(labels_train_numbered)


    #I started out by using dictVectorizer, numbering arrays and putting into linear model, but got very poor results
    #I learned more about Keras from: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    #I asked Ryan Callihan and Madeesh Kannan for advice on using Keras embeddings ~Sam
    model = Sequential()
    
    embedding_size = 100
    model.add(Embedding(input_dim = formpos_num.max_number(),output_dim = embedding_size,
                        input_length = all_feats_numbered_tr.shape[1]))
    
    # model.add(LSTM(embedding_size))
    # model.add(SimpleRNN(embedding_size, dropout = 0.2, recurrent_dropout = 0.1))
    model.add(GRU(embedding_size, recurrent_dropout = 0.1, dropout = 0.2))

    model.add(Dense(labels_num.max_number(), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(all_feats_numbered_tr, labels_train_numbered, epochs= 10, batch_size = 64, verbose=1)

    
    score = model.evaluate(all_feats_numbered_tst, labels_test_numbered)
    print("Accuracy: %.2f%%" % (score[1]*100))

    return model


def parse_sent(sent, classifier, features_num,labels_num):
    #needs to predict labels for a list of features that have been numbered
    #during EVERY tansition step
    c = transition.Config(sent)
    feat_list = []
    bigger_list = []
    feat_list_numd = []

    feat_list = features_to_list(sent, c)

    for feature in feat_list:
        feat_list_numd.append(features_num.number(feature, False))

    #I need to do this in order to get the right shape
    bigger_list.append(feat_list_numd)


    feat_list_numd = np.asarray(bigger_list)
 
  

    choices_sorted = []
    predictions = classifier.predict(feat_list_numd)

   #predict probabilities of act/arg pairs for every step of the parse
    for preds in predictions:
        preds_list = np.argsort(preds)[::-1]
        poss_acts = []
        for pred_step in preds_list:
            act_arg_pair = labels_num.value(pred_step)
            poss_acts.append(act_arg_pair)

        choices_sorted.append(poss_acts)

    while not c.is_terminal():
        for step in choices_sorted:
            for poss_choice in step:
                act,arg = poss_choice.split("\t")
                if not c.doable(act):
                    continue
                else:
                    getattr(c, act)(arg)
                    #found a working choice, get out of this step
                    break
    #is this what we want to return? I guess          
    return c.finish()


if '__main__' == __name__:
    import sys

    # do we need the temp path?
    try:
        train_file, val_file = sys.argv[1:]
    except ValueError:
        sys.exit("usage: {} train_file val_file".format(sys.argv[0]))

    labels_num = Numberer()
    features_num = Numberer()

    train_sents = format_as_sents(train_file)
    val_sents = format_as_sents(val_file)
    labels_and_features_tr = get_things(train_sents, verbose=False)
    labels_and_features_val = get_things(val_sents, verbose=False)
    model = train_classifier(features_num,labels_num,labels_and_features_tr,labels_and_features_val)

    #parse every val sent, compare with gold standard (output from oracle) for ALL the sents
    parses = []
    for sent in val_sents:
        parses.append(parse_sent(sent,model,features_num, labels_num))

    test_parser(parses, val_sents)
    print("yes!")

'''
a GRU was found to work better than the LSTM or SimpleRNN
The laptop was getting very hot when training on the full set for a long time, 
so results were recorded after 1 epoch:
Accuracy: 84.90% UAS: 6.78  LAS: 0.20
On smaller training and validation sets(about 1/10 of each file), the results for 10 epochs were:
Accuracy: 87.10% UAS: 6.93 LAS: 0.43
'''

   