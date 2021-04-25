import numpy as np
import networkx as nx
import pickle
import plotly.express as px

from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.dummy import DummyClassifier

# precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

from colorama import init
from colorama import Fore, Back, Style
from prettytable import PrettyTable
init()

# Set program to one of ['sox', 'svg', 'tiff', 'ttf', 'wav', 'xml']
program = 'wav'

## a class object for Kolmogorov equivalence classes
class Kclass:

    def __init__(self, members, birth, freq):
        self.members = members ## the members of the equivalence class
        self.birth = birth ## the first time the equivalence class appears
        self.freq = freq ## the number of events containing the equivalence class

        self.child = None
        self.parent = None ##
        self.status = 'new' ## status relative to the new events

    def add_child(self, c):
        self.child = c

    def add_parent(self, p):
        self.parent = p

    def size(self):
        return len(self.members)


# Load crashes
with open('./processed_callstacks/crashes_' + program + '.txt', 'rb') as f:
    crashes_pre = pickle.load(f)


# Arrays to store formatted crashes
crashes = []
crash_lines = []


# Read-in  crash sets from text file
term_dict = {'module' : [], 'function' : [], 'offset' : []}
for crash in crashes_pre:
    new = []
    for elem in crash:
        crash_lines.append(set(elem))
        term_dict['module'].append(elem[1])
        term_dict['function'].append(elem[0])
        term_dict['offset'].append(str(elem[2]))
        for item in elem:
            new.append(item)

    crashes.append(set(new))

# label data by which contains a function
function_labels = []
module_labels = []
offset_labels = []

functions = set(term_dict['function'])
modules = set(term_dict['module'])
offsets = set(term_dict['offset'])


# load the graph
G = nx.read_gpickle("./callstack_posets/" + program + "_depgraph")
G_nofunc = nx.read_gpickle("./callstack_posets/" + program + "_depgraph_nofunc")

## build a dictionary of all terms in the call-stack set
symbols = []
new_crashes = []
for crash in crashes:
    n = ''
    for elem in crash:
        n = n + str(elem) + ' '
        symbols.append(elem)
    new_crashes.append(n)

vocab = list(set(symbols))
symbols = set(symbols)


## Display number of crashes
print('\n', '-'*80, '\n')
print("Number of crashes = ", len(crashes))
print('\n', '-'*80, '\n')


## build a dictionary to store the frame-trace information
frame_traces = {}
frame_heights = {}

for symbol in symbols:
    frame_heights[symbol] = []
    frame_traces[symbol] = []

for crash in crashes_pre:
    for i in range(len(crash)):
        for sym in list(crash)[i]:
            frame_heights[sym].append(1-i/len(crash))

for i in range(len(crashes_pre)):
    for j in range(len(crashes_pre[i])):
        for elem in crashes_pre[i][j]:
            frame_traces[elem].append([i,j])



for sym in symbols:
    frame_heights[sym] = np.mean(frame_heights[sym])

print(frame_traces)


# store attributes of each full-data nodes
class_attributes = []
for n in G.nodes:

    attributes = [n.size(), n.freq]

    weighted_in = 0
    off_in = 0
    for elem in G.predecessors(n):
        weighted_in += n.freq / elem.freq
        off_in += elem.size()
    attributes.append(weighted_in)

    weighted_out = 0
    off_out = 0
    for elem in G.successors(n):
        weighted_out += elem.freq / n.freq
        off_out += elem.size()
    attributes.append(weighted_out)

    class_attributes.append(attributes)


# store attributes of each incomplete-data nodes
nofunc_class_attributes = []
for n in G_nofunc.nodes:

    attributes = [n.size(), n.freq]

    weighted_in = 0
    off_in = 0
    for elem in G_nofunc.predecessors(n):
        weighted_in += n.freq / elem.freq
        off_in += elem.size()
    attributes.append(weighted_in)

    weighted_out = 0
    off_out = 0
    for elem in G_nofunc.successors(n):
        weighted_out += elem.freq / n.freq
        off_out += elem.size()
    attributes.append(weighted_out)

    nofunc_class_attributes.append(attributes)


## Scale full-data class attributes
class_attributes = np.array(class_attributes)
scaler = MinMaxScaler()
class_attributes = scaler.fit_transform(class_attributes)

## Scale incomplete-data class attributes
nofunc_class_attributes = np.array(nofunc_class_attributes)[:,:4]
scaler = MinMaxScaler()
nofunc_class_attributes = scaler.fit_transform(nofunc_class_attributes)


## Label classes in full-data
for n in G.nodes:

    # label functions
    if len(functions.intersection(n.members)) == 0:
        function_labels.append(0)
    else:
        function_labels.append(1)

    # label modules
    if len(modules.intersection(n.members)) == 0:
        module_labels.append(0)
    else:
        module_labels.append(1)

    # label line numbers
    if len(offsets.intersection(n.members)) == 0:
        offset_labels.append(0)
    else:
        offset_labels.append(1)


## label nofunc equivalence classes
labels = []
for i in range(len(G_nofunc.nodes)):
    n = list(G_nofunc.nodes)[i]
    l = 0
    for m in G.nodes:
        if n.members.issubset(m.members):
            if len(functions.intersection(m.members)) != 0:
                l = 1

    labels.append(l)


## Initialise the logistic regression and load with libxml weights
clf = LogisticRegression(class_weight = 'balanced', max_iter =10000)
clf.fit(class_attributes, function_labels)
clf.coef_ = np.array([[5.18,1.05,-0.18,3.09]])  ### Note that these are the co-efs & intercept attained from the libxml data
clf.intercept_ = np.array([-0.61123205])
predictions = clf.predict(nofunc_class_attributes)

## Create a list of probabilities of whether a function is in each class
probs = clf.predict_proba(nofunc_class_attributes)
probs = probs[:,1]

## CAlculate precision and recall of the probabilities
precision, recall, _ = precision_recall_curve(labels, probs)
f1, model_auc = f1_score(labels, predictions), auc(recall, precision)
print("LogisticRegression Function Class Prediction: ")
print("F1 Score: ", f1)
print("AUC: ", model_auc)
print("SCORE ", clf.score(nofunc_class_attributes, labels))


## Run and evaluate dummy model
model = DummyClassifier(strategy='stratified')
model.fit(class_attributes, function_labels)
yhat = model.predict_proba(nofunc_class_attributes)
naive_probs = yhat[:, 1]

# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(labels, naive_probs)
null_f1 = f1_score(labels, naive_probs)
null_auc_score = auc(recall, precision)


## Print results of classification
print("Null Model Function Class Prediction: ")
print("Null Model Score: ", model.score(labels, naive_probs))
print("Null model F1 Score: ", null_f1)
print("Null model AUC: ", null_auc_score)

print("\n")
print("F1 Improvement: ", f1 - null_f1)
print("AUC Improvement: ", model_auc - null_auc_score, "\n")

print("Len probs: ", len(probs))
print("len G_nofunc", len(G_nofunc))
print("Coef: ", clf.coef_)


# build linenum-function associator
from random_word import RandomWords
r = RandomWords()
with open('/Users/admin/Desktop/mphil_backup/code3/tests/classify_nodes/nouns.pkl', 'rb') as f:
    words = pickle.load(f)
import random
random.shuffle(words)


# create arrays for actual and predicted function fts
predicted_fts = []
function_fts = []
for f in functions:
    function_fts.append(frame_traces[f])

# predict frame traces
def predict_fts(G_nofunc, frame_heights, frame_traces, p):

    print("\n")
    print("Running predict FTs with probability ", p)

    # create arrays for actual and predicted function fts
    predicted_fts = []


    for i in range(len(G_nofunc.nodes)):
        n = list(G_nofunc.nodes)[i]
        o = n.members.intersection(term_dict['offset'])
        m = n.members.intersection(term_dict['module'])

        # create a local dict of line number average frame heights
        offset_heights = {}
        for off in o:
            offset_heights[off] = frame_heights[off]

        # sort by average frame height
        sorted_o = [list(elem)[0] for elem in sorted(offset_heights.items(), key=lambda x: x[1], reverse=True)]

        if probs[i] >= p:
            if len(o) > 1:
                for elem in sorted_o[:-1]:
                    predicted_fts.append(frame_traces[elem])
            if len(o) == 1:
                    for elem in o:
                        if 0 in frame_traces[elem]:
                            predicted_fts.append(frame_traces[elem])
                        else:
                            predicted_fts.append(frame_traces[elem])
                    #predicted_fts.append(frame_traces[elem])

    return predicted_fts


# Run the predict fts algorithm
predicted_fts = predict_fts(G_nofunc, frame_heights, frame_traces, 0.5)


## ACCURACY ASSESSMENT

## create an array of true function frame traces
function_fts = []
for f in functions:
    function_fts.append(frame_traces[f])

def assess_accuracy(predicted_fts, function_fts):

    accurate = 0
    for i in range(len(predicted_fts)):
        if predicted_fts[i] in function_fts:
            accurate += 1

    precision = accurate/len(predicted_fts)
    recall = accurate/len(function_fts)

    ## print accuracy results
    print("Accurate Predictions: ", accurate)
    print("Total number of Function FTS: ", len(function_fts))
    print("Total Predictions: ", len(predicted_fts))
    print("Precision: ", precision)
    print("Recall: ", recall)

    return [precision,recall]


assess_accuracy(predicted_fts, function_fts)

## Scan different p-values
precisions = []
recalls = []
for p in np.arange(20)/20:

    predicted_fts = predict_fts(G_nofunc, frame_heights, frame_traces, p)

    x, y = assess_accuracy(predicted_fts, function_fts)

    precisions.append(x)
    recalls.append(y)

## Plot precision and recall against probability values
import plotly.graph_objects as go

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(20)/20, y=precisions,
                    mode='lines',
                    name='Precision'))
fig.add_trace(go.Scatter(x=np.arange(20)/20, y=recalls,
                    mode='lines',
                    name='Recall'))
fig.update_layout(
    title={
        'text': "Precision and Recall for " + program,
        'y':0.92,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title = "Probability Threshold")
fig.show()

fig.write_image("./" + program + "_precisionrecall.jpeg")
