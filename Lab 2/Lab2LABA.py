import nltk
from nltk.corpus import brown

# Use Brown corpus, split 80/20
print("\nPRE-PROCESSING IN PROGRESS")

brown_news_tagged = brown.tagged_sents(categories='news', tagset='universal')
train_len = int(len(brown_news_tagged)*0.8)
training_data = brown_news_tagged[:train_len]
test_data = brown_news_tagged[len(training_data):]

print(f"Training data size: {len(training_data)}")
print(f"Validation data size: {len(test_data)}")
print(f"Expected size: {len(training_data)+len(test_data)}, actual size: {len(brown_news_tagged)}")

# Instantiate and train the following taggers: Unigram; TnT; Perceptron; CRF
print("\nTRAINING MODELS")

unigram = nltk.UnigramTagger(training_data)
print("Unigram Tagger trained")
TnT = nltk.TnT()
TnT.train(training_data)
print("TnT Tagger trained")
perceptron = nltk.PerceptronTagger(training_data)
print("Perceptron Tagger trained")
#CRF = nltk.CRFTagger()
#CRF.train(training_data, "model.crf.tagger")
#print("CRF Tagger trained")

# Save the trained taggers (in a LABA Taggers Map), overwrite existing
import pickle
from pickle import dump
print("\nSAVING MODELS")

ugOutput = open("Unigram.pkl",'wb')
dump(unigram, ugOutput, -1)
ugOutput.close()
print("Trained Unigram Tagger Saved")

TnTOutput = open("TnT.pkl",'wb')
dump(TnT, TnTOutput, -1)
TnTOutput.close()
print("Trained TnT Tagger Saved")

perceptronOutput = open("Perceptron.pkl",'wb')
dump(perceptron, perceptronOutput, -1)
perceptronOutput.close()
print("Trained Perceptron Tagger Saved")

# THERE IS AN ISSUE IN THE CRF LIBRARY THAT MAKES IT UNABLE TO SAVE MODELS USING PICKLE
#CRFOutput = open("CRF.pkl",'wb')
#dump(CRF, CRFOutput, -1)
#CRFOutput.close()
#CRF = None
#print("Trained CRF Tagger Saved")

# Retrieve the pickle files of the trained models
print("\nLOADING MODELS")
f1 = open('Unigram.pkl', 'rb')
ldd_unigram = pickle.load(f1)
f1.close()
print("Trained Unigram Tagger Loaded")

f2 = open('TnT.pkl', 'rb')
ldd_TnT = pickle.load(f2)
f2.close()
print("Trained TnT Tagger Loaded")

f3 = open('Perceptron.pkl', 'rb')
ldd_perceptron = pickle.load(f3)
f3.close()
print("Trained Perceptron Tagger Loaded")

#f4 = open('CRF.pkl', 'rb')
#ldd_CRF = pickle.load(f4)
#f4.close()
#print("Trained CRF Tagger Loaded")

# Test the loaded models on test data
print("\nTESTING LOADED MODELS")
print(f"Loaded Unigram Tagger evaluation results: {ldd_unigram.evaluate(test_data)}")
print(f"Loaded TnT Tagger evaluation results: {ldd_TnT.evaluate(test_data)}")
print(f"Loaded Perceptron Tagger evaluation results: {ldd_perceptron.evaluate(test_data)}")
#print(f"Loaded CRF Tagger evaluation results: {ldd_CRF.evaluate(test_data)}")

# Tabulate and calculate accuracies, choose best one based on F1 value
from nltk.metrics import *
print("\nCALCULATING RESULTS")

def explode(list_of_lists): #List of lists to list
    return [item for sublist in list_of_lists for item in sublist]
def apply_tagger(tagger, corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]
def flatten(l):
    return list(map(lambda tuple: 'UNKNOWN' if tuple[1] is None else tuple[1], l))

ref = explode(brown_news_tagged) # Explode list
flat_ref = list(map(lambda tuple: tuple[1], ref))

unigram_tagged = explode(apply_tagger(ldd_unigram, brown_news_tagged))
print("###### Unigram Results ######")
print("Precision: ", precision(set(ref),set(unigram_tagged)))
print("Recall: ",    recall(set(ref),set(unigram_tagged)))
print("F measure: ", f_measure(set(ref),set(unigram_tagged)))
unigram_tagged = flatten(unigram_tagged)
print(ConfusionMatrix(flat_ref,unigram_tagged))

TnT_tagged = explode(apply_tagger(ldd_TnT, brown_news_tagged))
print("###### TnT Results ######")
print("Precision: ", precision(set(ref),set(TnT_tagged)))
print("Recall: ",    recall(set(ref),set(TnT_tagged)))
print("F measure: ", f_measure(set(ref),set(TnT_tagged)))
TnT_tagged = flatten(TnT_tagged)
print(ConfusionMatrix(flat_ref,TnT_tagged))

perceptron_tagged = explode(apply_tagger(ldd_perceptron, brown_news_tagged))
print("###### Perceptron Results ######")
print("Precision: ", precision(set(ref),set(perceptron_tagged)))
print("Recall: ",    recall(set(ref),set(perceptron_tagged)))
print("F measure: ", f_measure(set(ref),set(perceptron_tagged)))
perceptron_tagged = flatten(perceptron_tagged)
print(ConfusionMatrix(flat_ref,perceptron_tagged))

# Use the best tagger to tag 10 downloaded articles from 10 different news cites on one topic, select 3 nouns that best select the topic

# Make a table of comparisons