from nltk.metrics import *
from pickle import load
import sklearn.metrics as metrics
from pickle import dump
import nltk
import os
brown_tagged = nltk.corpus.brown.tagged_sents(
    categories='news')

# Split training and test data.
length = len(brown_tagged)
training = brown_tagged[int(length/5):]
test = brown_tagged[:100]

already_trained = os.path.isfile('unigram_tagger.pkl') and os.path.isfile(
    'tnt_tagger.pkl') and os.path.isfile('perceptron_tagger.pkl')

if (not already_trained):
    # Training
    unigram = nltk.UnigramTagger(training)
    print("Trained Unigram.")

    tnt = nltk.TnT()
    tnt.train(training)
    print("Trained TnT.")

    perceptron = nltk.PerceptronTagger()
    perceptron.train(training)
    print("Trained Perceptron.")

    # CRF skipped due to lack of time to train.
    # crf = nltk.CRFTagger()
    # crf.train(training, 'model.crf.tagger')
    # print("Trained CRF.")

    # Dump trained models as files for later use.
    unigram_output = open('unigram_tagger.pkl', 'wb')
    tnt_output = open('tnt_tagger.pkl', 'wb')
    perceptron_output = open('perceptron_tagger.pkl', 'wb')

    dump(unigram, unigram_output, -1)
    unigram_output.close()
    dump(tnt, tnt_output, -1)
    tnt_output.close()
    dump(perceptron, perceptron_output, -1)
    perceptron_output.close()

    print("Trained and saved models.")


for name in ['unigram_tagger', 'tnt_tagger', 'perceptron_tagger']:
    input = open(f'{name}.pkl', 'rb')
    tagger = load(input)
    input.close()

    print(f"------------ {name}:")

    test_sentences = [list(map(lambda pair: pair[0], sentence))
                      for sentence in test]
    reference_sentences = [list(map(lambda pair: pair[1], sentence))
                           for sentence in test]

    flattened_test_sentences = []
    flattened_reference_sentences = []
    [flattened_test_sentences.extend(sentence) for sentence in test_sentences]
    [flattened_reference_sentences.extend(
        sentence) for sentence in reference_sentences]

    result_tokens = tagger.tag(flattened_test_sentences)
    no_none = list(
        map(lambda pair: 'UNKNOWN' if pair[1] is None else pair[1], result_tokens))

    reference_sentences = flattened_reference_sentences
    result_tokens = no_none
    cm = ConfusionMatrix(reference_sentences, no_none)

    # print(cm)
    print("Precision: ", precision(set(reference_sentences), set(result_tokens)))
    print("Recall: ", recall(set(reference_sentences), set(result_tokens)))
    print("F measure: ", f_measure(set(reference_sentences), set(result_tokens)))
    print("Accuracy: ", accuracy(reference_sentences, result_tokens))

    # print(metrics.classification_report(reference_sentences, result_tokens))
