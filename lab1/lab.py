import nltk
import pprint
from collections import Counter


def single_input():
    text = input("Please input some text for parsing: ")
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    print(result)
    result.draw()


def multi_sentence_input(path):
    text = open(path, 'r').read()
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = "NP: {<DT><JJ>?<NN|NNS|NNP>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)

    nouns = []
    for tree in cp.parse(result).subtrees():
        if (tree[0][0].lower() == "the"):
            matches = tree[1:]
            nouns.append(" ".join(list(map(lambda m: m[0], matches))))

    print(f"There are {len(nouns)} definite nouns in the text.")
    counted = Counter(nouns)
    print("They are: ")
    sorted_by_value = sorted(counted.items(), key=lambda kv: kv[1])
    print(sorted_by_value)


multi_sentence_input("turtles.txt")
