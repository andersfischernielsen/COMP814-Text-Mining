import nltk


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
    grammar = "NP: {<DT><NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    print(result)
    # TODO: Extract and count the definite nouns and output them in a sorted ascending order (by count).
    result.draw()


multi_sentence_input("turtles.txt")
