import nltk


def single_input():
    text = input("Please input some text for parsing: ")
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    print(result)
    result.draw()


def multi_sentence_input():
    path = input("Please input path to text file for parsing: ")
    text = open(path, 'r')
    # TODO: Count the definite nouns and output them in a sorted ascending order (by count).
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    print(result)
    result.draw()
