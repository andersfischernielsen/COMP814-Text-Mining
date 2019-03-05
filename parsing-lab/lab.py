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
    text = input("Please input some multi sentence text for parsing: ")
    # TODO: Split long text into sentences, iterate over them and parse individually.
    # TODO: Extract definite nouns in the text from each sentence.
    # TODO: Count the definite nouns and output them in a sorted ascending order.
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    print(result)
    result.draw()
