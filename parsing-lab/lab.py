import nltk
import pprint


def single_input():
    text = input("Please input some text for parsing: ")
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)
    print(result)
    result.draw()


def multi_sentence_input(path):
    sum_dict = {}
    text = open(path, 'r').read()
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    grammar = "NP: {<DT><NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged)

    for tree in cp.parse(result).subtrees():
        if (tree[0][0] == "the"):
            if (tree[1][0] in sum_dict):
                sum_dict[tree[1][0]] = sum_dict[tree[1][0]] + 1
            else:
                sum_dict[tree[1][0]] = 1
    sorted_by_value = sorted(sum_dict.items(), key=lambda kv: kv[1])

    print(sorted_by_value)


multi_sentence_input("turtles.txt")
