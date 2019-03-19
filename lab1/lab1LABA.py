import nltk

def parseSingleSentence():
    sent = input("Insert input sentence here: ")
    taggedS = nltk.pos_tag(nltk.word_tokenize(sent)) 
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(taggedS)
    print(result)
    result.draw()

def parseText():
    #filePath = input("Insert input text here: ")
    #text = open(filePath, 'r').read()
    text = open("NZheraldText.txt", 'r').read()
    #print (text)
    taggedTxt = nltk.pos_tag(nltk.word_tokenize(text))
    #
    grammar = "NP: {<DT><NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(taggedTxt)
    nouns = list()
    for t in result:
        print(t)
        print(t[0][0])
        if t[0][0].lower() == 'the':
            nouns.append(t[0][2])
    print (len(nouns))
    print (nouns)
    sortedNouns = nouns.sort()
    print (sortedNouns)
    #print(result)
    #result.draw()

parseText()
#singleSentencePOS()