from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import nltk
import re

def read_file():
    file = open("data/data1.txt","r")
    data = file.read()
    file.close()
    return data

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = f" {current_chunk}"
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if continuous_chunk:
        named_entity = f" {current_chunk}"
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk

#txt = "Jacinda Ardern is the Prime Minister of New Zealand but Roenzo isn't."
#print (get_continuous_chunks(txt))

def main():
    txt = read_file()
    txt = txt.split("\n")

    results = {"persons":0, "locations":0}

    for i, val in enumerate(txt):
        for sent in nltk.sent_tokenize(val):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label') and (chunk.label() == 'PERSON' or chunk.label() == 'GPE'): 
                    NE = ' '.join(c[0] for c in chunk)

                    if chunk.label() == 'PERSON':
                        results["persons"] = results["persons"] + 1

                    if chunk.label() == 'GPE':
                        results["locations"] = results["locations"] + 1

                    #print(f"Found {NE} as {chunk.label()} in sentence no. {i}.")

    persons = results["persons"]
    locations = results["locations"]
    actual_persons = 4
    actual_locations = 26
    print(f"Found {persons} persons and {locations} locations in the given text.")
    print(f"Manually {actual_persons} persons and {actual_locations} were found in the same text.")
    
    #persons_recall = actual_persons/persons
    #locations_recall = actual_locations/locations
    #persons_precision = ???
    #locations_precision = ???
    #print(f"For PERSONS this yields a recall of {persons_recall} and precision of {persons_precision}.")
    #print(f"For LOCATIONS this yields a recall of {locations_recall} and precision of {locations_precision}.")

main()