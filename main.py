import spacy
import pytextrank
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, help="Source text file")
args = parser.parse_args()

# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")

with open(args.source) as f:
    text = f.read()

doc = nlp(text)
# examine the top-ranked phrases in the document
for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    # print(phrase.chunks)
