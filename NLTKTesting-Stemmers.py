import nltk
from nltk.stem import PorterStemmer

#            Testing stemmers
# Testing Porter Stemmer
stemmer = PorterStemmer()

plurals = ['caresses', 'flies', 'dies', 'mules', 'defined', 'died', 'dies', 'agrees', 'agrees', 'sizing', 'irrational', 'comical', 'fertilizer']

print("Porter Stemmer Test: \n")
for word in plurals:
    print(f"{word} >>> {stemmer.stem(word)}")
print(" \n")

#Testing Snowball Stemmer
from nltk.stem.snowball import SnowballStemmer
SnowballStemmer.languages

sn_stemmer = SnowballStemmer("english")

#Comparing both stemmers

snword = sn_stemmer.stem("generously")
portword = stemmer.stem("generously")

print("Snowball Stemmer: \n", snword)
print("Porter Stemmer: \n", portword)