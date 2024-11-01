import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
plurals = plurals = ['caresses', 'flies', 'dies', 'mules', 'defined', 'died', 'dies', 'agrees', 'agrees', 'sizing', 'irrational', 'comical', 'fertilizer']

print("Word Net Lemmatizer: \n")
for word in plurals:
    print(f"{word} >>> {lemmatizer.lemmatize(word)}")
