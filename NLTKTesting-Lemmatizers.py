import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ['caresses', 'flies', 'dies', 'mules', 'defined', 'died', 'was', 'agrees', 'agrees', 'sizing', 'irrational', 'comical', 'fertilizer']

print("Word Net Lemmatizer: \n")
for word in words:
    print(f"{word} >>> {lemmatizer.lemmatize(word)}")
