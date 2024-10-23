import nltk
import regex

text = """Monticello wasn't designated as UNESCO World Heritage Site until 1987"""

#           Testing tokenizers to compare which one is better

#Testing Regex
regextext = regex.split("[\s\.\,]", text)

print("Regex Test: \n", regextext)

#Testing nltk.word (BETTER OPTION)
wordtext = nltk.word_tokenize(text)

print("NLTK_WORD Test: \n", wordtext)
