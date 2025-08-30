from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

text = """On March 9, 2024, at 1331 eastern standard time, Southwest Airlines flight 4318 encountered 
turbulence during initial descent into Baltimore Washington International airport (BWI), 
Baltimore, Maryland. The flight was a regularly scheduled passenger flight from Northwest 
Florida Beaches International Airport (ECP), Panama City, Florida to BWI. As a result of the 
turbulence, one flight attendant sustained serious injuries. The airplane was not damaged, and 
the flight continued to BWI without further incident. 
The flight deck crew reported that they were aware of, and had received, pilot reports of 
turbulence at lower altitudes on arrival into BWI. They had coordinated with the cabin crew 
during their preflight briefing about securing the cabin early and reiterated this again (as well as 
asking them to take their seats afterward) just prior to starting a standard terminal arrival (the 
RAVNN SIX) into BWI. As they began the arrival, they learned of a pilot report of severe turbulence 
at flight level (FL) 250 from an airplane on arrival into nearby Ronald Reagan Washington 
National Airport (DCA), Arlington, Virginia. 
While the cabin crew were securing the cabin and galleys, the airplane encountered severe 
turbulence as it descended through FL270. The captain immediately made a public address 
announcement to the cabin for the flight attendants to take their seats. Flight attendants B and 
C were “tossed around….sent into the air” before landing on the floor of the aft galley. Flight 
attendant C sustained a hairline fracture to the left arm. A non-revenue Southwest Airlines flight 
attendant who was seated in the rear of the airplane rendered assistance to flight attendant C. 
The captain declared a medical emergency and received an expedited approach into BWI. The 
first officer coordinated with airline operations and arranged for medical personnel to meet the 
airplane at the gate. A  post-accident  review  of  weather  records  revealed  that  there  were  four  pilot  reports  of 
moderate to severe turbulence from FL160 to FL250 in an area to the southwest of the accident 
location. A high-resolution rapid refresh numerical model computed for the time and location of 
the  accident  revealed  that  conditions  were  conducive  for  moderate  clear  air  turbulence  from 
FL250 to FL280. Infrared and visible satellite imagery depicted a transverse wave cloud pattern 
(often observed in turbulent conditions) over the accident area. The cloud temperatures were 
consistent with cloud tops near 33,000 ft above mean sea level. 
A graphical Airmen’s Meteorological (G-AIRMET) information Tango (turbulence), issued by the 
National Weather Service, valid at the time of the accident advised of occasional moderate 
turbulence between FL180 and FL380 for much of the mid-Atlantic and northeast states. 
Graphic turbulence guidance products predicted areas of moderate turbulence over central 
Virginia moving northwest between 1300 and 1400 eastern standard time, at FL240."""

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation.
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Split into words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)  # Rejoin the words

def normalize_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text)
    return text


def top_3_most_used_words_with_context(text: str) -> str:

    words = preprocess_text(text).split()

    # Handle empty input
    if not words:
        return "No words found."

    # Count the frequency of each word
    word_counts = Counter(words)

    # Find the top 3 most common words
    most_common_words = word_counts.most_common(3)

    # Store context words for each common word
    context_words = {word: {'before': [], 'after': []} for word, _ in most_common_words}

    # Populate context for each common word
    for i, word in enumerate(words):
        if word in context_words:
            if i > 0:
                context_words[word]['before'].append(words[i - 1])
            if i < len(words) - 1:
                context_words[word]['after'].append(words[i + 1])

    result = "Top 3 most used words with context:\n"
    for word, count in most_common_words:
        before_counts = Counter(context_words[word]['before']).most_common(1)
        after_counts = Counter(context_words[word]['after']).most_common(1)
        before_word = before_counts[0][0] if before_counts else "None"
        after_word = after_counts[0][0] if after_counts else "None"
        result += (f"'{word}' with {count} occurrence(s)\n"
                   f"  Most common word before: '{before_word}'\n"
                   f"  Most common word after: '{after_word}'\n")

    return result.strip()

def most_common_phrases(text: str, phrase_length: int = 2, top_n: int = 5) -> str:
   
    words = preprocess_text(text).split()

    # Handle empty input or insufficient words for phrases
    if not words or len(words) < phrase_length:
        return "No phrases found."

    # Extract phrases
    phrases = [' '.join(words[i:i + phrase_length]) for i in range(len(words) - phrase_length + 1)]

    # Count the frequency of each phrase
    phrase_counts = Counter(phrases)

    # Find the most common phrases
    most_common_phrases = phrase_counts.most_common(top_n)

    result = f"Top {top_n} most common phrases (length {phrase_length}):\n"
    for phrase, count in most_common_phrases:
        result += f"'{phrase}' with {count} occurrence(s)\n"

    return result.strip()



print(preprocess_text(text) + "\n")

print(top_3_most_used_words_with_context(text) + "\n")

print(most_common_phrases(text, phrase_length=2, top_n=3) + "\n")
