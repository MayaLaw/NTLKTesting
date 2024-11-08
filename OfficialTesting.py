import nltk

text = """On March 9, 2024, at 1331 eastern standard time, Southwest Airlines flight 4318 encountered 
turbulence during initial descent into Baltimore Washington International airport (BWI), 
Baltimore, Maryland."""

text2 = """On March 9, 2024, at 1331 eastern standard time, Southwest Airlines flight 4318 encountered 
turbulence during initial descent into Baltimore Washington International airport (BWI), 
Baltimore, Maryland. The flight was a regularly scheduled passenger flight from Northwest 
Florida Beaches International Airport (ECP), Panama City, Florida to BWI."""

text3 = """On March 9, 2024, at 1331 eastern standard time, Southwest Airlines flight 4318 encountered 
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
airplane at the gate."""

wordtext = nltk.word_tokenize(text)
wordtext2 = nltk.word_tokenize(text2)
wordtext3 = nltk.word_tokenize(text3)

print("NLTK_WORD Test 1: \n", wordtext, "\n\n")
print("NLTK_WORD Test 2: \n", wordtext2, "\n\n")
print("NLTK_WORD Test 3: \n", wordtext3, "\n\n")