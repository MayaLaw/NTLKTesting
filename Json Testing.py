import os
import json
import re
import nltk
import squarify
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Initializing tools
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# File paths
folder_path = r"C:\Users\maya_\Documents\GitHub\NTLKTesting\ExtractedTxt"
event_list_path = os.path.join(folder_path, "defining_events_list.txt")
summary_output = os.path.join(folder_path, "summary.json")

# Regex patterns
location_pattern = re.compile(r'Location:\s*([\s\S]*?)\s*(?=(?:Accident|Incident) Number:)')
accnum_pattern = re.compile(r'(?:Accident|Incident) Number:\s*([^\s\n]+)')
datetime_pattern = re.compile(r'Date & Time:\s*([^\n]+)')
aircraft_pattern = re.compile(r'Aircraft:\s*([^\n]+)')
registration_pattern = re.compile(r'Registration:\s*([^\n]+)')
flight_conducted_pattern = re.compile(r'Flight Conducted Under:\s*([^\n]+)')
damage_pattern = re.compile(r'Aircraft Damage:\s*([^\n]+)')
event_pattern = re.compile(r'Defining Event:\s*([^\n]+)')
injuries_pattern = re.compile(r'Injuries:\s*([^\n]+)')
analysis_pattern = re.compile(r'Analysis\s*([\s\S]+)', re.IGNORECASE)

# Preprocessing helpers
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

def normalize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text):
    return normalize_text(clean_text(text))

def extract_field(pattern, text):
    match = pattern.search(text)
    return match.group(1).strip() if match else "Not Found"

def report_missing_fields(data_dict):
    missing = [field for field, value in data_dict.items() if value == "Not Found"]
    if missing:
        print(f"Missing fields in {data_dict['Filename']}: {', '.join(missing)}")

Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-large-v2")

print(os.getenv("OPENAI_API_KEY"))


Settings.llm = Ollama(
    model="qwen2.5:1.5b-instruct",
    base_url="http://localhost:11434",
    request_timeout=120.0,
    additional_kwargs={

        "num_ctx": 1024,   
        "num_predict": 64,    # labels are short; trim generation budget (will play around with)
        # make labels stable run-to-run
        "temperature": 0.2,   # (will try 0.0)
        "top_p": 0.9,
        "seed": 7
    }
)
# Text file processing and JSON creation
def process_text_files():
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            analysis = extract_field(analysis_pattern, content)
            processed_analysis = preprocess_text(analysis) if analysis != "Not Found" else "Not Found"
            top_words = sorted(Counter(processed_analysis.split()).items(), key=lambda x: x[1], reverse=True)[:3] if processed_analysis != "Not Found" else []

            location_match = location_pattern.search(content)
            location = re.sub(r'\s+', ' ', location_match.group(1).replace("\n", " ")) if location_match else "Not Found"

            json_data = {
                "Filename": filename,
                "Location": location,
                "Accident Number": extract_field(accnum_pattern, content),
                "Date & Time": extract_field(datetime_pattern, content),
                "Aircraft": extract_field(aircraft_pattern, content),
                "Registration": extract_field(registration_pattern, content),
                "Flight Conducted Under": extract_field(flight_conducted_pattern, content),
                "Aircraft Damage": extract_field(damage_pattern, content),
                "Defining Event": extract_field(event_pattern, content),
                "Injuries": extract_field(injuries_pattern, content),
                "Analysis": {
                    "Text": processed_analysis,
                    "Top Words": top_words
                }
            }

            report_missing_fields(json_data)

            json_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".json")
            with open(json_path, "w", encoding="utf-8") as out_file:
                json.dump(json_data, out_file, indent=4)
            print(f"Updated {json_path}")

# Master summary JSON creation
def create_summary_json():
    summary = []
    for file in os.listdir(folder_path):
        if file.endswith(".json") and file != "summary.json":
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                try:
                    summary.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"Could not decode {file}")
    with open(summary_output, "w", encoding="utf-8") as out:
        json.dump(summary, out, indent=4)
    print(f"Created summary file: {summary_output}")

# Event extraction from JSON
def extract_raw_events(folder_path):
    event_counter = Counter()
    records = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    event = data.get("Defining Event", "Not Found")
                    aircraft = data.get("Aircraft", "Unknown")
                    if event != "Not Found":
                        event_counter[event] += 1
                        records.append((event, aircraft))
                except:
                    pass
    return event_counter, records

# Visualizations
def plot_event_distribution(event_counter):
    events, counts = zip(*event_counter.items())
    plt.figure(figsize=(10, 6))
    plt.barh(events, counts, color='mediumseagreen')
    plt.xlabel("Occurrences")
    plt.title("Defining Event Distribution")
    plt.tight_layout()
    plt.show()

def plot_event_pie(event_counter):
    labels = list(event_counter.keys())
    sizes = list(event_counter.values())
    explode = [0.05 if size == max(sizes) else 0 for size in sizes]
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode, startangle=140)
    plt.axis("equal")
    plt.title("Defining Events Pie Chart")
    plt.tight_layout()
    plt.show()

def plot_event_treemap(event_counter):
    labels = [f"{e}\n{c}" for e, c in event_counter.items()]
    sizes = list(event_counter.values())
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=sizes, label=labels, alpha=0.8)
    plt.axis("off")
    plt.title("Defining Events Treemap")
    plt.tight_layout()
    plt.show()

# Event clustering with llm
def label_clusters_llm(cluster_to_events, max_samples=6):
    """
    Uses LlamaIndex's configured LLM to produce short, human-friendly labels
    for each cluster. No TF-IDF involved.
    """
    labels = {}
    for cid, evs in cluster_to_events.items():
        if not evs:
            labels[cid] = "Misc"
            continue
        # prob should tweak the given prompt for different results
        samples = evs[:max_samples]
        prompt = (
            "You are naming a text cluster. "
            "Given the sample items below, return a concise 3–6 word, noun-phrase label "
            "with no punctuation or quotes. Focus on what these items have in common.\n\n"
            "Samples:\n- " + "\n- ".join(samples)
        )
        label = Settings.llm.complete(prompt).text.strip()
        labels[cid] = label if label else "Misc"
    return labels

def cluster_events(events, num_clusters=10):
    """
    - Embeds events with LlamaIndex's embed model (e5-large-v2 here).
    - Clusters with KMeans (sklearn).
    - Labels clusters with an LLM via LlamaIndex (no TF-IDF).
    - Writes grouped_defining_events.txt.
    """

    nodes = [TextNode(text=e) for e in events]


    texts = [n.get_content() for n in nodes]
    embeddings = Settings.embed_model.get_text_embedding_batch(texts)  # List[List[float]] maybve

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    clustered = defaultdict(list)
    for ev, lbl in zip(events, labels):
        clustered[int(lbl)].append(ev)

    cluster_names = label_clusters_llm(clustered, max_samples=6)

    with open("grouped_defining_events.txt", "w", encoding="utf-8") as out:
        for cid in sorted(clustered):
            name = cluster_names.get(cid, f"Cluster {cid + 1}")
            out.write(f"\n--- Cluster {cid + 1}: {name} ---\n")
            for ev in sorted(clustered[cid]):
                out.write(f"  • {ev}\n")

    print("Clustered events saved to grouped_defining_events.txt")

# MAIN PROGRAM
if __name__ == "__main__":
    process_text_files()
    create_summary_json()

    with open(event_list_path, "r", encoding="utf-8") as f:
        events = [line.strip() for line in f if line.strip()]


    cluster_events(events)

    event_counter, records = extract_raw_events(folder_path)
    plot_event_distribution(event_counter)
    plot_event_pie(event_counter)
    plot_event_treemap(event_counter)

# USED VISUALIZATIONS/OTHER METHODS
#
#
#
# def write_defining_events_list(folder_path, output_filename="defining_events_list.txt", grouped=False):
#     defining_events = set()

#     for file in os.listdir(folder_path):
#         if file.endswith(".json") and file != "summary.json":
#             file_path = os.path.join(folder_path, file)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 try:
#                     data = json.load(f)
#                     raw_event = data.get("Defining Event", "Not Found").strip()
#                     event = map_event_to_group(raw_event) if grouped else raw_event
#                     defining_events.add(event)
#                 except json.JSONDecodeError:
#                     print(f"Warning: Could not decode {file}")

#     output_path = os.path.join(folder_path, output_filename)
#     with open(output_path, "w", encoding="utf-8") as outfile:
#         for event in sorted(defining_events):
#             outfile.write(event + "\n")

#     print(f"\nCreated list of defining events: {output_filename}")

# write_defining_events_list(folder_path)

# def plot_event_aircraft_heatmap(folder_path):
#     records = []

#     for file in os.listdir(folder_path):
#         if file.endswith(".json") and file != "summary.json":
#             file_path = os.path.join(folder_path, file)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 try:
#                     data = json.load(f)
#                     aircraft = data.get("Aircraft", "Unknown")
#                     raw_event = data.get("Defining Event", "Not Found").strip()
#                     event_group = map_event_to_group(raw_event)
#                     records.append((event_group, aircraft))
#                 except json.JSONDecodeError:
#                     print(f"Warning: Could not decode {file}")

#     if not records:
#         print("No data available for heatmap.")
#         return

#     # Create DataFrame
#     df = pd.DataFrame(records, columns=["Event Group", "Aircraft"])
#     heatmap_data = df.pivot_table(index="Event Group", columns="Aircraft", aggfunc=len, fill_value=0)

#     # Plot heatmap
#     plt.figure(figsize=(14, 8))
#     sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
#     plt.title("Heatmap: Defining Events per Aircraft Type")
#     plt.ylabel("Event Group")
#     plt.xlabel("Aircraft Type")
#     plt.tight_layout()
#     plt.show()

# plot_event_aircraft_heatmap(folder_path)

# def plot_top_words_across_json(folder_path, top_n=10):
#     word_counter = Counter()

#     for file in os.listdir(folder_path):
#         if file.endswith(".json") and file != "summary.json":
#             file_path = os.path.join(folder_path, file)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 try:
#                     data = json.load(f)
#                     if isinstance(data, dict) and "Analysis" in data:
#                         top_words = data["Analysis"].get("Top Words", [])
#                         for word, freq in top_words:
#                             word_counter[word] += freq
#                 except json.JSONDecodeError:
#                     print(f"Warning: Could not decode {file}")

#     # Get top N words
#     most_common = word_counter.most_common(top_n)
#     if most_common:
#         words, counts = zip(*most_common)
#         plt.figure(figsize=(8, 5))
#         plt.bar(words, counts, color='steelblue')
#         plt.title(f"Top {top_n} Most Frequent Words in All Analyses")
#         plt.xlabel("Words")
#         plt.ylabel("Frequency")
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No top words found to plot.")


# plot_top_words_across_json(folder_path)



# def plot_top_words_for_event(folder_path, defining_event="Loss of control on ground", top_n=10, exclude_words=None):
#     all_words = []
#     exclude_words = set(word.lower() for word in (exclude_words or []))  # Convert to lowercase set

#     for filename in os.listdir(folder_path):
#         if filename.endswith(".json") and filename != "summary.json":
#             file_path = os.path.join(folder_path, filename)

#             with open(file_path, "r", encoding="utf-8") as json_file:
#                 try:
#                     data = json.load(json_file)

#                     if data.get("Defining Event", "").strip().lower() == defining_event.lower():
#                         analysis_text = data.get("Analysis", {}).get("Text", "")
#                         words = analysis_text.split()
#                         all_words.extend(words)

#                 except json.JSONDecodeError:
#                     print(f"Warning: Skipped corrupted JSON: {filename}")
#                     continue
#
#
#     filtered_words = [word for word in all_words if word.lower() not in exclude_words]
#     word_counts = Counter(filtered_words)
#     common_words = word_counts.most_common(top_n)

#     if not common_words:
#         print(f"No matching analysis found for Defining Event: '{defining_event}'")
#         return

#     # Plot
#     words, counts = zip(*common_words)
#     plt.figure(figsize=(10, 6))
#     plt.bar(words, counts, color='skyblue')
#     plt.title(f"Top {top_n} Words in 'Analysis' for Event: {defining_event}")
#     plt.xlabel("Words")
#     plt.ylabel("Frequency")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# plot_top_words_for_event(
#     folder_path=r"C:\Users\maya_\Documents\GitHub\NTLKTesting\ExtractedTxt",
#     defining_event="Loss of control on ground",
#     top_n=15,
#     exclude_words=["aircraft", "pilot", "airplane"]
# )

# plot_top_words_for_event(folder_path, defining_event="Loss of control on ground", top_n=15)



# def search_word_frequency_by_event(folder_path, search_word):
#     word = search_word.lower()
#     event_counts = {}

#     for filename in os.listdir(folder_path):
#         if filename.endswith(".json") and filename != "summary.json":
#             file_path = os.path.join(folder_path, filename)

#             with open(file_path, "r", encoding="utf-8") as json_file:
#                 try:
#                     data = json.load(json_file)
#                     event = data.get("Defining Event", "Unknown").strip()

#                     analysis_text = data.get("Analysis", {}).get("Text", "").lower()
#                     word_count = analysis_text.split().count(word)

#                     if word_count > 0:
#                         if event not in event_counts:
#                             event_counts[event] = 0
#                         event_counts[event] += word_count

#                 except json.JSONDecodeError:
#                     print(f"Skipped corrupt JSON: {filename}")
#                     continue

#     if not event_counts:
#         print(f"No occurrences of the word '{search_word}' found.")
#         return

#     # Plot results
#     events = list(event_counts.keys())
#     counts = list(event_counts.values())

#     plt.figure(figsize=(12, 6))
#     plt.bar(events, counts, color='salmon')
#     plt.xlabel("Defining Event")
#     plt.ylabel(f"Occurrences of '{search_word}'")
#     plt.title(f"Frequency of '{search_word}' in Analysis by Defining Event")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()

# search_word_frequency_by_event(
#     folder_path=r"C:\Users\maya_\Documents\GitHub\NTLKTesting\ExtractedTxt",
#     search_word="turbulence"
# ) 
