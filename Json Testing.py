import os
import json
import re
import nltk
import squarify
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import unicodedata
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
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


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


PAREN_RE = re.compile(r"\s*\([^)]*\)")  # remove parentheticals like "(non-impact)"

SYNONYMS = {
    "controlled flight into terr/obj": "controlled flight into terrain",
    "cfit": "controlled flight into terrain",
    "birdstrike": "bird strike",
    "wildlife encounter (non-bird)": "wildlife encounter",
    "runway incursion veh/ac/person": "runway incursion",
    "collision with terr/obj": "collision with terrain/object",
    "aircraft wake turb encounter": "wake turbulence encounter",
    "vfr encounter with imc": "vfr into imc",
    "loss of engine power (total)": "engine power loss total",
    "loss of engine power (partial)": "engine power loss partial",
}

BUCKET_RULES = {
    "Wildlife hazards": [
        r"\bbird\b", r"\bwildlife\b", r"\banimal\b", r"\bdeer\b"
    ],
    "Weather / environment": [
        r"\bwind ?shear\b", r"\bwindshear\b", r"\bthunderstorm\b",
        r"\bturbulence\b", r"\bicing\b", r"\bweather\b", r"\bmicroburst\b"
    ],
    "Fuel system": [
        r"\bfuel\b", r"\bstarvation\b", r"\bexhaustion\b", r"\bcontamination\b"
    ],
    "Runway / landing": [
        r"\brunway\b", r"\bhard landing\b", r"\babnormal runway contact\b",
        r"\bundershoot\b", r"\bovershoot\b", r"\bexcursion\b", r"\bincursion\b",
        r"\btailstrike\b", r"\blanding gear\b"
    ],
    "Loss of control / aerodynamic": [
        r"\bloss of control\b", r"\bstall\b", r"\bspin\b", r"\binflight upset\b",
        r"\bloss of lift\b", r"\bloss of visual reference\b"
    ],
    "CFIT / collision": [
        r"\bcfit\b", r"\bcontrolled flight into terrain\b",
        r"\bcollision\b", r"\bmidair\b", r"\bterrain/object\b"
    ],
    "Systems / mechanical": [
        r"\bsys/comp\b", r"\bmalf\b", r"\bfailure\b", r"\belectrical\b",
        r"\bflight control\b", r"\bpowerplant\b", r"\buncontained engine failure\b",
        r"\bpart\b.*\bseparation\b", r"\bstructural failure\b"
    ],
    "Navigation / deviations": [
        r"\bnavigation\b", r"\bcourse deviation\b", r"\baltitude deviation\b", r"\bwrong surface\b"
    ],
    "Operations / ground / admin": [
        r"\bair traffic\b", r"\bloading\b", r"\bcabin safety\b",
        r"\bground handling\b", r"\bpreflight\b", r"\bdispatch\b",
        r"\blow altitude\b", r"\bmedical event\b", r"\bsecurity\b"
    ],
    "Misc / unknown": [
        r"\bmiscellaneous\b", r"\bunknown\b", r"\bother\b"
    ]
}

RULE_TO_REGEX = {k: [re.compile(pat) for pat in pats] for k, pats in BUCKET_RULES.items()}

def assign_bucket(evt: str) -> str | None:
    """Return the first matching high-level bucket or None."""
    low = evt.lower()
    for bucket, regs in RULE_TO_REGEX.items():
        if any(r.search(low) for r in regs):
            return bucket
    return None

def cluster_within(texts, label_prefix, max_k=10):
    # embeddings + k selection + KMeans (your tuned version)
    embs = Settings.embed_model.get_text_embedding_batch(texts)
    X = normalize(np.array(embs), norm="l2")
    # keep k sane vs size
    k_max = max(2, min(max_k, len(texts)-1))
    k_min = min(4, k_max)
    k = choose_k_by_silhouette(X, k_min=k_min, k_max=k_max)
    k = max(2, min(k, len(texts)-1))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    lbls = km.fit_predict(X)
    grouped = defaultdict(list)
    for s, l in zip(texts, lbls):
        grouped[int(l)].append(s)
    # name clusters with LLM
    names = label_clusters_llm(grouped, max_samples=8)
    # prefix label with the rule-bucket for clarity, but keep it short
    names = {cid: f"{label_prefix}: {names[cid]}" for cid in names}
    return grouped, names

def rule_first_cluster(events):
    # 1) apply rule buckets
    rule_bins = defaultdict(list)
    leftovers = []
    for e in events:
        b = assign_bucket(e)
        if b: rule_bins[b].append(e)
        else: leftovers.append(e)

    # 2) cluster within each rule bucket (skip tiny buckets)
    all_clusters, all_names = {}, {}
    next_id = 0
    for bucket_name, items in rule_bins.items():
        if len(items) < 3:
            # very small: treat as one cluster verbatim
            all_clusters[next_id] = sorted(items)
            all_names[next_id] = f"{bucket_name}"
            next_id += 1
        else:
            g, names = cluster_within(sorted(items), bucket_name)
            for cid in sorted(g):
                all_clusters[next_id] = sorted(g[cid])
                all_names[next_id] = names[cid]
                next_id += 1

    # 3) cluster leftovers together (if any)
    if len(leftovers) == 1:
        all_clusters[next_id] = leftovers
        all_names[next_id] = "Unassigned: single item"
    elif len(leftovers) >= 2:
        g, names = cluster_within(sorted(leftovers), "Unassigned")
        for cid in sorted(g):
            all_clusters[next_id] = sorted(g[cid])
            all_names[next_id] = names[cid]
            next_id += 1

    return all_clusters, all_names


def normalize_event_name(s: str) -> str:
    if not s or s.strip().lower() == "not found":
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    s = PAREN_RE.sub("", s)                    # drop trailing qualifiers in ()
    s = re.sub(r"\s+", " ", s).strip()
    low = s.lower()
    low = low.replace("terr/obj", "terrain/object").replace("ac/", "aircraft ")
    low = low.replace("veh/ac/person", "vehicle/aircraft/person")
    low = low.replace("imc", "instrument conditions")
    # apply synonym collapses
    rep = low
    for k, v in SYNONYMS.items():
        rep = rep.replace(k, v)
    # title-case *after* normalization for presentation
    return " ".join(w for w in rep.split() if w).strip()

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



Settings.llm = Ollama(
    model="qwen2.5:1.5b-instruct",
    base_url="http://localhost:11434",
    request_timeout=120.0,
    additional_kwargs={

        "num_ctx": 1024,   
        "num_predict": 64,    # labels are short; trim generation budget (will play around with)
        # make labels stable run-to-run
        "temperature": 0.0,   # (will try 0.0)
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
def label_clusters_llm(cluster_to_events, max_samples=8):
    labels = {}
    FEWSHOT = (
        "You name text clusters with a concise 2–5 word noun phrase. "
        "Return ONLY the label—no quotes, no punctuation, no extra text.\n\n"
        "Examples:\n"
        "Samples:\n- Fuel starvation\n- Fuel exhaustion\n- Fuel contamination\n"
        "Label: Fuel system issues\n\n"
        "Samples:\n- Bird strike\n- Wildlife encounter\n"
        "Label: Wildlife hazards\n\n"
    )
    for cid, evs in cluster_to_events.items():
        if not evs:
            labels[cid] = "Miscellaneous"
            continue
        samples = evs[:max_samples]
        prompt = (
            FEWSHOT +
            "Samples:\n- " + "\n- ".join(samples) + "\n"
            "Label:"
        )
        raw = Settings.llm.complete(prompt).text.strip()
        # sanitize: keep it short, no punctuation/quotes
        lab = re.sub(r'["“”\.\:]+', '', raw).strip()
        lab = re.sub(r'\s+', ' ', lab)
        # clamp length
        labels[cid] = " ".join(lab.split()[:6]) if lab else "Miscellaneous"
    return labels

def choose_k_by_silhouette(X, k_min=4, k_max=18, random_state=42):
    best_k, best_score = None, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        # guard: silhouette requires >1 cluster and <n_samples unique labels
        if len(set(labels)) > 1 and len(set(labels)) < len(labels):
            score = silhouette_score(X, labels, metric="euclidean")
            if score > best_score:
                best_k, best_score = k, score
    return best_k or max(8, min(12, k_max))  # sensible fallback


def cluster_events(events, num_clusters=None):
    # normalize + dedupe (you already added this; keep it)
    events = [normalize_event_name(e) for e in events]
    events = [e for e in events if e]
    events = sorted(set(events))

    if len(events) < 2:
        with open("grouped_defining_events.txt", "w", encoding="utf-8") as out:
            out.write("\n(no events)\n" if not events else f"\n--- Cluster 1: Single Group (1) ---\n  • {events[0]}\n")
        print("Not enough events to cluster.")
        return

# Rule-first → then subcluster
    clustered, cluster_names = rule_first_cluster(events)
# Merge exact-duplicate labels (e.g., "Fuel system: Fuel system issues")
    clustered, cluster_names = combine_duplicate_clusters(clustered, cluster_names)

    # Write out
    with open("grouped_defining_events.txt", "w", encoding="utf-8") as out:
        for cid in sorted(clustered, key=lambda c: (-len(clustered[c]), c)):
            name = cluster_names.get(cid, f"Cluster {cid + 1}")
            out.write(f"\n--- Cluster {cid + 1}: {name} ({len(clustered[cid])}) ---\n")
            for ev in sorted(clustered[cid]):
                out.write(f"  • {ev}\n")
    print("Clustered events saved to grouped_defining_events.txt")

def _normalize_label_key(label: str) -> str:
    """Make labels comparable: lowercase, strip punctuation, collapse spaces."""
    if not label:
        return ""
    key = label.lower()
    key = re.sub(r"[^a-z0-9]+", " ", key)   # drop punctuation/separators
    key = re.sub(r"\s+", " ", key).strip()
    return key

def combine_duplicate_clusters(
    clustered: dict[int, list[str]],
    cluster_names: dict[int, str]
) -> tuple[dict[int, list[str]], dict[int, str]]:
    """
    Merge clusters that have the same (normalized) label.
    Keeps the first occurrence's label; unions and sorts the events.
    """
    label_to_newid: dict[str, int] = {}
    new_clusters: dict[int, list[str]] = {}
    new_names: dict[int, str] = {}
    next_id = 0

    # determines order so merges are stable run-to-run
    for cid in sorted(clustered):
        label = cluster_names.get(cid, f"Cluster {cid + 1}")
        key = _normalize_label_key(label)

        if key not in label_to_newid:
            label_to_newid[key] = next_id
            new_clusters[next_id] = list(clustered[cid])
            new_names[next_id] = label  
            next_id += 1
        else:
            target = label_to_newid[key]
            new_clusters[target].extend(clustered[cid])

    # tidying up: dedupe + sort events within each merged cluster? might need to reowkr
    for nid in new_clusters:
        new_clusters[nid] = sorted(set(new_clusters[nid]))

    return new_clusters, new_names

# MAIN PROGRAM
if __name__ == "__main__":
    process_text_files()
    create_summary_json()

    with open(event_list_path, "r", encoding="utf-8") as f:
        events = [line.strip() for line in f if line.strip()]


    cluster_events(events)

    event_counter, records = extract_raw_events(folder_path)
    if len(event_counter) > 0:
        plot_event_distribution(event_counter)
        plot_event_pie(event_counter)
        plot_event_treemap(event_counter)
    else:
        print("No events found for plotting.")

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
