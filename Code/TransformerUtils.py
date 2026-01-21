# Context-Based Video Search 
### Step 1: Install Required Packages
# Install essential libraries for NLP, YouTube access, plotting, and widgets
#!pip install -q transformers sentence-transformers youtube-transcript-api yt-dlp nltk ipywidgets matplotlib

### Step 2: Download NLTK Resources
# Download tokenization data needed for BLEU score calculation
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')

### Step 3: Import Required Libraries
# Import NLP models, YouTube APIs, plotting tools, tokenizers, and utility libraries
from transformers import BartTokenizer, BartForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import yt_dlp, torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import urllib.parse
import json
from statistics import mean
import time

### Step 4: Load Pre-trained Models
# Load BART for summarization, MiniLM for semantic similarity,
# GPT-2 for perplexity scoring, and a zero-shot classifier
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model.eval()

### Step 5: Define Utility Functions

# Summarizes a query using BART

def summarize_bart(text, max_len=40):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=max_len, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Converts seconds to timestamp string

def seconds_to_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

# Adds time-based navigation to a YouTube URL

def format_seek_url(base_url, seconds):
    if seconds is None:
        return base_url
    return f"{base_url}&t={int(seconds)}s"

# Gets transcript of a video using YouTubeTranscriptApi

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception:
        return []

# Finds best matching phrase from transcript for the given query

def find_best_seek_time(query, transcript):
    if not transcript:
        return None, ""
    max_sim = -1
    best_time = 0
    best_phrase = ""
    for entry in transcript:
        text = entry["text"]
        score = util.cos_sim(embedder.encode(query, convert_to_tensor=True), embedder.encode(text, convert_to_tensor=True)).item()
        if score > max_sim:
            max_sim = score
            best_time = entry["start"]
            best_phrase = text
    return best_time, best_phrase

# Performs YouTube search using yt-dlp

def search_youtube_via_yt_dlp(query, max_results=10):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'force_generic_extractor': True,
        'noplaylist': True,
        'default_search': 'ytsearch',
        'simulate': True,
        'dump_single_json': True
    }
    query = f"ytsearch{max_results}:{query}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(query, download=False)
            return result.get("entries", [])
        except Exception:
            return []

# Extracts and enriches metadata from YouTube result

def enrich_video_metadata(video):
    video_id = video.get("id")
    url = f"https://www.youtube.com/watch?v={video_id}"
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
    view_count = video.get("view_count", 0)
    likes = video.get("like_count")
    if likes is None:
        likes = int(view_count * 0.035)
    return {
        "id": video_id,
        "Title": video.get("title", "Unknown"),
        "Thumbnail": thumbnail_url,
        "Video URL": url,
        "Seek URL": url,
        "Views": view_count,
        "Likes": likes,
        "upload_date": video.get("upload_date", "N/A")
    }

# Calculates BLEU score for similarity

def bleu_score(reference, hypothesis):
    try:
        ref_tokens = [word_tokenize(reference.lower().strip())]
        hyp_tokens = word_tokenize(hypothesis.lower().strip())
        if not ref_tokens[0] or not hyp_tokens:
            return 0.0001
        smoothie = SmoothingFunction().method4
        return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
    except:
        return 0.0001

# Calculates Perplexity using GPT-2

def perplexity_score(text):
    if not text.strip():
        return 100.0
    encodings = gpt2_tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

# Plots BLEU score

def plot_bleu(df):
    df_sorted = df.sort_values(by='BLEU Score', ascending=False)
    plt.figure(figsize=(10, 4))
    plt.plot(df_sorted['Title'], df_sorted['BLEU Score'], marker='o', color='blue')
    plt.xticks(rotation=45, ha='right')
    plt.title("BLEU Scores per Video")
    plt.ylabel("BLEU Score")
    plt.xlabel("Video Title")
    plt.tight_layout()
    plt.show()

# Plots Perplexity score

def plot_perplexity(df):
    df_sorted = df.sort_values(by='Perplexity')
    plt.figure(figsize=(10, 4))
    plt.plot(df_sorted['Title'], df_sorted['Perplexity'], marker='x', color='red')
    plt.xticks(rotation=45, ha='right')
    plt.title("Perplexity Scores per Video")
    plt.ylabel("Perplexity")
    plt.xlabel("Video Title")
    plt.tight_layout()
    plt.show()

### Step 6: Run Search and Evaluation
# Handles the full pipeline:
# 1. Summarizes input if needed
# 2. Searches YouTube
# 3. Evaluates video transcript relevance
# 4. Computes and stores BLEU/Perplexity metrics
# 5. Saves summary output to JSON
# 6. Displays graphs

def run_context_search(user_query, category,filter):
    print(f"\nUser Query: {user_query}")

    if category == 'other':
        summary = summarize_bart(user_query)
        if summary.strip().lower() == user_query.strip().lower():
            summary = " ".join(user_query.split()[-6:])
        print(f"Summarized Query: {summary}")
    else:
        summary = user_query.strip()
        print(f"Direct Query Used (No Summarization): {summary}")

    print(f"User Selected Category: {category}")
    videos = search_youtube_via_yt_dlp(summary, max_results=10)

    results = []
    bleu_scores = []
    ppl_scores = []

    for video in videos:
        video = enrich_video_metadata(video)
        transcript = get_transcript(video["id"])
        seek_time, phrase = find_best_seek_time(summary, transcript)

        bleu = bleu_score(summary, phrase)
        ppl = perplexity_score(phrase)
        bleu_scores.append(bleu)
        ppl_scores.append(ppl)

        result = {
            "Category": category,
            "id": video["id"],
            "Title": video.get("Title", "Unknown"),
            "title": video.get("Title", "Unknown"),
            "Thumbnail": video.get("Thumbnail"),
            "Video URL": video.get("Video URL"),
            "Seek URL": format_seek_url(video["Seek URL"], seek_time) if seek_time else video.get("Video URL"),
            "Views": video.get("Views", 0),
            "Likes": video.get("Likes", 0),
            "upload_date": video.get("upload_date", "N/A"),
            "BLEU Score": round(bleu, 3),
            "Perplexity": round(ppl, 2),
            "bleu_score": round(bleu,2),
            "ppl_score": round(ppl,2)
        }
        results.append(result)

    df = pd.DataFrame(results)

    top_viewed = df.sort_values(by="Views", ascending=False).head(3).to_dict(orient="records")
    top_liked = df.sort_values(by="Likes", ascending=False).head(3).to_dict(orient="records")
    top_recent = df.sort_values(by="upload_date", ascending=False).head(3).to_dict(orient="records")

    avg_bleu = round(mean(bleu_scores), 3)
    avg_ppl = round(mean(ppl_scores), 2)

    grouped_results = {
        "category": category,
        "average_bleu_score": avg_bleu,
        "average_perplexity": avg_ppl,
        "most_viewed": top_viewed,
        "most_liked": top_liked,
        "most_recent": top_recent
    }

    with open("search_output.json", "w") as f:
        json.dump(grouped_results, f, indent=2)
    print("\n📁 Output saved to 'search_output.json'")

    print("\n📊 Average Metrics:")
    print(f"Average BLEU Score: {avg_bleu}")
    print(f"Average Perplexity Score: {avg_ppl}")

    plot_perplexity(df)
    plot_bleu(df)
    results = []

    if(filter == 'Most Viewed'):
        results = df.sort_values(by="Views", ascending=False).head(3)
    elif(filter == 'Most Liked'):
        results = df.sort_values(by="Likes", ascending=False).head(3)
    elif(filter == 'Most Recent'):
        results = df.sort_values(by="upload_date", ascending=True).head(3)


    #return df
    return results

### Step 7: Simple Text Input Instead of UI
# Take category and query input from the user to start the process
#category = input("Enter category (movie/music/other): ").strip().lower()
#query = input("Enter your search query: ").strip()
#df = run_context_search(query, category)
