"""
===============================================================================
File:           YTUtils.py
Description:    Utility functions for extracting, processing, and analyzing
                YouTube video metadata and transcripts. Integrates YouTube Data API,
                Hugging Face NLP models, and OpenAI-based enhancements. Designed
                for use in video retrieval, summarization, and ML workflows.

Author:         [Mahesh Babu A K]
Created:        [2025-06-10]
Last Updated:   [2025-06-23]
Python Version: 3.8+

Dependencies:
    - google-api-python-client
    - youtube-transcript-api
    - transformers
    - openai
    - pandas
    - requests
    - webbrowser
    - MLUtils (custom module)
    - OpenAIUtils (custom module)
===============================================================================
"""

# For accessing the YouTube Data API
from googleapiclient.discovery import build

# For applying pre-trained NLP models (e.g., summarization, sentiment analysis)
from transformers import pipeline

# For regular expression operations
import re

# For interacting with the OpenAI API
from openai import OpenAI

# For making HTTP requests (e.g., for thumbnails or external data)
import requests

# For handling date and time in UTC
from datetime import datetime, timezone, timedelta

# For working with structured data
import pandas as pd

# For working with JSON responses and file dumps
import json

# For extracting and formatting YouTube transcripts
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api.formatters import TextFormatter

# For opening URLs in the system's default web browser
import webbrowser

# Custom utility modules for NLP and ML operations
import OpenAIUtils
import MLUtils



# Initialize YouTube API
YOUTUBE_API_KEY = 'AIzaSyDaHWpC0gegPmZXctsqis88m-3eY8Lj7Sw'  # Replace with your actual API key
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

API_KEY = 'AIzaSyDaHWpC0gegPmZXctsqis88m-3eY8Lj7Sw'  # Replace with your own API key

# search youtube vidoes using youtube API
def search_youtube_videos(query, max_results=20):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    search_response = youtube.search().list(
        q=query,
        type="video",
        part="id",
        maxResults=max_results
    ).execute()

    # return the retrieved video ids
    video_ids = [item["id"]["videoId"] for item in search_response["items"]]
    return video_ids

# check if video has text in transcript
def video_has_text_in_transcript(video_id, search_text):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        for entry in transcript:
            if search_text.lower() in entry['text'].lower():
                return True
    except (TranscriptsDisabled, VideoUnavailable):
        return False
    return False


# search for top videos given the query and search text
def find_top_videos_with_text(query, search_text, limit=3):
    video_ids = search_youtube_videos(query, max_results=25)
    matching_videos = []

    # build a list of matching video
    for video_id in video_ids:
        if video_has_text_in_transcript(video_id, search_text):
            matching_videos.append(f"https://www.youtube.com/watch?v={video_id}")
            if len(matching_videos) == limit:
                break

    return matching_videos

# get the timestamp of start of the video clip based on the query input
def get_timestamp_for_query(video_id: str, query: str):
    try:
        # get the transcript of the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except VideoUnavailable:
        print(f"Video {video_id} is unavailable.")
        return None
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video {video_id}.")
        return None
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

    # Normalize query for case-insensitive matching
    query_lower = query.lower()
    start_time = None

    # find the start time and return it
    for entry in transcript:
        if query_lower in entry['text'].lower():
            start_time = entry['start']
            print(f"Query found at {start_time:.2f} seconds")
            return start_time

    print("Query not found in transcript.")
    return None

# play youtube video given the url using webbrowser
def play_youtube_video(url):
    if "youtube.com/watch" in url or "youtu.be" in url:
        webbrowser.open(url)
    else:
        print("Invalid YouTube URL")

# get the transcript based on given language preference
def get_transcript(video_id, language_preference=['en']):
    try:
        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Get the transcript in preferred language
        for lang in language_preference:
            if transcript_list.find_transcript([lang]):
                transcript = transcript_list.find_transcript([lang])
                break
        else:
            transcript = transcript_list.find_transcript([transcript_list._TranscriptList__transcripts[0].language_code])

        # Format transcript as plain text
        formatter = TextFormatter()
        formatted_transcript = formatter.format_transcript(transcript.fetch())
        
        return formatted_transcript
    
    except Exception as e:
        return f"Error: {str(e)}"

# Handle preprocessing to ensure better match during search
def clean_input(text):

    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Utilization the facebook/bart-large-cnn model for text summarization
def generate_search_query(text):
    """Use a summarization model to create a concise search query"""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=30, min_length=5, do_sample=False)
    return summary[0]['summary_text']

# Search youtube and return top video results
def search_youtube(query, max_results=10):

    # build the youtube video request and invoke it
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video',
        relevanceLanguage='en',
        safeSearch='moderate'
    )
    response = request.execute()

    # get a tuple of video title and url
    results = []
    for item in response['items']:
        video_data = {
            'title': item['snippet']['title'],
            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }
        results.append(video_data)
    return results

# search youtube vidoes given text input
def find_youtube_videos_from_text(input_text):
    print(f"Searching youtube with: {input_text}")
    cleaned = clean_input(input_text)

    query = input_text.strip()  # Use the cleaned input directly for simplicity
    print(f"Generated Search Query: {query}")
    videos = search_youtube(query)
    return videos


# custom build context based intelligent search
def context_based_intelligent_search(query,selected_category, selected_filter):

    # create the instruction to extract the intent and summarize the query (Music, Movie)
    instruct_music = "Act as an expert YouTube video search assistant and text summarizer. Determine if the intent is to search song by title or by lyrics and return only lyrics part or the title of the song as response and nothing else"
    instruct_movie = "Act as an expert YouTube video search assistant and text summarizer. Determine if the intent is to search for movie by dialogue or by title and return only dialogue part or the title of the movie as response and nothing else"

    # define a variable to hold the response
    response = query.strip()  # Default to the original query if no output is generated
    # Based on the category given make request to Open AI to summarize the query
    if(selected_category == 'Music'):
       response = OpenAIUtils.get_summarized_response(query, instruct_music)
    elif(selected_category == 'Movie'):
       response = OpenAIUtils.get_summarized_response(query, instruct_movie)

    # print the response
    print("Response:")
    print(response)
    
    # use the summarized response text to search for youtube videos
    videos = find_youtube_videos_from_text(response)
    metadata = get_video_metadata(videos[0]['url']) 

    results = []
    df = pd.DataFrame()  # Start with empty DataFrame

    # enumerate through the videos and get the metadata of the videos from the search result
    for video in videos:
        # get the metadata given the youtube url and concatenate to the data frame
        metadata = get_video_metadata(video['url'])  # returns a dict
        if isinstance(metadata, dict):
            mtd = add_score_metrics(metadata,query,response)
            df = pd.concat([df, pd.DataFrame([mtd])], ignore_index=True)

    # filter the videos (select top 3 only) based on the filter criteria specified
    # this is the most important part of this custom build youtube video search
    if(selected_filter == 'Most Viewed'):
        results = df.sort_values(by="view_count", ascending=False).head(3)
    elif(selected_filter == 'Most Liked'):
        results = df.sort_values(by="like_count", ascending=False).head(3)
    elif(selected_filter == 'Most Recent'):
        results = df.sort_values(by="age_days", ascending=True).head(3)


    # print the results
    print(results)
    #MLUtils.plot_bleu_ppl_score(results)

    return results

# include the perplexity and bleu score metrics along with video metadata
def add_score_metrics(mtd, query, resp):

    candidate = query
    candidate += mtd['title']

    reference = resp;
    scores =  MLUtils.compute_bleu_perplexity_score(candidate,resp)
    mtd['bleu_score'] = scores['bleu_score']
    mtd['ppl_score'] = scores['ppl_score']
    #print(f"BLEU Score: {mtd['bleu_score']}")
    #print(f"Perplexity Score: {mtd['ppl_score']}")
    return mtd



# extract video id from the url
def extract_video_id(url):

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# get the video metadata using youtube metadata API
def get_video_metadata(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    api_url = (
        f"https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,contentDetails,statistics"
        f"&id={video_id}&key={API_KEY}"
    )

    response = requests.get(api_url)
    if response.status_code != 200:
        return {"error": f"API request failed with status code {response.status_code}"}

    data = response.json()
    if not data['items']:
        return {"error": "Video not found or API quota exceeded"}

    item = data['items'][0]
    snippet = item['snippet']
    statistics = item['statistics']
    content_details = item['contentDetails']

    # Video length in seconds
    duration_iso = content_details['duration']
    length_seconds = iso_duration_to_seconds(duration_iso)

    # Calculate how old the video is
    published_at = snippet['publishedAt']
    published_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - published_dt).days


    return {
        "video_url": video_url,
        "title": snippet['title'],
        "default_thumbnail": snippet['thumbnails']['default']['url'],
        "video_length_seconds": length_seconds,
        "view_count": int(statistics.get('viewCount', 0)),
        "like_count": int(statistics.get('likeCount', 0)),
        "age_days": age_days
    }

# get the video duration in seconds
def iso_duration_to_seconds(duration):
    # Converts ISO 8601 duration (e.g. PT1H2M30S) to seconds
    match = re.match(
        r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',
        duration
    )
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

# format the number as Million or Thousands
# to be used for displaying the Most Viewed, Most Liked, Most Recent numbers
def format_num(num):
    try:
        num = int(float(num))
    except (ValueError, TypeError):
        return "0"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)

# format the age of the video in Years and Months or Days
def format_days_age(num):
    try:
        num = int(float(num))
    except (ValueError, TypeError):
        return "0"
    if num >= 365:
        years = int((num / 365))
        months = int(((num % 365) / 12))
        return f"{years} Years {months} Months"
    elif num < 365:
        num = int(num)
        return f"{num} Days"
    return str(num)
