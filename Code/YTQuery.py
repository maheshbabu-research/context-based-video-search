"""
===============================================================================
File:           YTQuery.py
Description:    Utility script for extracting data from the YouTube Data API and
                applying natural language processing using Hugging Face Transformers.
                Includes basic text preprocessing using regular expressions.

Author:         [Mahesh Babu A K]
Created:        [2025-06-10]
Last Updated:   [2025-06-23]
Python Version: 3.8+

Dependencies:
    - google-api-python-client
    - transformers (Hugging Face)
    - re (Python standard library)
===============================================================================
"""

# For interacting with the YouTube Data API (v3)
from googleapiclient.discovery import build

# For applying pre-trained NLP models (e.g., sentiment analysis, summarization)
from transformers import pipeline

# For regular expression-based text cleaning and preprocessing
import re


from googleapiclient.discovery import build
from transformers import pipeline
import re

# Initialize YouTube API
YOUTUBE_API_KEY = 'AIzaSyDaHWpC0gegPmZXctsqis88m-3eY8Lj7Sw'  # Replace with your own API key
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# preprocess the input text for better search match
def clean_input(text):

    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Use a summarization model to create a concise search query
def generate_search_query(text):
    # use the facebook/bart-large-cnn model for summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=60, min_length=5, do_sample=False)
    return summary[0]['summary_text']

# Search for youtube video given the search query input using Youtube API
def search_youtube(query, max_results=10):

    # construct the query request
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    # invoke the youtube search
    request = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video',
        relevanceLanguage='en',
        safeSearch='moderate'
    )
    response = request.execute()

    # get the title and the url
    results = []
    for item in response['items']:
        video_data = {
            'title': item['snippet']['title'],
            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }
        results.append(video_data)
    return results

# find youtube vidoes from given text input query
def find_youtube_videos_from_text(input_text):
    cleaned = clean_input(input_text)
    query = generate_search_query(cleaned)
    print(f"Generated Search Query: {query}")
    videos = search_youtube(query)
    return videos

# Example Usage to test these functions
'''
if __name__ == '__main__':
    sample_input1 = """
    Hastala la vista, baby!
    """

    sample_input2 = """
    I live my day as if it was the last."""

    sample_input3 = """
    Agentic AI multi-modal agents.
    """

    results = find_youtube_videos_from_text(sample_input2)
    for i, video in enumerate(results, 1):
        print(f"{i}. {video['title']} - {video['url']}")
'''