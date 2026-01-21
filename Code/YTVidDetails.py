"""
File:           YTVidDetails.py

Author:         [Mahesh Babu A K]
Created:        [2025-06-10]
Last Updated:   [2025-06-23]
Python Version: 3.8+

Description:
    This module is designed to fetch and process details about YouTube videos.
    It includes functionality for making HTTP requests to video pages, extracting
    metadata using regular expressions, and formatting timestamps.

Usage:
    Import this module in your project to retrieve and parse details from YouTube video URLs.

Dependencies:
    - requests: for sending HTTP requests to YouTube.
    - re: for parsing and extracting video metadata using regular expressions.
    - datetime: for managing and converting YouTube video timestamps.

"""

import requests  # For making HTTP requests to fetch YouTube video page content
import re        # For using regular expressions to parse HTML or JSON data
from datetime import datetime, timezone  # For handling and converting video timestamps


API_KEY = 'AIzaSyDaHWpC0gegPmZXctsqis88m-3eY8Lj7Sw'  # Use your own API_KEY here

# extract the youtube video id from the url using regular expression
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# get the video metadata given the video url
def get_video_metadata(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    # use google youtube API v3
    api_url = (
        f"https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,contentDetails,statistics"
        f"&id={video_id}&key={API_KEY}"
    )

    # get teh response 
    response = requests.get(api_url)
    if response.status_code != 200:
        return {"error": f"API request failed with status code {response.status_code}"}

    # get the JSON structure response
    data = response.json()
    if not data['items']:
        return {"error": "Video not found or API quota exceeded"}

    # get different category of metadata items required
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

    # return required video metadata to display to user using video card
    return {
        "title": snippet['title'],
        "default_thumbnail": snippet['thumbnails']['default']['url'],
        "video_length_seconds": length_seconds,
        "view_count": int(statistics.get('viewCount', 0)),
        "like_count": int(statistics.get('likeCount', 0)),
        "age_days": age_days
    }

# convert iso duration format to sections
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

# Example usage
# initialize video_url (using Zara Larson - Lush Life) as example
video_url = "https://www.youtube.com/watch?v=tD4HCZe-tew"
metadata = get_video_metadata(video_url)
# print metadata of example url
print("Printing example Metadata ...")
print(metadata)


title = metadata['title']
thumbnail = metadata['default_thumbnail']
length_seconds = metadata['video_length_seconds']
views = metadata['view_count']
likes = metadata['like_count']
age_in_days = metadata['age_days']

# Print the variables
print(f"Title: {title}")
print(f"Thumbnail: {thumbnail}")
print(f"Length (seconds): {length_seconds}")
print(f"Views: {views}")
print(f"Likes: {likes}")
print(f"Age (days): {age_in_days}")
