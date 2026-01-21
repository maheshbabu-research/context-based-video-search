"""
===============================================================================
File:           Ctx_Video_Search.py
Description:    A context-aware video search interface integrating speech 
                recognition, GUI, and AI-enhanced video recommendation tools.

                This application combines Tkinter and PyQt5 for GUI, uses 
                OpenAI for semantic understanding, and leverages custom modules 
                for YouTube search, video detail parsing, and ML-based ranking.

                Features:
                - GUI-based video search with voice input
                - Interactive YouTube embedding and result display
                - Data visualization and ML support
                - External API and web integration (OpenAI, YouTube, etc.)

Author:         [Mahesh Babu A K]
Created:        [2025-06-10]
Last Updated:   [2025-06-23]
Python Version: 3.8+

Dependencies:
    - tkinter
    - PyQt5
    - speech_recognition
    - pandas, numpy, matplotlib
    - PIL (Pillow)
    - requests
    - openai
    - webbrowser, threading, os, sys
    - Custom modules: YTQuery, YTVidDetails, YTUtils, TransformerUtils, MLUtils

===============================================================================
"""

# GUI toolkit for building interface elements
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

# For opening URLs in a browser
import webbrowser

# For speech recognition functionality
import speech_recognition as sr

# For handling background tasks
import threading

# Data analysis and numerical computations
import pandas as pd
import numpy as np

# Plotting and visualization
import matplotlib.pyplot as plt

# For image processing and display in GUI
from PIL import Image, ImageTk

# For making HTTP requests and handling image byte streams
import requests
from io import BytesIO

# Interacting with the operating system
import os

# OpenAI API client
from openai import OpenAI

# Access to system-specific parameters and functions
import sys

# PyQt5 for advanced GUI components
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import QUrl

# Custom project-specific modules
import YTQuery
import YTVidDetails
import YTUtils
import TransformerUtils
import MLUtils


vids_list = []  # Global list to store video details
vid_metdata_list = []  # Global list to store video metadata

# ------------------
# Following are the global variable declaration for use across functions
# ------------------
global_window = None  # Global reference to the main Tkinter window

# Global QApplication instance (only one per process)
qt_app = None
qt_windows = []  # To keep references to open windows
qt_thread = None  # Thread for running the Qt event loop


category_frame = None # Global reference to checkboxes frame
checkbox_vars = None  # Global reference to checkbox variables
category_var = None  # Global reference to the category variable
selected_category = None  # Global reference to the selected category
categories = ["Music", "Movie", "Other"]  # Global reference to categories


filter_frame = None  # Global reference to the filter frame
filter_vars = None  # Global reference to filter variables  
filter_var = None  # Global reference to the filter variable
selected_filter = None  # Global reference to the selected filter
filters = ["Most Viewed", "Most Liked", "Most Recent"]


ml_tech_frame = None
ml_tech_var = None
selected_ml_tech = None
ml_techs = ["Gen AI", "Transformers", "Reinforcement Learning"] # Global reference to ML technologies

frames_list = []  # List to keep track of frames for clearing
container_vids_frame = None  # Global reference to the container frame for video thumbnails
clickable_frame = None  # Global reference to the clickable frame for video thumbnails


output_box = None  # Global reference to the output box

status_var = None  # Global reference to the status variable
status_bar = None  # Global reference to the status bar

entry = None  # Global reference to the search entry field

vids_frame = None  # Global reference to the video frame

container_vid1_frame = None  # Global reference to the first video container frame
container_vid2_frame = None  # Global reference to the second video container frame 
container_vid3_frame = None  # Global reference to the third video container frame

top_vids_result = None      # Global reference to the top video results
ppl_score_label = None      # Global refrence to perplexity score label
bleu_score_label = None     # Global referene to the bleu score label

# Create the frame and the radio buttons for Categories choices (Music,Movie,Other)
def create_category_choices():
    # global variables to use and hold the category choices
    global global_window, category_frame, category_var, categories, selected_category
    category_frame = tk.LabelFrame(global_window, text="Category" , bg='skyblue')
    category_frame.pack(pady=10, padx=10, fill="x")

    category_var = tk.StringVar(value="Music")
    selected_category = category_var.get()  # Initialize selected category
    # Radio button options
    categories = ["Music", "Movie", "Other"]
    for cat in categories:
        rb = tk.Radiobutton(category_frame, text=cat, variable=category_var, value=cat, bg='skyblue', font=("Arial", 12))
        rb.pack(side='left', padx=5, pady=5)  # Horizontal layout

# Create the filter choices frame and the radio buttons for (Most Viewed, Most Liked, Most Recent)
def create_filter_choices():

    # global variables to use and hold the filter choices
    global global_window, filter_vars, filter_frame, filter_var, selected_filter, filters

    filter_frame = tk.LabelFrame(global_window, text="Filter" , bg='skyblue')
    filter_frame.pack(pady=10, padx=10, fill="x")

    filter_var = tk.StringVar(value="Most Viewed")
    selected_filter = filter_var.get()  # Initialize selected filter
    filters = ["Most Viewed", "Most Liked", "Most Recent"]
    for filter in filters:
        rb = tk.Radiobutton(filter_frame, text=filter, variable=filter_var, value=filter, bg='skyblue', font=("Arial", 12))
        rb.pack(side='left', padx=5, pady=5)  # Horizontal layout

# Create the ML Tech choices frame and the radio buttons for the (Gen AI, Transformers, Reinforcement Learning)
def create_ml_tech_choices():

    # global variable to use and hold ML Tech choices
    global global_window, ml_tech_frame, ml_tech_var, selected_ml_tech, ml_techs

    ml_tech_frame = tk.LabelFrame(global_window, text="ML Approach" , bg='skyblue')
    ml_tech_frame.pack(pady=10, padx=10, fill="x")

    ml_tech_var = tk.StringVar(value="Gen AI")
    selected_ml_tech = ml_tech_var.get()  # Initialize selected filter
    ml_techs = ["Gen AI", "Transformers", "Reinforcement Learning"]
    for ml_tech in ml_techs:
        rb = tk.Radiobutton(ml_tech_frame, text=ml_tech, variable=ml_tech_var, value=ml_tech, bg='skyblue', font=("Arial", 12))
        rb.pack(side='left', padx=5, pady=5)  # Horizontal layout

# Create the search related fields (Entry, Speak, Search and Clear buttons and Labels)
def create_search_field():
    global ppl_score_label, bleu_score_label
    
    # Add the Search Youtube label
    tk.Label(global_window, text="Search YouTube", font=("Arial", 16), bg='skyblue').pack(pady=10)
    global entry
    entry = tk.Entry(global_window, font=("Arial", 14), width=100, bg='lightyellow')
    entry.pack(pady=5)

    # create the buttons frame
    btn_frame = tk.Frame(global_window, bg='skyblue')
    btn_frame.pack()
   
    # Create the Speak button
    tk.Button(
    btn_frame, 
    text="🎤 Speak", 
    command=lambda: threading.Thread(target=get_audio_input, args=(entry, status_var, global_window)).start(),
    bg="lightgreen",width=12, height=1, font=("Arial", 16)
    ).pack(side=tk.LEFT, padx=5)

    # Create the Search button
    tk.Button(
        btn_frame, 
        text="🔍 Search", 
        command=lambda: on_search(),
        bg="orange",width=12, height=1, font=("Arial", 16)
    ).pack(side=tk.LEFT, padx=5)

    # create the clear button
    tk.Button(
        btn_frame, 
        text="🧹 Clear", 
        command=lambda: entry.delete(0, tk.END), 
        bg="white",width=12, height=1, font=("Arial", 16)
    ).pack(side=tk.LEFT, padx=5)


    # create the score frame
    score_frame = tk.Frame(btn_frame, bg='skyblue')
    score_frame.pack(padx=60)

    # ceeate the perplexity score label field 
    ppl_score_label = tk.Label(score_frame, text="Avg Perplexity Score:", font=("Arial", 12), bg='skyblue')
    ppl_score_label.pack(side=tk.LEFT,pady=30)

    # create the BLEU score label fields
    bleu_score_label = tk.Label(score_frame, text="Avg BLEU Score:", font=("Arial", 12), bg='skyblue')
    bleu_score_label.pack(side=tk.LEFT,pady=30)

# create the output box to display results (textual ouput)
def create_output_box():
    # global variables to use and hold the output box
    global global_window, output_box
    output_box = scrolledtext.ScrolledText(global_window, height=8, wrap=tk.WORD, state='disabled', font=("Arial", 10), bg='skyblue')
    output_box.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)    

# create the vids frame
def create_vids_frame():
    # global variables to use and hold the video frame and elements
    global global_window, vids_frame
    vids_frame = tk.Frame(global_window, bg='skyblue')
    vids_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    # Add a label to the frame
    tk.Label(vids_frame, text="Video Search Details", font=("Arial", 16), bg='skyblue').pack(pady=5)
    
# create status bar
def create_status_bar():
    # global variables to use status bar
    global global_window, status_var, status_bar
    status_var = tk.StringVar()
    status_var.set("Ready.")

    # create status bar
    status_bar = tk.Label(global_window, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor='w', bg='white')
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# clear all children under this frame
# this is required to dynamic clear the child elements and re-create on repeated search
def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()
# cleaer all children expect the label inside this frame
def clear_except_labels(frame):
    for widget in frame.winfo_children():
        if not isinstance(widget, tk.Label):
            widget.destroy()

# clear frames
def clear_frames():
    global frames_list
    for frame in frames_list:
        clear_frame(frame)
    frames_list.clear()  # Clear the list after clearing all frames

def clear_scores():
    global ppl_score_label, bleu_score_label
    ppl_score_label.config(text=f"Avg Perplexity Score: ")
    bleu_score_label.config(text=f"Avg BLEU Score: ")

# setup the audio input thread, create and listen using the recognizer for speech input
def get_audio_input_thread(entry, status_var, root):
    # initialize the recognizer
    recognizer = sr.Recognizer()
    # check the microphone sources
    with sr.Microphone() as source:
        def update_status(msg):
            root.after(0, lambda: status_var.set(msg))

        # update listening status
        update_status("Listening for audio...")
        try:
            # listen to audio input
            audio = recognizer.listen(source, timeout=3)

            # speech to text using google recognizer
            text = recognizer.recognize_google(audio)

            # update the entry text field with text from speech
            root.after(0, lambda: entry.delete(0, tk.END))
            root.after(0, lambda: entry.insert(0, text))

            # update the status of the recognizer
            update_status("Audio recognized.")

        except sr.UnknownValueError:
            update_status("Could not understand audio.")
            root.after(0, lambda: messagebox.showerror("Error", "Could not understand audio"))
        except sr.RequestError:
            update_status("Speech recognition failed.")
            root.after(0, lambda: messagebox.showerror("Error", "Speech recognition service failed"))

def get_audio_input(entry, status_var, root):
    # Run the audio processing in a background thread
    threading.Thread(target=get_audio_input_thread, args=(entry, status_var, root), daemon=True).start()


# Search button action
def on_search():
    # define the global variables to be used during search action
    global global_window, entry
    global selected_category, entry, selected_filter, selected_ml_tech

    # get the status of the category, filter and MT Tech choices radio buttons
    selected_category = category_var.get()
    selected_filter = filter_var.get()
    selected_ml_tech = ml_tech_var.get()


    # Run in a background thread to fetch the video search results
    query = entry.get().strip()
    threading.Thread(target=fetch_and_disp_vid_details, args=(query,selected_category, selected_filter), daemon=True).start()

# thead handler for the video search results
def fetch_and_disp_vid_details(query, selected_category, selected_filter):
    
    # declare global variables
    global global_window, output_box, vids_frame, container_vids_frame, clickable_frame, vid_metdata_list, vids_list, top_vids_result

    global bleu_score_label, ppl_score_label

    clear_frames()
    clear_except_labels(vids_frame)
    clear_scores()
    vid_metdata_list.clear()  # Clear previous metadata
    vids_list.clear()  # Clear previous video details

    
    output_box.configure(state='normal')
    output_box.delete(1.0, tk.END)

    # selectively handle search results updates
    if(selected_ml_tech == 'Gen AI'):
        top_vids_result = YTUtils.context_based_intelligent_search(query, selected_category, selected_filter)
        ind = 1
        for idx, (_, metadata) in enumerate(top_vids_result.iterrows(), start=1):
            print(f"{idx}. {metadata['title']} - Views: {metadata['view_count']}")
            vid_metdata_list.append(metadata)  # Store metadata for later use
            video_data = {
                'title': metadata['title'],
                'url': metadata['video_url']   
            }

            string_output = f"{ind}. {video_data['title']} - {video_data['url']}\n"
            output_box.insert(tk.END, string_output)

            video_id = metadata['video_url'].split('v=')[-1]

            
            if(selected_category == 'Music'):
                transcript_text = YTUtils.get_transcript(video_id)
                ts = ""
                ts = transcript_text.lower()
                if("error" not in ts):
                    start_time_ret = YTUtils.get_timestamp_for_query(video_id,ts[100:20])
                    if( start_time_ret is not None):
                        print(f"Opening video at {int(start_time_ret)} seconds...")
                        seekTime = "Seek Time:"
                        startUrl = f"https://youtu.be/{video_id}&t={int(start_time_ret)}s\n"
                        seekTime += startUrl
                        output_box.insert(tk.END, seekTime)
                        output_box.insert(tk.END, transcript_text)
            print(string_output)
            vids_list.append(video_data)
            disp_yt_video_info(metadata, ind)
            ind += 1
        MLUtils.plot_bleu_ppl_score(top_vids_result)
        avg_score = MLUtils.avg_bleu_perplexity_score(top_vids_result)
        print(avg_score['average_bleu_score'])
        print(avg_score['average_ppl_score'])

        ppl_score_label.config(text=f"Avg Perplexity Score:{avg_score['average_ppl_score']}")
        bleu_score_label.config(text=f"Avg BLEU Score:{avg_score['average_bleu_score']}")



    elif(selected_ml_tech == 'Transformers'):
        top_vids_result  = TransformerUtils.run_context_search(query, selected_category, selected_filter)
        ind = 1
        for idx, (_, res) in enumerate(top_vids_result.iterrows(), start=1):
            print(f"{idx}. {res['Title']} - Views: {res['Views']}")
            metadata = {"title":res['Title'],
                        "video_url":res['Video URL'],
                          "default_thumbnail":res['Thumbnail'],
                          "video_length_seonds": '3:22',
                          "view_count":res['Views'],
                          "like_count":res['Likes'],
                          #"age_days":res['upload_date']
                          "age_days":777
                        }
            vid_metdata_list.append(metadata)  # Store metadata for later use
            video_data = {
                'title': metadata['title'],
                'url': metadata['video_url']   
            }

            string_output = f"{ind}. {video_data['title']} - {video_data['url']}\n"
            output_box.insert(tk.END, string_output)
            print(string_output)
            vids_list.append(video_data)
            disp_yt_video_info(metadata, ind)
            ind += 1

        MLUtils.plot_bleu_ppl_score(top_vids_result)
        avg_score = MLUtils.avg_bleu_perplexity_score(top_vids_result)
        print(avg_score['average_bleu_score'])
        print(avg_score['average_ppl_score'])

        ppl_score_label.config(text=f"Avg Perplexity Score:{avg_score['average_ppl_score']}")
        bleu_score_label.config(text=f"Avg BLEU Score:{avg_score['average_bleu_score']}")


    
# create the video display cards from video search results
def search_and_print_videos(query, selected_category, selected_filter):

    # declare the global variables to be used
    global global_window, output_box, vids_frame, container_vids_frame, clickable_frame, vid_metdata_list, vids_list

    clear_frames()
    clear_except_labels(vids_frame)
    clear_scores()
    vid_metdata_list.clear()  # Clear previous metadata
    vids_list.clear()  # Clear previous video details

    results = YTQuery.find_youtube_videos_from_text(query)
    output_box.configure(state='normal')
    output_box.delete(1.0, tk.END)
    for i, video in enumerate(results, 1):
        metadata = YTVidDetails.get_video_metadata(video['url'])
        vids_list.append(video)  # Store video details for later use
        vid_metdata_list.append(metadata)  # Store metadata for later use
        if metadata:
            disp_yt_video_info(metadata,i)
            string_output = f"{i}. {video['title']} - {video['url']}\n"
            output_box.insert(tk.END, string_output)
            print(f"{i}. {video['title']} - {video['url']}")
    output_box.configure(state='disabled')

# Convert seconds to MM:SS format
def format_length(seconds):
    minutes = seconds // 60
    sec = seconds % 60
    return f"{minutes}:{sec:02d}"

#  Display the youtube video info (metadata) using display card
def disp_yt_video_info(metadata, index=0): 
    # global variables to display youtube video card
    global global_window, qt_app, qt_windows, qt_thread, vids_frame, container_vids_frame, clickable_frame
    global container_vid1_frame, container_vid2_frame, container_vid3_frame, frames_list

    # This function can be expanded to fetch and display more metadata if needed
    print("Displaying YouTube video information...")


    # Create a frame to hold the thumbnails and use grid inside it
    thumbnail_frame = tk.Frame(global_window, bg='skyblue')
    thumbnail_frame.pack(pady=5)  # This uses pack, safe because it's a separate container

    # List of thumbnails
    thumbnails = []

    print("index:", index)
    # Create main frame
    container_vids_frame = ttk.Frame(vids_frame, padding=10)
    container_vids_frame.pack(side='left', padx=10)
    
    # Load thumbnail image
    try:
        response = requests.get(metadata['default_thumbnail'])
        img_data = Image.open(BytesIO(response.content))
        img_data = img_data.resize((120, 90))  # Resize for display
        thumbnail_img = ImageTk.PhotoImage(img_data)
    except Exception as e:
        thumbnail_img = None
        print("Failed to load thumbnail:", e)

    # Container frame to hold thumbnail + info + stats
    clickable_frame = ttk.Frame(container_vids_frame, padding=10, borderwidth=2, relief='ridge')
    clickable_frame.pack(side='top',  pady=10)

    # --- Thumbnail + Title ---
    info_frame = ttk.Frame(clickable_frame)
    info_frame.pack(side='top', pady=5)

    if thumbnail_img:
        thumbnail_label = ttk.Label(info_frame, image=thumbnail_img)
        thumbnail_label.image = thumbnail_img
        thumbnail_label.pack(side='left', padx=10)

    text_info_frame = ttk.Frame(info_frame)
    text_info_frame.pack(side='left')

    ttk.Label(text_info_frame, text="Title: " + metadata['title'], wraplength=190, justify="left").pack(anchor='w')

    if(selected_ml_tech != "Transformers"):
        ttk.Label(text_info_frame, text="Length: " + format_length(metadata['video_length_seconds'])).pack(anchor='w')
        age_str = YTUtils.format_days_age(metadata['age_days'])
        ttk.Label(text_info_frame, text=f"Age: {age_str}").pack(anchor='w')

    # --- Views and Likes ---
    stats_frame = ttk.Frame(clickable_frame)
    stats_frame.pack(side='top', pady=5)

    views_str = YTUtils.format_num(metadata['view_count'])
    views_label = tk.Label(stats_frame, text=f"👁 Views: {views_str}", fg="blue")
    views_label.pack(side='left', padx=20)

    likes_str = YTUtils.format_num(metadata['like_count'])
    likes_label = tk.Label(stats_frame, text=f"❤️ Likes: {likes_str}", fg="red")
    likes_label.pack(side='left', padx=20)

    if(index == 1):
        container_vid1_frame = container_vids_frame
        bind_vid1_click_recursive(container_vid1_frame, on_vid1_click)
        frames_list.append(container_vid1_frame)  # Keep track of this frame for clearing later
    elif(index == 2):
        container_vid2_frame = container_vids_frame
        bind_vid2_click_recursive(container_vid2_frame, on_vid2_click)
        frames_list.append(container_vid2_frame)  # Keep track of this frame for clearing later
    elif(index == 3):
        container_vid3_frame = container_vids_frame
        bind_vid3_click_recursive(container_vid3_frame, on_vid3_click)
        frames_list.append(container_vid3_frame)  # Keep track of this frame for clearing later

# click handler for 1st video card click
def on_vid1_click(event):
    print("vid1 clicked!")

    global vids_list, selected_category, output_box
    selected_category = category_var.get()

    # get the 1st video's url that is stored from the youtube search query result
    url = vids_list[0]['url']

    # launch the custom build youtube video player
    launch_ytvid_player(0)

# click handler for 2nd video card click
def on_vid2_click(event):
    print("vid2 clicked!")
    url = vids_list[1]['url']

    # launch the custom build youtube video player
    launch_ytvid_player(0)


# click handler for 3rd video card click
def on_vid3_click(event):
    print("vid3 clicked!")

    url = vids_list[1]['url']

    # launch the custom build youtube video player
    launch_ytvid_player(0)

# button ids to unique bind the three video card clicks on any of its children UI controls
vid_button1 = "<Button-1>"
vid_button2 = "<Button-2>"
vid_button3 = "<Button-3>"

# bind vid1 click event recursive to child controls
def bind_vid1_click_recursive(widget, callback):
    widget.bind("<Button-1>", callback)
    for child in widget.winfo_children():
        bind_vid1_click_recursive(child, callback)

# bind vid2 click event recursive to child controls
def bind_vid2_click_recursive(widget, callback):
    widget.bind("<Button-2>", callback)
    for child in widget.winfo_children():
        bind_vid2_click_recursive(child, callback)  

# bind vid3 click event recursive to child controls
def bind_vid3_click_recursive(widget, callback):
    widget.bind("<Button-3>", callback)
    for child in widget.winfo_children():
        bind_vid3_click_recursive(child, callback)       


# launch the custom Youtube Video Player
def launch_ytvid_player(index):
    # global variable to launch custom youtube video player
    global qt_app, qt_windows, qt_thread

    # PlayerWindow implement using QT5
    class PlayerWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Context-based YouTube Video Player {index + 1}")
            self.resize(1200, 900)

            layout = QVBoxLayout()
            self.browser = QWebEngineView()
            self.browser.setUrl(QUrl(vids_list[index]['url']))  # Use the URL from the vids_list
            layout.addWidget(self.browser)
            self.setLayout(layout)

        def closeEvent(self, event):
            # Warn user
            # messagebox.showwarning("Closing", "Video playback will stop.")
            # Remove self from window tracker
            if self in qt_windows:
                qt_windows.remove(self)
            if not qt_windows and qt_app:
                qt_app.quit()
            event.accept()

    try:
        if not qt_app:
            qt_app = QApplication(sys.argv)

        window = PlayerWindow()
        window.show()
        qt_windows.append(window)

        # Start Qt loop in background once
        if not hasattr(qt_app, 'started'):
            qt_app.started = True
            qt_thread = threading.Thread(target=qt_app.exec_, daemon=True)
            qt_thread.start()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to open video player: {e}")


    class QuietPage(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
            # Suppress console output from JS (e.g. CORS, ad scripts)
            pass

        def create_browser_with_quiet_page():
            browser = QWebEngineView()
            browser.setPage(QuietPage())  # Attach the custom page
            browser.setUrl(QUrl(vids_list[index]['url']))  # Use the URL from the vids_list
            return browser

# Alternative method to launch webbrowser to show Youtube Video Search Results
def open_youtube_search(query, filters, category, status_var, output_box):
    status_var.set("Searching...")
    base_url = "https://www.youtube.com/results?search_query="

    # Optional filter simulation
    filter_map = {
        "HD Only": ",hd",
        "Short Videos": ",short",
        "Recently Uploaded": ",recent"
    }
    filter_string = ''.join([filter_map[f] for f in filters])
    
    # Add category to query
    full_query = f"{query} {category}{filter_string}"
    search_url = base_url + full_query.replace(' ', '+')

    # Simulated output
    output_box.configure(state='normal')
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, f"Search Query: {query}\n")
    output_box.insert(tk.END, f"Filters: {', '.join(filters) if filters else 'None'}\n")
    output_box.insert(tk.END, f"Category: {category}\n")
    output_box.insert(tk.END, f"Opening in browser: {search_url}\n")
    output_box.configure(state='disabled')

    status_var.set("Search complete.")
    webbrowser.open(search_url)

# This is the main function of this Ctx_YT_Video_Search UI
# Various modularized functions are invoked here to create respective UI
# which intern has invocations to respective backend API methods found in 
# various Utils files like YTQuery, YTVidDetails, YTUils, MLUtils, OpenAPIUtils etc
def main():

    # global variable used for the main window of the UI
    global global_window

    # Initialize the main window
    global_window = tk.Tk()
    global_window.title("Context based YouTube Video Search Engine")

    # Change the background color using the 'configure' method
    global_window.configure(bg='skyblue')

    # Disable the maximize button by preventing resizing in both directions
    global_window.resizable(False, False)

    # set the geometry of the window (4x3)
    global_window.geometry("1200x900")

    # create categories choices frame and radio buttons
    create_category_choices()

    # create filter choices frame and radio buttons
    create_filter_choices()

    # create ML Tech choices frame and radio buttons
    create_ml_tech_choices()

    # create search related frame and UI elements
    create_search_field()

    # create the output box
    create_output_box()

    # create the vids frame and other UI elements to display video search results
    create_vids_frame()

    # create the status bar used to update speech recognizer status etc.
    create_status_bar()

    # run the main window loop
    global_window.mainloop()


# invoke the main function
if __name__ == "__main__":
    main()
