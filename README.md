# context-based-video-search
Context Based Youtube Video Search

**Table of Contents**

1. [Overview](#overview)  
2. [Objectives](#objectives)  
3. [Problem Statement](#problem-statement)  
4. [Solution](#solution)  
5. [Dataset](#dataset)  
6. [Methods Used](#methods-used)  
7. [Technologies](#technologies)  
8. [Installation](#installation)  
9. [Usage](#usage)  
10. [Team Members](#team-members)  
11. [Acknowledgments](#acknowledgments)  
12. [License](#license)  

---
### Overview
This project aims to Enable precise video moment search using natural language and context-aware NLP.
for e.g. searching movie videos by dialogues or music videos by lyrics to view the Most Viewed, Most Liked and Most Recent Youtube videos
by leveraging Generative AI (ChatGPT) and other transformer models and Reinforcement Learning.
The project also aims to implement video transcript search and video topics timestamps to be able to quickly view relevant video clips inside the video.

### Objectives
- Develop a video moments search application
- Enable contextual natural language query to search movie videos by dialogue or music by lyrics
- View the Most Viewed, Most Liked and Most Recent Videos
- View Video Transcript and jump to video clip within video based on timestamp

### Problem Statement
- Unsuccessful video content search because of
- Overwhelming sea of content
- Frustrating to search specific moments like search by dialogue, lyrics or phrases
- No intuitive context-aware tools to locate exact scenes using Natural Language

### Solution
Context-Aware Video Search with Generative AI
Natural language-powered video moment search engine to search by dialogue, music lyrics or contextual text search
to list top youtube videos and jump directly to relevant timestamp in video.

### Dataset
We are utilizing the Youtube video Metadata using the Youtube API
Addtionally we will be utilizing a subset of the YouTube-8M Segments Dataset a very popular large scaled labelled video dataset comprising millions of video IDs
geared with diverse vocabulary facility and high-quality machine generated annotations of over 3800 visual entries. 

**Dataset link:** 
Link to Dataset: https://research.google.com/youtube8m/

### Methods Used
- Data Exploration and Filtering
- Natural Language Processing
- Machine Learning
- Python Programming
- TKInter in python for user interface

### Technologies
- Youtube API for accessing video metadata
- Open AI API for intent extraction and query summarization
- Python Libraries (TensorFlow, PyTorch, NLTK, TKInter, HuggingFace Transformers, Pandas, Dataset,PyQT5)
- UI development using TKInter, Dynamic UI for video search display, Widgets, WebBrowser, Youttube Player
- Transformer Models 
- Reinforcement Learning

### Installation

To set up the conversational chatbot project, follow these steps:

1. **Clone the Repository:**
   Open a terminal and run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/maheshbabu-usd/aai-510-group04-context-video-search.git
   ```

2. **Navigate to the Project Directory:**
   Change into the project directory:
   ```bash
   cd aai-510-group04-context-video-search/CtxVideoSearch/
   ```

3. **Set Up a Virtual Environment (Optional but Recommended):**
   Create a virtual environment to manage dependencies. You can use `venv` for this purpose:
   ```bash
   python -m venv CtxVidSearch
   ```
   Activate the virtual environment:
   - On Windows:
     ```bash
     CtxVidSearch\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source CtxVidSearch/bin/activate
     ```

4. **Install Dependencies:**
   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run the chatbot application, follow these steps:
1. **Install the pre-requisites**
   Install Dependencies using (pip install -r requirements.txt)
2. **Ensure to have Youtube API Key and Open AI API Key**
   Check and ensure to have the Youtube API Key and Open AI API Key and update the keys in the corresponding python files to use the APIs
3. **Launch the Application:**
   In the terminal, while still in the project directory, execute the following command:
   ```bash
   python Ctx_YT_Vid_Search.py
   ```

2. **Interact with the Context Based Video Search Application:**
   - A Graphical User Interface (GUI) window will open.
   - You can select the categories (Music, Movie or other) and filters (Most Viewed, Most Liked or Most Recent) and
     either provide speech input or text input to search youtube videos.
   - The top 3 video research results based on the category, filter and search query are display in the video card
   - Clicked on the video card to view the Video in the custom build Youtube Video Player

3. **Features:**
  - Natural Language Input (Speech + Text)
  - Search by Category:{Music, Movie or Other}
  - Filters: {Most Viewed, Most Liked, Most Recent}
  - Video Search Details - Dynamically Updated
  - Video Transcript & Video Seek
  - Choice of ML Approaches
  - Model Evaluation: BLEU Score and Perplexity Score

### Team Members

Mahesh Babu • Keerthana • Arkabo Biswas

### Acknowledgments
We would like to thank to our Instructor and project advisor Dr. Raj Garg for his invaluable guidance and support throughout the project.
Additionally, we would like to extend special thanks to YouTube team for catering us such a valuable resource for conducting our research. 

### License
This project is licensed under the MIT License - see the LICENSE file for details.

