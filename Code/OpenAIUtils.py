"""
===============================================================================
File:           OpenAIUtils.py
Description:    Utility function to invoke the OpenAI API to get
                response to prompt engineered queries

Author:         [Mahesh Babu A K]
Created:        [2025-06-10]
Last Updated:   [2025-06-23]
Python Version: 3.8+

Dependencies:
    - nltk
    - torch
    - transformers
    - openai
===============================================================================
"""

# For calculating BLEU score and applying smoothing techniques
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# For tokenizing input and loading pretrained GPT-2 model
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# For tensor operations and GPU acceleration
import torch

import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    #api_key=os.environ.get("OPENAI_API_KEY"),
    api_key= "<Use_OpenAI_Key>"
)

def get_openai_response(prompt, instruct=None):
    response = client.responses.create(
    model="gpt-4.1",
    instructions=instruct,
    input=prompt,)

    output = response.output_text
    #print(output)
    return output
    
def get_summarized_response(prompt, instruct=None):
    
    ret_text = prompt.strip()  # Default to the original prompt if no output is generated
    checkPrompt = prompt.strip()
    checkPrompt = checkPrompt.lower()

    doQuery = False
    output = ""

    if ("lyrics" in checkPrompt 
        or checkPrompt.startswith("lyrics") 
        or checkPrompt.endswith("lyrics")):
        doQuery = True
    elif ("song" in checkPrompt 
        or checkPrompt.startswith("song") 
        or checkPrompt.endswith("song")):
        doQuery = True

    if (doQuery==True):
        output = get_openai_response(prompt, instruct)
        #print(output)
        output = output.strip()

    
    if output:
        if ":" in output:
            # Extract the part after the first colon
            ret_text = output.split(":", 1)[1].strip()
        else:
            ret_text = output.strip()

    #print(ret_text)
    return ret_text

'''
if __name__ == '__main__':
    instruct1 = "Act as an expert YouTube video search assistant and text summarizer. Determine if the intent is to search song by title or by lyrics and return only lyrics part or the title of the song as response and nothing else"
    prompt1 = "I want to find YouTube videos having the name of the song like Lush life" 

    instruct2 = "Act as an expert YouTube video search assistant and text summarizer. Determine if the intent is to search song by title or by lyrics and return only lyrics part or the title of the song as response and nothing else"
    prompt2 = "I want to find YouTube videos having the lyrics I live my day as if it was the last."

    instruct3 = "Act as an expert YouTube video search assistant and text summarizer. Determine if the intent is to search song by title or by lyrics and return only lyrics part or the title of the song as response and nothing else"
    prompt3 = "I live my day as if it was the last."

    get_summarized_response(prompt1, instruct1)
'''
