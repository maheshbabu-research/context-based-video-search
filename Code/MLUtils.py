"""
===============================================================================
File:           MLUtils.py
Description:    This script evaluates generated text using BLEU scores, utilizing
                NLTK and Hugging Face Transformers (GPT-2). It includes optional
                tokenization smoothing and GPU support via PyTorch.

Author:         [Mahesh Babu A K]
Created:        [2025-06-10]
Last Updated:   [2025-06-23]
Python Version: 3.8+

Dependencies:
    - nltk
    - torch
    - transformers
===============================================================================
"""

# BLEU score calculation and smoothing from NLTK
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# GPT-2 tokenizer and model from Hugging Face Transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# PyTorch for model handling and tensor operations
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta
import urllib.parse
import json
from statistics import mean

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')

# Compute the bleu and perplexity score
def compute_bleu_perplexity_score(candidate, reference):
    # BLEU Score
    # example candidate and reference
    #candidate = "I want to find YouTube videos having the name of the song like Lush life".split()
    #reference = ["Find YouTube videos of songs similar to Lush Life.".split()]

    # get the smoothing function
    smoothie = SmoothingFunction().method1

    # get the bleu score
    bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)

    # Perplexity using GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # example text input
    #text = "I want to find YouTube videos having the name of the song like Lush life"
    inputs = tokenizer(candidate, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)

    bleu_label = 'bleu_score'
    ppl_label = 'ppl_score'

    bppl_scores = {
    bleu_label: f"{bleu_score:.2f}",
    ppl_label: f"{perplexity.item():.2f}"
    }

    return bppl_scores

# Given a data frame with bleu score and perplexity score convert score columns to numeric and returns their averages
def avg_bleu_perplexity_score(df):
    # Convert string to numeric (coerce errors to NaN)
    df['bleu_score'] = pd.to_numeric(df['bleu_score'], errors='coerce')
    df['ppl_score'] = pd.to_numeric(df['ppl_score'], errors='coerce')

    # Compute means, skipping NaN values
    avg_bleu = df['bleu_score'].mean()
    avg_ppl = df['ppl_score'].mean()

    return {
        'average_bleu_score': round(avg_bleu, 2),
        'average_ppl_score': round(avg_ppl, 2)
    }

# plot bleu and perplexity score 
def plot_bleu_ppl_score(df):
    #df_sorted = bppl_scores.sort_values(by='BLEU Score', ascending=False)

    # Trim titles longer than 25 characters and add ellipsis
    df['title_trimmed'] = np.where(
        df['title'].str.len() > 30,
        df['title'].str[:25] + "...",
        df['title']
    )

    # Plotting placeholder
    plt.figure(figsize=(10, 8))
    plt.plot(df['title_trimmed'], df['bleu_score'], marker='o', label='BLEU')
    plt.plot(df['title_trimmed'], df['ppl_score'], marker='x', label='Perplexity')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('Video Title')
    plt.ylabel('Score')
    plt.title('BLEU and Perplexity Scores per Video')
    plt.legend()
    plt.tight_layout()
    plt.show()

# plot bleu and perplexity score sorted by bleu score
def plot_bleu_ppl(df):
    df_sorted = df.sort_values(by='BLEU Score', ascending=False)
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted['Title'], df_sorted['BLEU Score'], marker='o', label='BLEU')
    plt.plot(df_sorted['Title'], df_sorted['Perplexity'], marker='x', label='Perplexity')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Video Title')
    plt.ylabel('Score')
    plt.title('BLEU and Perplexity Scores per Video')
    plt.legend()
    plt.tight_layout()
    plt.show()



