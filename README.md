# 🎬 IMDb Sentiment Analyzer

A deep learning web app that predicts whether a movie review is **positive** or **negative**, built with Keras and Gradio.

## Demo

![App Screenshot](https://github.com/user-attachments/assets/5dd76116-e4b8-4d9b-8c2f-e5943d0e5f2a)

## Features

- Enter any movie review and get instant sentiment prediction
- Displays confidence probability score
- Trained deep learning model (Keras)
- Simple and shareable web interface via Gradio

## Tech Stack

| Layer | Tool |
|---|---|
| Model | Keras (saved as `best_model.h5`) |
| Interface | Gradio |
| Language | Python |

## Installation

```bash
git clone https://github.com/BerkeTozkoparan/imdb-sentiment-analyzer.git
cd imdb-sentiment-analyzer
pip install -r requirements.txt
python i̇mbd_app.py
```

## Requirements

```
tensorflow>=2.20.0
keras>=3.10.0
gradio==3.44.0
numpy
```

## Usage

1. Run the app
2. Paste or type a movie review into the text box
3. Click **Submit** to get the sentiment prediction and confidence score

## Example

> *"This movie was an absolute disaster. Terrible acting, a nonsensical plot, and completely wasted my time."*
> → **Negative** (high confidence)
