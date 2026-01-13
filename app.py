from flask import Flask, request, jsonify, render_template
import math
import os
import re
import numpy as np
import nltk
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

app = Flask(__name__)

# --- Resource Loading ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("Loading AI Model... Please wait.")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    if encodings.input_ids.size(1) == 0: return 100.0
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    return math.exp(outputs.loss.item())

def compute_burstiness(text):
    sentences = sent_tokenize(text)
    if len(sentences) < 2: return 0.0
    lengths = [len(word_tokenize(s)) for s in sentences]
    return np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0.0

def compute_mechanics(text):
    ox = len(re.findall(r",\s\w+,\s(and|or)\s", text))
    dash = text.count("—")
    semi = text.count(";")
    return (0.8 * ox) + (0.8 * dash) + (-1.0 * semi)

def compute_contractions(text):
    regs = [r"\b\w+(?:'t|'re|'ve|'ll|'d|'m)\b", r"\b(?:he|she|it|that|there|who|what|where|when|why|how)'s\b"]
    total = sum(len(re.findall(reg, text, re.I)) for reg in regs)
    words = word_tokenize(text)
    return total / len(words) if words else 0.0

def calculate_ai_probability(p, b, m, c):
    # --- 1. THE NEW AI TARGET (mu) ---
    # We increased the PPL target from 15.1 to 21.0 to match DistilGPT2's baseline.
    mu = np.array([21.0, 0.05, 1.0, 0.01]) 
    
    # --- 2. THE INPUT ---
    x = np.array([p, b, m, c])
    
    # --- 3. THE SCALES ---
    scales = np.array([50.0, 25.0, 1.0, 1.0]) 
    
    # --- 4. THE DELTA CALCULATION ---
    delta = x - mu
    
    # Adjusting the 'Perfect AI' plateau for the new PPL baseline
    if 20.9 <= p <= 21.1:
        delta[0] = 0
    else:
        delta[0] = abs(p - 21.0) - 0.1 

    # --- 5. THE DISTANCE & PROBABILITY ---
    d_m = math.sqrt(np.sum(scales * (delta**2)))
    probability = math.exp(-0.012 * d_m)
    
    return round(probability * 100, 2), {
        "ppl_weight": round(scales[0] * (delta[0]**2), 2),
        "burst_weight": round(scales[1] * (delta[1]**2), 2)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "No text"}), 400

        # Calculate metrics
        p = compute_perplexity(text)
        b = compute_burstiness(text)
        m = compute_mechanics(text)
        c = compute_contractions(text)
        
        # Get the new results and the new dictionary
        prob, debug_data = calculate_ai_probability(p, b, m, c)

        return jsonify({
            "probability": prob,
            "metrics": {
                "perplexity": round(p, 2),
                "burstiness": round(b, 3),
                "mechanics": round(m, 2),
                "contractions": round(c, 4)
            },
            "debug_info": {
                # UPDATED: These keys match your new calculate_ai_probability function
                "ppl_weight": debug_data["ppl_weight"],
                "burst_weight": debug_data["burst_weight"]
            }
        })
    except Exception as e:
        print(f"Server Error: {e}") # This will show up in your terminal
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 1. Get the port from Railway's environment (it defaults to 5000 for local testing)
    port = int(os.environ.get("PORT", 5000))
    
    # 2. Set host to '0.0.0.0' so it's accessible from outside the server
    app.run(host="0.0.0.0", port=port)