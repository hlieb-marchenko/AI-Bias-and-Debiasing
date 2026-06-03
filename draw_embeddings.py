# embeddings_plot_pairs.py
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import numpy as np
import re
import sys

model_name = "distilbert-base-uncased"

print("Loading model and tokenizer (this may take a while)...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
print("Loaded.\n")

def get_static_word_vector(word, embeddings, tokenizer):
    token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if len(token_ids) == 0:
        print(f"Warning: no tokens found for '{word}'; returning zero vector.")
        return np.zeros(embeddings.shape[1], dtype=float)
    vecs = embeddings[token_ids]
    return vecs.mean(axis=0)

def add_line(fig, start, end, color='blue', width=4, name=None):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines+markers',
        line=dict(color=color, width=width),
        marker=dict(size=3),
        name=name
    ))

# Default words
default_words = "doctor, nurse, man, woman, engineer, teacher, king, queen"
print("Enter words (comma-separated). Press Enter to use default:")
print(f"Default: {default_words}")
s = input("> ").strip()
if s == "":
    s = default_words
words = [w.strip() for w in s.split(",") if w.strip()]
if len(words) == 0:
    print("No words provided. Exiting.")
    sys.exit(0)

# compute static vectors
word_vecs = {w: get_static_word_vector(w, embeddings, tokenizer) for w in words}

# PCA to 3D
n_components = min(3, len(words))
vec_array = np.vstack([word_vecs[w] for w in words])
pca = PCA(n_components=n_components)
coords = pca.fit_transform(vec_array)
if n_components < 3:
    coords = np.hstack([coords, np.zeros((coords.shape[0], 3 - n_components))])

# show list to user (0-based indices)
print("\nWords and indices (0-based):")
for i, w in enumerate(words):
    c = coords[i]
    print(f"{i}: {w}  -> (PC1={c[0]:.4f}, PC2={c[1]:.4f}, PC3={c[2]:.4f})")

# Parsing function for explicit pairs
def parse_pairs(s, words):
    """
    Accepts strings like:
      "0-2,1-3"
      "1 and 3, 2 and 4"
      "doctor-king, nurse-queen"
      "(1,3)&(2,4)"
    Indices are interpreted as 0-based; if an index is within 1..len(words) but not 0..len-1,
    we also try treating it as 1-based.
    """
    if not s or not s.strip():
        return []
    pairs = []
    # split by top-level separators
    tokens = re.split(r'[,&;]+', s)
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        # replace common words with hyphen delimiter
        t = re.sub(r'\band\b', '-', t, flags=re.I)
        t = re.sub(r'\bto\b', '-', t, flags=re.I)
        t = t.replace(':', '-')
        # extract two items: prefer numbers or contiguous word characters
        # split on hyphen or any sequence of non-alphanumeric characters
        parts = re.split(r'[-\s]+', t)
        parts = [p for p in parts if p != '']
        if len(parts) < 2:
            # fallback: find numbers or words via regex
            nums = re.findall(r'\d+', t)
            words_found = re.findall(r"[A-Za-z']+", t)
            if len(nums) >= 2:
                parts = nums[:2]
            elif len(words_found) >= 2:
                parts = words_found[:2]
            else:
                print(f"Could not parse pair '{t}', skipping.")
                continue
        a, b = parts[0], parts[1]
        def resolve(x):
            if x.isdigit():
                idx = int(x)
                if 0 <= idx < len(words):
                    return words[idx]
                if 1 <= idx <= len(words):
                    # allow 1-based index too
                    return words[idx-1]
                return None
            # case-insensitive exact match
            for w in words:
                if w.lower() == x.lower():
                    return w
            # prefix match
            for w in words:
                if w.lower().startswith(x.lower()):
                    return w
            return None
        ra = resolve(a)
        rb = resolve(b)
        if ra is None or rb is None:
            print(f"Could not resolve pair items '{a}' or '{b}' to known words; skipping.")
            continue
        pairs.append((ra, rb))
    # unique preserving order
    seen = set()
    unique = []
    for p in pairs:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            unique.append(p)
    return unique

print("\nEnter explicit pairs to connect (comma separated). Examples:")
print("  0-2,1-3        (use indices, 0-based; 1-based also accepted)")
print("  (1 and 3)&(2 and 4)")
print("  doctor-king, nurse-queen")
print("Leave blank for no pairs.")
pair_input = input("> ").strip()
pairs = parse_pairs(pair_input, words)
if len(pairs) == 0:
    print("No pairs selected.")

# Option to draw difference vectors
draw_diff = False
if len(pairs) > 0:
    print("\nAlso draw difference vectors (b - a) from origin for each pair? (y/N)")
    yn = input("> ").strip().lower()
    draw_diff = (yn == 'y' or yn == 'yes')

# Build plot
fig = go.Figure()
# blue vectors from origin
for i, w in enumerate(words):
    add_line(fig, start=[0,0,0], end=coords[i], color='blue', width=4, name=w)

# labels
fig.add_trace(go.Scatter3d(
    x=coords[:,0],
    y=coords[:,1],
    z=coords[:,2],
    mode='text',
    text=words,
    textposition='top center',
    showlegend=False
))

# add red connectors/diffs for each explicit pair
for a, b in pairs:
    ia = words.index(a)
    ib = words.index(b)
    pa = coords[ia]
    pb = coords[ib]
    add_line(fig, start=pa, end=pb, color='red', width=6, name=f"{a}↔{b}")
    if draw_diff:
        diff = pb - pa
        add_line(fig, start=[0,0,0], end=diff, color='green', width=4, name=f"{b}-{a} (diff)")

out_file = "embeddings.html"
fig.write_html(out_file)
print(f"\nSaved interactive plot to {out_file}. Open it in a browser to explore the 3D plot.")