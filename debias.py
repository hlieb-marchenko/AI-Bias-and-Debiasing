# debias_distilbert_projection_selective.py

import random
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# ============================================================
# 1. Setup
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 64
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3

LAMBDA_PROJ = 0.1

# ============================================================
# 2. Model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)

# ============================================================
# 3. Probe sentences (for inspecting embeddings)
# ============================================================

probe_sentences = [
    "The doctor is very competent.",
    "The nurse is very competent.",
    "He is very competent.",
    "She is very competent.",
    "The engineer completed the task successfully.",
    "The assistant completed the task successfully."
]

# ============================================================
# 4. Sentence embedding (mean pooling)
# ============================================================

def sentence_embedding(model, input_ids, attention_mask):

    outputs = model.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    hidden = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1)

    pooled = (hidden * mask).sum(1) / mask.sum(1)

    return pooled

def embed_sentence(text):

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():

        vec = sentence_embedding(
            model,
            enc["input_ids"],
            enc["attention_mask"]
        )

    return vec.squeeze(0)

# ============================================================
# 5. Gender direction
# ============================================================

male_sentences = [
    "He is a doctor",
    "He is a teacher",
    "The man arrived",
    "The father is speaking"
]

female_sentences = [
    "She is a doctor",
    "She is a teacher",
    "The woman arrived",
    "The mother is speaking"
]

def build_gender_direction():

    male_vecs = []
    female_vecs = []

    for s in male_sentences:
        male_vecs.append(embed_sentence(s).cpu().numpy())

    for s in female_sentences:
        female_vecs.append(embed_sentence(s).cpu().numpy())

    male_vecs = np.array(male_vecs)
    female_vecs = np.array(female_vecs)

    g = male_vecs.mean(axis=0) - female_vecs.mean(axis=0)

    g = g / np.linalg.norm(g)

    return torch.tensor(g, dtype=torch.float32, device=device)

g_t = build_gender_direction()

print("Gender direction shape:", g_t.shape)

# ============================================================
# 6. Probe projections BEFORE training
# ============================================================

print("\nSentence projections BEFORE training:")

for s in probe_sentences:

    h = embed_sentence(s)

    proj = torch.dot(h, g_t).item()

    print(f"{proj:+.4f} | {s}")

# ============================================================
# 7. Gender word detection (for selective debiasing)
# ============================================================

gender_words = {
    "he","him","his",
    "she","her","hers",
    "man","woman",
    "father","mother",
    "boy","girl",
    "male","female",
    "husband","wife",
    "brother","sister"
}

def contains_gender_word(text):

    tokens = text.lower().replace(".", "").split()

    for t in tokens:
        if t in gender_words:
            return True

    return False

# ============================================================
# 8. Load CrowS-Pairs dataset
# ============================================================

print("\nLoading CrowS-Pairs dataset...")

url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

df = pd.read_csv(url)

texts = []
labels = []
neutral_mask = []

for _, row in df.iterrows():

    if row["bias_type"] != "gender":
        continue

    s1 = row["sent_more"]
    s2 = row["sent_less"]

    texts.append(s1)
    labels.append(1)

    if contains_gender_word(s1):
        neutral_mask.append(0)
    else:
        neutral_mask.append(1)

    texts.append(s2)
    labels.append(0)

    if contains_gender_word(s2):
        neutral_mask.append(0)
    else:
        neutral_mask.append(1)

print("Loaded gender samples:", len(texts))

# ============================================================
# 9. Dataset
# ============================================================

class TextDataset(Dataset):

    def __init__(self, texts, labels, neutral_mask):

        self.texts = texts
        self.labels = labels
        self.neutral_mask = neutral_mask

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}

        item["labels"] = torch.tensor(self.labels[idx])
        item["neutral_mask"] = torch.tensor(self.neutral_mask[idx]).float()

        return item

dataset = TextDataset(texts, labels, neutral_mask)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 10. Loss
# ============================================================

def compute_loss(batch):

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    neutral_mask = batch["neutral_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    task_loss = outputs.loss

    h = sentence_embedding(model, input_ids, attention_mask)

    proj = torch.sum(h * g_t, dim=1)

    proj_loss = proj ** 2

    mask_sum = neutral_mask.sum()

    if mask_sum > 0:
        proj_loss = (proj_loss * neutral_mask).sum() / mask_sum
    else:
        proj_loss = torch.tensor(0.0).to(device)

    total_loss = task_loss + LAMBDA_PROJ * proj_loss

    return total_loss

# ============================================================
# 11. Training
# ============================================================

optimizer = AdamW(model.parameters(), lr=LR)

print("\nTraining...")

for epoch in range(EPOCHS):

    model.train()

    running = 0

    for batch in loader:

        optimizer.zero_grad()

        loss = compute_loss(batch)

        loss.backward()

        optimizer.step()

        running += loss.item()

    print("Epoch", epoch + 1, "loss", running / len(loader))

# ============================================================
# 12. Probe projections AFTER training
# ============================================================

print("\nSentence projections AFTER training:")

for s in probe_sentences:

    h = embed_sentence(s)

    proj = torch.dot(h, g_t).item()

    print(f"{proj:+.4f} | {s}")

# ============================================================
# 13. Save
# ============================================================

SAVE_DIR = "./debias_crows_model"

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("\nModel saved to:", SAVE_DIR)