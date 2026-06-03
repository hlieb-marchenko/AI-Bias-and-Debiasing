import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

LAMBDA_PROJ = 0.8
MAX_LEN = 64
EPOCHS = 15
TESTS_NUMBER = 5

# --------------------------------------------------
# gender direction
# --------------------------------------------------

male_sentences = [
    "He is a doctor",
"He is a teacher",
"He is an engineer",
"He is a lawyer",
"He is a scientist",
"He is a programmer",
"He is a mechanic",
"He is a pilot",
"He is a chef",
"He is a musician",
"He is a writer",
"He is a professor",
"He is a manager",
"He is a designer",
"He is a carpenter",
"He is a farmer",
"He is a soldier",
"He is a firefighter",
"He is a police officer",
"He is a researcher",
"The man arrived",
"The man entered the room",
"The man spoke to the crowd",
"The man finished the project",
"The man solved the problem",
"The man drove the car",
"The man fixed the machine",
"The man wrote the report",
"The man explained the idea",
"The man built the house",
"The boy ran across the field",
"The boy finished his homework",
"The boy played the piano",
"The boy read the book",
"The boy solved the puzzle",
"The father helped his child",
"The father cooked dinner",
"The father drove to work",
"The father repaired the door",
"The father told a story",
"The brother opened the window",
"The brother carried the bag",
"The brother answered the question",
"The brother cleaned the room",
"The brother started the engine"
]

female_sentences = [
    "She is a doctor",
"She is a teacher",
"She is an engineer",
"She is a lawyer",
"She is a scientist",
"She is a programmer",
"She is a mechanic",
"She is a pilot",
"She is a chef",
"She is a musician",
"She is a writer",
"She is a professor",
"She is a manager",
"She is a designer",
"She is a carpenter",
"She is a farmer",
"She is a soldier",
"She is a firefighter",
"She is a police officer",
"She is a researcher",
"The woman arrived",
"The woman entered the room",
"The woman spoke to the crowd",
"The woman finished the project",
"The woman solved the problem",
"The woman drove the car",
"The woman fixed the machine",
"The woman wrote the report",
"The woman explained the idea",
"The woman built the house",
"The girl ran across the field",
"The girl finished her homework",
"The girl played the piano",
"The girl read the book",
"The girl solved the puzzle",
"The mother helped her child",
"The mother cooked dinner",
"The mother drove to work",
"The mother repaired the door",
"The mother told a story",
"The sister opened the window",
"The sister carried the bag",
"The sister answered the question",
"The sister cleaned the room",
"The sister started the engine",
]

def embed_sentence(text):

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.hidden_states[-1]
    vec = hidden.mean(dim=1)

    return vec.squeeze(0)

male_vecs = torch.stack([embed_sentence(s) for s in male_sentences])
female_vecs = torch.stack([embed_sentence(s) for s in female_sentences])

g = male_vecs.mean(dim=0) - female_vecs.mean(dim=0)
g = g / torch.norm(g)

g_t = g.detach()

# ----------------
# Test
# ----------------

print("Test BEFORE training:")
prompt = "The nurse said that"

for _ in range(TESTS_NUMBER):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_length=20,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )
    print(tokenizer.decode(out[0]))

# --------------------------------------------------
# training data
# --------------------------------------------------

print("Loading CrowS-Pairs dataset...")

url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

df = pd.read_csv(url)

sentences = []

for _, row in df.iterrows():

    if row["bias_type"] != "gender":
        continue

    s1 = row["sent_more"]
    s2 = row["sent_less"]

    sentences.append(s1)
    sentences.append(s2)

print("Loaded sentences:", len(sentences))

# --------------------------------------------------
# training loop
# --------------------------------------------------

for epoch in range(EPOCHS):

    total = 0

    for text in sentences:

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN
        ).to(device)

        input_ids = enc["input_ids"]

        outputs = model(
            input_ids=input_ids,
            labels=input_ids
        )

        lm_loss = outputs.loss

        hidden = outputs.hidden_states[-1]
        sentence_vec = hidden.mean(dim=1)

        proj = torch.sum(sentence_vec * g_t, dim=1)
        proj_loss = (proj ** 2).mean()

        loss = lm_loss + LAMBDA_PROJ * proj_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    print("epoch", epoch, "loss", total)

# ----------------
# Test
# ----------------

print("Test AFTER training:")
prompt = "The nurse said that"

for _ in range(TESTS_NUMBER):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_length=20,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )
    print(tokenizer.decode(out[0]))


model.save_pretrained("debiased_gpt2")
tokenizer.save_pretrained("debiased_gpt2")

