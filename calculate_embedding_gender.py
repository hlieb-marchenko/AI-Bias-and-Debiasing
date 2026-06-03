from transformers import AutoTokenizer, AutoModel
import numpy as np

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()

def get_static_word_vector(word, embeddings, tokenizer):  
    # returns averaged embedding for all wordpiece tokens of `word`  
    token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]  
    vecs = embeddings[token_ids]           # shape (n_subtokens, dim)  
    return vecs.mean(axis=0)  
  
  
def make_gender_direction(male_words, female_words, embeddings, tokenizer, normalize=True):  
    male_vecs = np.array([get_static_word_vector(w, embeddings, tokenizer) for w in male_words])  
    female_vecs = np.array([get_static_word_vector(w, embeddings, tokenizer) for w in female_words])  
    g = male_vecs.mean(axis=0) - female_vecs.mean(axis=0)  
    if normalize:  
        g = g / np.linalg.norm(g)  
    return g  
  
  
def projection_score(vec, direction):  
    # scalar projection (signed); positive means same side as direction  
    return float(np.dot(vec, direction))  
  
  
def cosine(a, b):  
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))  
  
  
# Example pronoun lists (you can expand)  
male_attr = ["he", "him", "his", "man", "male"]  
female_attr = ["she", "her", "hers", "woman", "female"]  
  
  
# build gender axis  
g = make_gender_direction(male_attr, female_attr, embeddings, tokenizer)  
  
  
# compute projections for your words  
words = input("Enter words (comma-separated): ").strip().split(",") 
word_vecs = {w: get_static_word_vector(w, embeddings, tokenizer) for w in words}  
for w in words:  
    w.strip()
    proj = projection_score(word_vecs[w], g)         # scalar along gender axis  
    cos_f = np.mean([cosine(word_vecs[w], get_static_word_vector(a, embeddings, tokenizer)) for a in female_attr])  
    cos_m = np.mean([cosine(word_vecs[w], get_static_word_vector(a, embeddings, tokenizer)) for a in male_attr])  
    print(f"{w:12s}  proj={proj:+.4f}   cos_f={cos_f:.4f}  cos_m={cos_m:.4f}  cos_f-cos_m={cos_f-cos_m:+.4f}")