import pandas as pd

df = pd.read_csv('top1000.csv')

df = df.dropna()

# Convert each column to a string during concatenation to avoid TypeErrors
df['description'] = (
    df['title_english'].astype(str) + ' ' + 
    df['genres'].astype(str) + ' ' + 
    df['format'].astype(str) + ' episodes: ' + 
    df['episodes'].astype(str)
)

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(sentences):
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

  with torch.no_grad():
      model_output = model(**encoded_input)

  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

  return sentence_embeddings



sentences = ['Some great movie', 'Another funny movie']
result = get_embeddings(sentences)
print("Sentence embeddings:")
print(result)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

anime_embeddings = model.encode(df['description'].tolist())

from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(query, embeddings, df, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]