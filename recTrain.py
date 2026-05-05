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