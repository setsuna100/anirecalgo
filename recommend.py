import pandas as pd
import pickle
import difflib
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle('anime_df.pkl')

with open('anime_embeddings.pkl', 'rb') as f:
    anime_embeddings_spoilers = pickle.load(f)

with open('anime_embeddings_no_spoilers.pkl', 'rb') as f:
    anime_embeddings_no_spoilers = pickle.load(f)

model = SentenceTransformer('./local_anime_model')

all_genres_tags = set()
if 'genres' in df.columns:
    for g_val in df['genres'].dropna():
        for g in str(g_val).split('|'):
            if g.strip(): all_genres_tags.add(g.strip().lower())
if 'tags' in df.columns:
    for t_val in df['tags'].dropna():
        for t in str(t_val).split('|'):
            if t.strip(): all_genres_tags.add(t.strip().lower())

valid_keywords = sorted(list(all_genres_tags), key=len, reverse=True)

def clean_id(val):
    s = str(val).strip()
    return s[:-2] if s.endswith('.0') else s

id_to_idx = {clean_id(df.iloc[i]['id']): i for i in range(len(df))}
franchise_graph = {aid: set() for aid in id_to_idx}

if 'prequel_id' in df.columns and 'sequel_id' in df.columns:
    for aid, idx in id_to_idx.items():
        row = df.iloc[idx]
        if pd.notna(row.get('prequel_id')) and str(row['prequel_id']).strip():
            franchise_graph[aid].update(clean_id(x) for x in str(row['prequel_id']).split('|') if x.strip())
        if pd.notna(row.get('sequel_id')) and str(row['sequel_id']).strip():
            franchise_graph[aid].update(clean_id(x) for x in str(row['sequel_id']).split('|') if x.strip())

for node, neighbors in list(franchise_graph.items()):
    for neighbor in neighbors:
        if neighbor not in franchise_graph:
            franchise_graph[neighbor] = set()
        franchise_graph[neighbor].add(node)

visited = set()
idx_to_earliest_idx = {}

for node in franchise_graph:
    if node not in visited:
        component = set()
        stack = [node]
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                component.add(curr)
                for neighbor in franchise_graph.get(curr, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        valid_indices = [id_to_idx[n] for n in component if n in id_to_idx]
        if valid_indices:
            def get_sort_key(i):
                y = df.iloc[i].get('season_year')
                try: return float(y)
                except (ValueError, TypeError): return 9999.0
            
            earliest_idx = min(valid_indices, key=lambda i: (get_sort_key(i), i))
            for i in valid_indices:
                idx_to_earliest_idx[i] = earliest_idx

def get_recommendations(query, top_n=5, use_spoilers=False, allow_ecchi=False):
    original_query_lower = query.lower()
    search_query = query
    
    found_titles = []
    for i in range(len(df)):
        row = df.iloc[i]
        t_eng = str(row['title_english']) if pd.notna(row['title_english']) and row['title_english'] else ""
        t_rom = str(row['title_romaji']) if pd.notna(row['title_romaji']) and row['title_romaji'] else ""
        
        if len(t_eng) > 3 and t_eng.lower() in original_query_lower:
            found_titles.append((t_eng, i))
        if len(t_rom) > 3 and t_rom.lower() in original_query_lower:
            found_titles.append((t_rom, i))
            
    found_titles.sort(key=lambda x: len(x[0]), reverse=True)
    
    replaced_titles = []
    for title, idx in found_titles:
        if any(title.lower() in r.lower() for r in replaced_titles):
            continue
        match_row = df.iloc[idx]
        genres_str = str(match_row['genres']).replace('|', ' ') if pd.notna(match_row['genres']) else ""
        tags_str = str(match_row['tags']).replace('|', ' ') if pd.notna(match_row['tags']) else ""
        
        search_query = re.sub(re.escape(title), f"{genres_str} {tags_str}", search_query, flags=re.IGNORECASE)
        replaced_titles.append(title)
        
    if replaced_titles:
        titles_str = ", ".join(replaced_titles)
        print(f"\n  [Detected anime in prompt: {titles_str}. Searching by tags instead...]")
        
    mentioned_keywords = []
    temp_query = original_query_lower
    for kw in valid_keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, temp_query):
            mentioned_keywords.append(kw)
            temp_query = re.sub(pattern, ' ', temp_query)
            
    if mentioned_keywords:
        display_kws = ", ".join([kw.title() for kw in mentioned_keywords])
        print(f"\n  [Detected genres/tags in prompt: {display_kws}]")

    query_embedding = model.encode([search_query])

    target_embeddings = anime_embeddings_spoilers if use_spoilers else anime_embeddings_no_spoilers

    similarities = cosine_similarity(query_embedding, target_embeddings)[0]
    
    mean_scores = pd.to_numeric(df['mean_score'], errors='coerce').fillna(0)
    normalized_scores = mean_scores / 100.0
    
    score_boost = mean_scores.apply(lambda x: 1.2 if x >= 75 else 1.0)
    
    format_boost = df['format'].apply(lambda x: 1.0 if str(x).upper() in ['TV', 'MOVIE'] else 0.8)
    
    keyword_boost = []
    for i in range(len(df)):
        row = df.iloc[i]
        item_keywords = set()
        if 'genres' in df.columns and pd.notna(row['genres']):
            item_keywords.update([g.strip().lower() for g in str(row['genres']).split('|')])
        if 'tags' in df.columns and pd.notna(row['tags']):
            item_keywords.update([t.strip().lower() for t in str(row['tags']).split('|')])
        
        matches = sum(1 for kw in mentioned_keywords if kw in item_keywords)
        keyword_boost.append(1.0 + (0.3 * matches)) # 30% score boost per matching keyword
        
    combined_scores = ((similarities * 0.8) + (normalized_scores.values * score_boost.values * 0.2)) * format_boost.values * pd.Series(keyword_boost).values
    
    sorted_indices = combined_scores.argsort()[::-1]
    
    selected_indices = []
    seen_titles = []
    query_lower = original_query_lower

    for original_idx in sorted_indices:
        idx = idx_to_earliest_idx.get(int(original_idx), int(original_idx))
        
        if idx in selected_indices:
            continue
            
        # Filter out Ecchi/Hentai if not allowed
        if not allow_ecchi:
            g_str = str(df.iloc[idx]['genres']).lower() if pd.notna(df.iloc[idx]['genres']) else ""
            t_str = str(df.iloc[idx]['tags']).lower() if pd.notna(df.iloc[idx]['tags']) else ""
            if 'ecchi' in g_str or 'hentai' in g_str or 'ecchi' in t_str or 'hentai' in t_str:
                continue

        row = df.iloc[idx]
        t_eng = str(row['title_english']) if pd.notna(row['title_english']) and row['title_english'] else ""
        t_rom = str(row['title_romaji']) if pd.notna(row['title_romaji']) and row['title_romaji'] else ""
        
        # Skip if the show is directly mentioned in the user's prompt
        if (len(t_eng) > 3 and t_eng.lower() in query_lower) or (len(t_rom) > 3 and t_rom.lower() in query_lower):
            continue
            
        is_duplicate = False
        for seen in seen_titles:
            if t_eng and (t_eng.lower() in seen.lower() or seen.lower() in t_eng.lower() or difflib.SequenceMatcher(None, t_eng.lower(), seen.lower()).ratio() > 0.75):
                is_duplicate = True
                break
            if t_rom and (t_rom.lower() in seen.lower() or seen.lower() in t_rom.lower() or difflib.SequenceMatcher(None, t_rom.lower(), seen.lower()).ratio() > 0.75):
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
            
        selected_indices.append(idx)
        if t_eng: seen_titles.append(t_eng)
        if t_rom: seen_titles.append(t_rom)
        
        if len(selected_indices) == top_n:
            break

    return df.iloc[selected_indices]

print("System Ready!\n" + "-" * 50)