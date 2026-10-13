import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(db_path='anime_database.db'):
    """Loads user ratings from the SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM user_ratings", conn)
    conn.close()
    
    # Drop duplicates in case the userdata fetch script was run multiple times
    df = df.drop_duplicates(subset=['user', 'anime_id'])
    return df

def get_collaborative_recommendations(target_user, top_n=10):
    """Generates recommendations for a target user using User-User Collaborative Filtering."""
    df = load_data()
    
    if target_user not in df['user'].values:
        print(f"User '{target_user}' not found in the database.")
        return []

    # 1. Create the User-Item Matrix
    # Rows are users, Columns are anime titles, Values are ratings
    user_item_matrix = df.pivot(index='user', columns='title', values='rating').fillna(0)
    
    # 2. Compute User Similarity using Cosine Similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # 3. Get the most similar users to the target user
    similar_users = user_sim_df[target_user].sort_values(ascending=False)
    similar_users = similar_users.drop(target_user) # Remove self
    
    # Filter down to users with a positive similarity score
    similar_users = similar_users[similar_users > 0]
    
    if similar_users.empty:
        print("No similar users found to make recommendations.")
        return []

    # 4. Find anime the target user HAS NOT seen
    target_user_seen = set(df[df['user'] == target_user]['title'])
    all_anime = set(df['title'])
    unseen_anime = all_anime - target_user_seen
    
    # 5. Calculate predicted scores for unseen anime
    # This does a weighted average of ratings from similar users
    recommendations = {}
    
    for anime in unseen_anime:
        total_weight = 0
        weighted_sum = 0
        
        # Look at how the most similar users rated this anime
        for sim_user, sim_score in similar_users.items():
            rating = user_item_matrix.loc[sim_user, anime]
            if rating > 0:  # If the similar user actually rated it
                weighted_sum += sim_score * rating
                total_weight += sim_score
                
        if total_weight > 0:
            predicted_rating = weighted_sum / total_weight
            recommendations[anime] = predicted_rating
            
    # 6. Sort the recommendations by predicted rating
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recs[:top_n]

def get_item_collaborative_recommendations(target_anime, top_n=10):
    """Generates recommendations based on Item-Item similarity across ALL users' data."""
    df = load_data()
    
    if target_anime not in df['title'].values:
        print(f"Anime '{target_anime}' not found in the database.")
        return []

    # 1. Create the Item-User Matrix (transpose of User-Item)
    # Rows are anime titles, Columns are users, Values are ratings
    item_user_matrix = df.pivot(index='title', columns='user', values='rating').fillna(0)
    
    # 2. Compute Item Similarity using Cosine Similarity across all users
    item_similarity = cosine_similarity(item_user_matrix)
    item_sim_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)
    
    # 3. Get the most similar items to the target anime
    similar_items = item_sim_df[target_anime].sort_values(ascending=False)
    similar_items = similar_items.drop(target_anime) # Remove self
    
    # Filter down to items with a positive similarity score
    similar_items = similar_items[similar_items > 0]
    
    return list(similar_items.head(top_n).items())

def get_global_top_recommendations(top_n=10, min_ratings=3):
    """Generates global recommendations by finding the highest-rated anime across ALL users."""
    df = load_data()
    
    # Calculate average rating and number of ratings per anime
    stats = df.groupby('title').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    )
    
    # Filter for anime that have been rated by at least 'min_ratings' users
    qualified = stats[stats['rating_count'] >= min_ratings]
    top_anime = qualified.sort_values(by='avg_rating', ascending=False)
    
    # Return list of (title, avg_rating)
    return [(index, row['avg_rating']) for index, row in top_anime.head(top_n).iterrows()]

def export_cf_embeddings():
    """Exports the CF User-Item matrix as aligned embeddings for the main recommendation script."""
    import pickle
    import os
    
    print("Exporting CF embeddings aligned with anime_df.pkl...")
    if not os.path.exists('anime_df.pkl'):
        print("Error: anime_df.pkl not found. Run tagTrain.py first.")
        return
        
    anime_df = pd.read_pickle('anime_df.pkl')
    df = load_data()
    
    # Create matrix mapped by anime_id (instead of title for safe matching)
    df['anime_id'] = df['anime_id'].astype(str)
    item_user_matrix = df.pivot(index='anime_id', columns='user', values='rating').fillna(0)
    
    cf_embeddings = []
    num_users = len(item_user_matrix.columns)
    
    for _, row in anime_df.iterrows():
        aid = str(row['id']).replace('.0', '')
        if aid in item_user_matrix.index:
            cf_embeddings.append(item_user_matrix.loc[aid].values)
        else:
            cf_embeddings.append(np.zeros(num_users)) # Pad missing anime with 0s
            
    cf_embeddings = np.array(cf_embeddings)
    
    with open('cf_embeddings.pkl', 'wb') as f:
        pickle.dump(cf_embeddings, f)
        
    print(f"Exported CF embeddings for {len(cf_embeddings)} shows based on {num_users} users.")

if __name__ == "__main__":
    export_cf_embeddings()
    print("\n" + "="*50 + "\n")

    # Test it with one of the users we pulled data for in userdata.py
    test_user = "Stark700" 
    print(f"Generating Collaborative Filtering Recommendations for: {test_user}...\n")
    
    recs = get_collaborative_recommendations(test_user, top_n=10)
    for i, (title, score) in enumerate(recs, 1):
        print(f"{i}. {title} (Predicted Score: {score:.2f})")

    print("\n" + "="*50 + "\n")
    
    # Test Item-Item CF based on all users
    test_anime = "Steins;Gate" # Make sure this matches a title exactly in your DB
    print(f"Generating Item-Item Recommendations based on ALL users for: {test_anime}...\n")
    item_recs = get_item_collaborative_recommendations(test_anime, top_n=10)
    for i, (title, sim_score) in enumerate(item_recs, 1):
        print(f"{i}. {title} (Similarity Score: {sim_score:.2f})")

    print("\n" + "="*50 + "\n")

    # Test Global Popularity Recommendations
    print("Generating Global Top Anime across ALL users...\n")
    global_recs = get_global_top_recommendations(top_n=10, min_ratings=3)
    for i, (title, avg_rating) in enumerate(global_recs, 1):
        print(f"{i}. {title} (Average Rating: {avg_rating:.2f})")