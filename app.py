from flask import Flask, request, jsonify, render_template
from user_recommend import fetch_user_list_api, get_recommendations_for_user
from recommend import get_recommendations
import pandas as pd

# Set template_folder='.' and static_folder='.' so Flask looks in the current directory
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='/static')

# Serve the frontend HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    username = data.get('username')
    allow_ecchi = data.get('allow_ecchi', False)
    
    if not username:
        return jsonify({"error": "Username is required"}), 400
        
    try:
        # Fetch data and generate recommendations using existing logic
        user_data = fetch_user_list_api(username)
        if not user_data:
            return jsonify({"error": "No user data found or list is private."}), 404
            
        recs = get_recommendations_for_user(user_data, top_n=10, allow_ecchi=allow_ecchi)
        return jsonify({"recommendations": recs})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint to get text-based recommendations
@app.route('/text_recommend', methods=['POST'])
def text_recommend():
    data = request.json
    query = data.get('query')
    use_spoilers = data.get('use_spoilers', False)
    allow_ecchi = data.get('allow_ecchi', False)
    
    if not query:
        return jsonify({"error": "Search query is required"}), 400
        
    try:
        # Call your local NLP model logic
        recs_df = get_recommendations(query, top_n=10, use_spoilers=use_spoilers, allow_ecchi=allow_ecchi)
        
        recs = []
        for _, row in recs_df.iterrows():
            t_eng = str(row['title_english']) if pd.notna(row['title_english']) and row['title_english'] else ""
            t_rom = str(row['title_romaji']) if pd.notna(row['title_romaji']) and row['title_romaji'] else ""
            title = t_eng if t_eng else t_rom
            
            genres = str(row['genres']).replace("|", ", ") if pd.notna(row['genres']) else "Unknown"
            fmt = row['format'] if pd.notna(row['format']) else "Unknown"
            
            episodes = ""
            if pd.notna(row['episodes']) and str(row['episodes']).strip():
                try:
                    episodes = int(float(row['episodes']))
                except ValueError:
                    episodes = row['episodes']
            score = row['mean_score'] if pd.notna(row['mean_score']) else "N/A"
            
            year = ""
            if pd.notna(row['season_year']) and str(row['season_year']).strip():
                try:
                    year = int(float(row['season_year']))
                except ValueError:
                    year = row['season_year']
            
            match_score = f"{row['match_score'] * 100:.1f}%" if 'match_score' in row else "N/A"
            
            recs.append({
                "title": title,
                "format": fmt,
                "episodes": episodes,
                "score": score,
                "genres": genres,
                "year": year,
                "match_score": match_score
            })
            
        return jsonify({"recommendations": recs})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)