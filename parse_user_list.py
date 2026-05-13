import requests
import re
import csv
import sys

ANILIST_API = "https://graphql.anilist.co"

# ── GraphQL Query ──────────────────────────────────────────────────────────────
# This query fetches a user's entire anime list collection.
# We retrieve the list names (e.g., "Completed", "Planning"), user scores, 
# progress, and media details (title, genres, format, average score).
USER_LIST_QUERY = """
query ($userName: String) {
  MediaListCollection(userName: $userName, type: ANIME) {
    lists {
      name
      entries {
        status
        score
        progress
        media {
          id
          title {
            romaji
            english
          }
          format
          episodes
          averageScore
          genres
        }
      }
    }
  }
}
"""

def extract_username(url: str) -> str:
    """Extracts the AniList username from a given profile URL."""
    # Matches patterns like https://anilist.co/user/Username/
    match = re.search(r'anilist\.co/user/([^/]+)', url, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""

def fetch_user_list(username: str) -> dict:
    """Fetches the user's anime list from the AniList API."""
    print(f"Fetching anime list for user: {username}...")
    variables = {"userName": username}
    
    response = requests.post(
        ANILIST_API,
        json={"query": USER_LIST_QUERY, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=15
    )
    
    if response.status_code == 404:
        raise ValueError("User not found. Please check the profile link.")
    
    response.raise_for_status()
    result = response.json()
    
    if "errors" in result:
        raise ValueError(f"AniList API error: {result['errors'][0]['message']}")
        
    return result["data"]

def save_to_csv(data: dict, username: str):
    """Parses the API response and saves it to a CSV file."""
    filename = f"{username}_anilist.csv"
    
    fieldnames = [
        "list_name", "user_status", "media_id", "title_english", "title_romaji", 
        "user_score", "progress", "episodes", "format", "average_score", "genres"
    ]
    
    lists = data.get("MediaListCollection", {}).get("lists", [])
    
    if not lists:
        print("This user has no anime lists or their lists are private.")
        return

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        total_entries = 0
        for anime_list in lists:
            list_name = anime_list.get("name", "Unknown")
            
            for entry in anime_list.get("entries", []):
                media = entry.get("media", {})
                titles = media.get("title", {})
                genres = "|".join(media.get("genres", []))
                
                writer.writerow({
                    "list_name": list_name,
                    "user_status": entry.get("status", ""),
                    "media_id": media.get("id", ""),
                    "title_english": titles.get("english", ""),
                    "title_romaji": titles.get("romaji", ""),
                    "user_score": entry.get("score", ""),
                    "progress": entry.get("progress", ""),
                    "episodes": media.get("episodes", ""),
                    "format": media.get("format", ""),
                    "average_score": media.get("averageScore", ""),
                    "genres": genres
                })
                total_entries += 1
                
    print(f"Success! Saved {total_entries} anime entries to {filename}.")

if __name__ == "__main__":
    print("=== AniList User Data Parser ===")
    profile_url = input("Enter the AniList profile link (e.g., https://anilist.co/user/Username/): ").strip()
    
    username = extract_username(profile_url)
    if not username:
        print("Error: Could not extract a valid username from the provided URL.")
        sys.exit(1)
        
    try:
        user_data = fetch_user_list(username)
        save_to_csv(user_data, username)
    except Exception as e:
        print(f"An error occurred: {e}")