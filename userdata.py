import requests
import pandas as pd
import time
import sqlite3

# 1. Paste the exact access token string you received here
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjBiYWM0MmZkYmVkYjgzOGM3ZmRlZTFiYmM2NmQxMTdjZjk1MWE2YzcyMzc3NGY5ZmExZjMyMjdiMzk2ZmI1NDBjNzlkMmUwNzEzNDI0MzM5In0.eyJhdWQiOiJkZmZjN2NhNThjNWM3MTYyMjI4ZWUyMGYxMmEyNjk1NyIsImp0aSI6IjBiYWM0MmZkYmVkYjgzOGM3ZmRlZTFiYmM2NmQxMTdjZjk1MWE2YzcyMzc3NGY5ZmExZjMyMjdiMzk2ZmI1NDBjNzlkMmUwNzEzNDI0MzM5IiwiaWF0IjoxNzc5MDI4MjUwLCJuYmYiOjE3NzkwMjgyNTAsImV4cCI6MTc4MTcwNjY1MCwic3ViIjoiODMyMjI5NyIsInNjb3BlcyI6W119.hOX--GkcKKVzZR17UXbJ8fktjhFJQ98oVFx-YdosFtT9yIIqN9q0kmnEmW9D3re4q4P6FZEeTkm7jTCKEaykemjfs6abGQRI9dU9QZgoHOzaNRoT5KdHmDKzrorbVMQE1hvWTusdC8cZDTBVQxuR8DzL4BfN1UcVdVfZ0WDBbvpte24mynucqaM48WKd5qbmqCFwGU2HfxXIMyWrQblgkT-BugIRqlxoXuQ5mV2ZoZJibZWurvvQbsp7RMPwK0Q5mzSjsG-Leo_PkgyEdiwo0_gv5-SAnJ-lbpW_LDBkLeCVLE2c78oHhvoxVJg2vsqPC1AqBoa8YF8UzZlt4TlOOQ"

headers = {
    'Authorization': f"Bearer {ACCESS_TOKEN}"
}

# 2. Set the parameters. We request 'list_status' to get the user's 1-10 scores.
params = {
    'limit': 1000,           # MAL allows up to 1000 per request
    'fields': 'list_status', # Required to see the user's ratings
    'status': 'completed'    # Only fetch shows they have finished
}

target_users = [
    "abystoma2", "Stark700", "Disapeared_Ghost", "NextUniverse", "Yemi_Hikari", 
    "JVskunkape", "BohemianRhapsody", "iNinjeek", "lancelot200", "AnimeFan500", 
    "JuanShot87", "Short_Circut", "JiangHaoyi1979", "ST63LTH", "WingZeroRequiem", 
    "TheRockzSG", "Xaeveax", "MagicFlier", "eblf2013", "IridescentJaune", 
    "Hellspawn28", "Lemon", "RenaPsychoKiller", "Orulyon", "inim", 
    "BonBonToro", "bunny1ov3r", "Shura-shurato", "ImNotAnOtaku1", "iwsifakismag"
    "Kineta", "tingy", "Tensho", "Leknaat", "cyruz", "Denji", "Jinbou", "Vindshot",
    "Aidan", "Aquarius", "azazel_lns", "BlizZar", "goncix3000", "Jicetus", "LittleKitty24", "MagicPolly"
    "Mythologically", "Naruleach", "Red-Comet", "Timcampy", "Xealias", "-Mitsukii", "Dancho", "Dreik55", "iShipMyFriends"
    "Katayone", "KTGita22", "KuroDubZero", "Meedor", "Phloup", "phren", "rlavus", "stinghy", "CC", "Fluffygreygrass", "Karinara", "KayKimii", "KOTFTW", "Kunii", "Mako995", "ScaryOwl"
    "SushiRoe", "Twintail_Daemon", "Aiimee", "cyruz", "DatRandomDude", "Kineta", "mobulidae", "Naitik7897", "nirererin", "Snow", "Syureria", "SyverenWaterlow", "Vindstot", "ACasualViewer"
    "AlexPaulLEWZ", "danyelll", "gaussian_", "kekekeKaj", "LiIlie", "Maffy", "Pipe", "Cestlavie_", "cmanjakaa", "CukieMunster", "lil-pie", "Rosaluna_", "ToumaTachibana", "beeped5", "KawaiiTendou", "Kookerz"
]

all_user_ratings = []

for username in target_users:
    print(f"Fetching data for user: {username}...")
    api_url = f"https://api.myanimelist.net/v2/users/{username}/animelist"
    
    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code == 200:
        raw_data = response.json()
        
        user_count = 0
        for item in raw_data.get('data', []):
            anime_id = item['node']['id']
            title = item['node']['title']
            score = item['list_status'].get('score', 0)
            
            # Only keep shows the user actually gave a 1-10 rating to
            if score > 0:
                all_user_ratings.append({
                    'user': username,
                    'anime_id': anime_id,
                    'title': title,
                    'rating': score
                })
                user_count += 1
        print(f"  -> Added {user_count} rated shows.")
    else:
        print(f"  -> Failed to fetch data. Error {response.status_code}")
        
    # Sleep for 1 second between requests to respect MAL's rate limits
    time.sleep(1)

if all_user_ratings:
    # Convert everything to a single Pandas DataFrame
    df = pd.DataFrame(all_user_ratings)
    
    print("\nSuccess! Here is a preview of the structured data:")
    print(df.head())
    print(f"\nTotal rated shows found across all users: {len(df)}")

    # Write the combined DataFrame directly into the SQL table
    conn = sqlite3.connect('anime_database.db')
    df.to_sql(name='user_ratings', con=conn, if_exists='append', index=False)
    print("Data successfully saved to the database!")
    conn.close()
else:
    print("No data was collected.")