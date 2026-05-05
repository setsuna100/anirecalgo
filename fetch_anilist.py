import requests
import csv
import time

ANILIST_URL = "https://graphql.anilist.co"

QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
    }
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      title {
        romaji
        english
        native
      }
      averageScore
      meanScore
      popularity
      favourites
      genres
      tags {
        name
        category
        rank
        isMediaSpoiler
      }
      status
      episodes
      season
      seasonYear
      format
      studios(isMain: true) {
        nodes {
          name
        }
      }
    }
  }
}
"""

def fetch_page(page, per_page=50):
    variables = {"page": page, "perPage": per_page}
    response = requests.post(
        ANILIST_URL,
        json={"query": QUERY, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]["Page"]

def fetch_top_1000():
    all_anime = []
    per_page = 50
    total_needed = 1000
    page = 1

    print("Fetching top 1000 anime from AniList...")

    while len(all_anime) < total_needed:
        print(f"  Fetching page {page} ({len(all_anime)}/{total_needed})...")
        page_data = fetch_page(page, per_page)
        media = page_data["media"]

        if not media:
            print("No more results returned.")
            break

        all_anime.extend(media)

        if not page_data["pageInfo"]["hasNextPage"]:
            print("Reached last page.")
            break

        page += 1
        # Respect AniList rate limit: 90 requests/minute
        time.sleep(0.7)

    return all_anime[:total_needed]

def build_row(anime):
    title_english = anime["title"].get("english") or ""
    title_romaji = anime["title"].get("romaji") or ""
    title_native = anime["title"].get("native") or ""

    genres = "|".join(anime.get("genres") or [])

    # Filter out spoiler tags and sort by rank descending
    tags = anime.get("tags") or []
    non_spoiler_tags = [t for t in tags if not t.get("isMediaSpoiler")]
    non_spoiler_tags.sort(key=lambda t: t.get("rank", 0), reverse=True)
    tag_names = "|".join(t["name"] for t in non_spoiler_tags)
    tag_categories = "|".join(t["category"] for t in non_spoiler_tags)

    studios = anime.get("studios", {}).get("nodes") or []
    main_studio = studios[0]["name"] if studios else ""

    return {
        "id": anime["id"],
        "title_english": title_english,
        "title_romaji": title_romaji,
        "title_native": title_native,
        "format": anime.get("format") or "",
        "status": anime.get("status") or "",
        "episodes": anime.get("episodes") or "",
        "season": anime.get("season") or "",
        "season_year": anime.get("seasonYear") or "",
        "average_score": anime.get("averageScore") or "",
        "mean_score": anime.get("meanScore") or "",
        "popularity": anime.get("popularity") or "",
        "favourites": anime.get("favourites") or "",
        "genres": genres,
        "tags": tag_names,
        "tag_categories": tag_categories,
        "main_studio": main_studio,
    }

def export_csv(anime_list, output_path):
    fieldnames = [
        "id", "title_english", "title_romaji", "title_native",
        "format", "status", "episodes", "season", "season_year",
        "average_score", "mean_score", "popularity", "favourites",
        "genres", "tags", "tag_categories", "main_studio",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for anime in anime_list:
            writer.writerow(build_row(anime))

    print(f"\nExported {len(anime_list)} anime to {output_path}")

if __name__ == "__main__":
    anime_list = fetch_top_1000()
    output_path = "top1000.csv"
    export_csv(anime_list, output_path)