import requests
import json
import sys
from textwrap import wrap


ANILIST_API = "https://graphql.anilist.co"


# ── GraphQL Queries ────────────────────────────────────────────────────────────

SEARCH_QUERY = """
query ($search: String, $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
    }
    media(search: $search, type: ANIME, sort: SEARCH_MATCH) {
      id
      title {
        romaji
        english
        native
      }
      format
      status
      episodes
      duration
      season
      seasonYear
      averageScore
      popularity
      genres
      description(asHtml: false)
    }
  }
}
"""

DETAILS_QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    id
    title {
      romaji
      english
      native
    }
    format
    status
    episodes
    duration
    season
    seasonYear
    startDate { year month day }
    endDate   { year month day }
    averageScore
    meanScore
    popularity
    favourites
    genres
    tags { name rank isMediaSpoiler }
    studios(isMain: true) { nodes { name } }
    description(asHtml: false)
    siteUrl
    rankings {
      rank
      type
      context
      allTime
      season
      year
    }
    relations {
      edges {
        relationType
        node {
          title { romaji }
          format
          status
        }
      }
    }
    characters(sort: ROLE, perPage: 6) {
      edges {
        role
        node { name { full } }
        voiceActors(language: JAPANESE) { name { full } }
      }
    }
    nextAiringEpisode {
      episode
      airingAt
    }
  }
}
"""

TRENDING_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, sort: TRENDING_DESC) {
      id
      title { romaji english }
      format
      status
      episodes
      averageScore
      popularity
      genres
    }
  }
}
"""

TOP_RATED_QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, sort: SCORE_DESC, format_in: [TV, MOVIE, OVA]) {
      id
      title { romaji english }
      format
      status
      episodes
      averageScore
      popularity
      genres
    }
  }
}
"""


# ── API Helper ─────────────────────────────────────────────────────────────────

def query_anilist(query: str, variables: dict) -> dict:
    """Send a GraphQL query to AniList and return the data."""
    response = requests.post(
        ANILIST_API,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()
    if "errors" in result:
        raise ValueError(f"AniList API error: {result['errors']}")
    return result["data"]


# ── Formatters ─────────────────────────────────────────────────────────────────

def format_date(date: dict | None) -> str:
    if not date or not date.get("year"):
        return "Unknown"
    parts = [str(date["year"])]
    if date.get("month"):
        parts.append(f"{date['month']:02d}")
    if date.get("day"):
        parts.append(f"{date['day']:02d}")
    return "-".join(parts)


def clean_description(text: str | None, max_chars: int = 400) -> str:
    if not text:
        return "No description available."
    # Remove common HTML remnants
    for tag in ["<br>", "<br/>", "<i>", "</i>", "<b>", "</b>"]:
        text = text.replace(tag, " ")
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def score_bar(score: int | None, width: int = 20) -> str:
    if score is None:
        return "N/A"
    filled = round((score / 100) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score}/100"


def print_separator(char: str = "─", width: int = 60):
    print(char * width)


def print_anime_card(anime: dict, index: int | None = None):
    """Print a compact summary card for a single anime."""
    titles = anime.get("title", {})
    title = titles.get("english") or titles.get("romaji") or "Unknown Title"
    romaji = titles.get("romaji", "")

    prefix = f"[{index}] " if index is not None else ""
    print(f"\n{prefix} {title}")
    if romaji and romaji != title:
        print(f"    ({romaji})")

    meta = []
    if anime.get("format"):
        meta.append(anime["format"])
    if anime.get("seasonYear"):
        season = anime.get("season", "")
        meta.append(f"{season} {anime['seasonYear']}".strip())
    if anime.get("episodes"):
        meta.append(f"{anime['episodes']} eps")
    if anime.get("duration"):
        meta.append(f"{anime['duration']}min/ep")
    if meta:
        print(f"    {' · '.join(meta)}")

    if anime.get("averageScore"):
        print(f"    Score: {score_bar(anime['averageScore'])}")

    if anime.get("genres"):
        print(f"    Genres: {', '.join(anime['genres'][:5])}")

    status = anime.get("status", "").replace("_", " ").title()
    if status:
        print(f"    Status: {status}")


def print_anime_details(anime: dict):
    """Print full details for a single anime."""
    titles = anime.get("title", {})
    title = titles.get("english") or titles.get("romaji") or "Unknown"

    print_separator("═")
    print(f"  {title}")
    if titles.get("romaji") and titles["romaji"] != title:
        print(f"  {titles['romaji']}")
    if titles.get("native"):
        print(f"  {titles['native']}")
    print_separator("═")

    # Core info
    meta_pairs = [
        ("Format",    (anime.get("format") or "").replace("_", " ")),
        ("Status",    (anime.get("status") or "").replace("_", " ").title()),
        ("Season",    f"{anime.get('season','')} {anime.get('seasonYear','')}".strip()),
        ("Episodes",  str(anime.get("episodes") or "?")),
        ("Duration",  f"{anime['duration']} min/ep" if anime.get("duration") else ""),
        ("Air Start", format_date(anime.get("startDate"))),
        ("Air End",   format_date(anime.get("endDate"))),
    ]
    for label, value in meta_pairs:
        if value:
            print(f"  {label:<12} {value}")

    print()

    # Scores
    if anime.get("averageScore"):
        print(f"  Score     {score_bar(anime['averageScore'])}")
    if anime.get("popularity"):
        print(f"  Popularity  {anime['popularity']:,} users")
    if anime.get("favourites"):
        print(f"  Favourites  {anime['favourites']:,}")

    print()

    # Studios
    studios = [s["name"] for s in (anime.get("studios", {}).get("nodes") or [])]
    if studios:
        print(f"  Studio(s): {', '.join(studios)}")

    # Genres
    if anime.get("genres"):
        print(f"  Genres:    {', '.join(anime['genres'])}")

    # Tags (non-spoiler, top 5)
    tags = [t["name"] for t in (anime.get("tags") or []) if not t.get("isMediaSpoiler")][:5]
    if tags:
        print(f"  Tags:      {', '.join(tags)}")

    # Rankings
    rankings = anime.get("rankings") or []
    top_rankings = [r for r in rankings if r["rank"] <= 50][:3]
    if top_rankings:
        print("\n   Rankings:")
        for r in top_rankings:
            scope = "All Time" if r.get("allTime") else f"{r.get('season','')} {r.get('year','')}".strip()
            print(f"    #{r['rank']} {r['context'].title()} ({scope})")

    # Next airing
    nae = anime.get("nextAiringEpisode")
    if nae:
        import datetime
        air_time = datetime.datetime.fromtimestamp(nae["airingAt"]).strftime("%Y-%m-%d %H:%M")
        print(f"\n   Next Episode: Ep {nae['episode']} at {air_time}")

    # Description
    print("\n   Synopsis:")
    desc = clean_description(anime.get("description"), max_chars=600)
    for line in wrap(desc, width=56):
        print(f"    {line}")

    # Characters
    chars = anime.get("characters", {}).get("edges") or []
    if chars:
        print("\n   Characters:")
        for edge in chars:
            char_name = edge["node"]["name"]["full"]
            role = edge.get("role", "").title()
            vas = [v["name"]["full"] for v in (edge.get("voiceActors") or [])]
            va_str = f" (VA: {vas[0]})" if vas else ""
            print(f"    {role:<12} {char_name}{va_str}")

    # Relations
    relations = anime.get("relations", {}).get("edges") or []
    if relations:
        print("\n  🔗 Related:")
        for edge in relations[:6]:
            rel_type = edge.get("relationType", "").replace("_", " ").title()
            node = edge["node"]
            rel_title = node["title"]["romaji"]
            rel_fmt = (node.get("format") or "").replace("_", " ")
            print(f"    {rel_type:<14} {rel_title} [{rel_fmt}]")

    if anime.get("siteUrl"):
        print(f"\n  {anime['siteUrl']}")

    print_separator("═")


# ── Feature Functions ──────────────────────────────────────────────────────────

def search_anime(query: str, per_page: int = 10):
    """Search anime by name and display results."""
    print(f"\n Searching for: \"{query}\"")
    print_separator()

    data = query_anilist(SEARCH_QUERY, {"search": query, "page": 1, "perPage": per_page})
    page = data["Page"]
    results = page["media"]

    if not results:
        print("No results found.")
        return []

    total = page["pageInfo"]["total"]
    print(f"Found {total} result(s). Showing top {len(results)}:\n")

    for i, anime in enumerate(results, 1):
        print_anime_card(anime, index=i)

    return results


def get_anime_details(anime_id: int):
    """Fetch and display full details for an anime by ID."""
    print(f"\n Fetching details for ID: {anime_id}")
    data = query_anilist(DETAILS_QUERY, {"id": anime_id})
    print_anime_details(data["Media"])


def show_trending(per_page: int = 10):
    """Display currently trending anime."""
    print("\n Trending Anime Right Now")
    print_separator()
    data = query_anilist(TRENDING_QUERY, {"page": 1, "perPage": per_page})
    for i, anime in enumerate(data["Page"]["media"], 1):
        print_anime_card(anime, index=i)


def show_top_rated(per_page: int = 10):
    """Display top-rated anime of all time."""
    print("\n Top Rated Anime (All Time)")
    print_separator()
    data = query_anilist(TOP_RATED_QUERY, {"page": 1, "perPage": per_page})
    for i, anime in enumerate(data["Page"]["media"], 1):
        print_anime_card(anime, index=i)


# ── Interactive CLI ────────────────────────────────────────────────────────────

def interactive_menu():
    """Run an interactive search session."""
    print("\n" + "═" * 60)
    print("    AniList Anime Search")
    print("═" * 60)

    while True:
        print("\nOptions:")
        print("  1. Search anime by name")
        print("  2. Get details by AniList ID")
        print("  3. Trending anime")
        print("  4. Top rated anime")
        print("  q. Quit")
        print_separator()

        choice = input("Choose an option: ").strip().lower()

        if choice == "q":
            print("Sayonara! ")
            break

        elif choice == "1":
            query = input("Enter anime name: ").strip()
            if not query:
                continue
            results = search_anime(query)
            if results:
                pick = input("\nEnter a number to see full details (or press Enter to skip): ").strip()
                if pick.isdigit():
                    idx = int(pick) - 1
                    if 0 <= idx < len(results):
                        get_anime_details(results[idx]["id"])

        elif choice == "2":
            anime_id = input("Enter AniList anime ID: ").strip()
            if anime_id.isdigit():
                get_anime_details(int(anime_id))
            else:
                print("Invalid ID.")

        elif choice == "3":
            show_trending()
            pick = input("\nEnter a number to see full details (or press Enter to skip): ").strip()
            # We'd need to store results — for simplicity, prompt for ID
            if pick.isdigit():
                print("(Tip: use option 2 with the ID shown on AniList to get details)")

        elif choice == "4":
            show_top_rated()

        else:
            print("Unknown option.")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Quick CLI: python anilist_search.py "attack on titan"
        term = " ".join(sys.argv[1:])
        results = search_anime(term)
        if results and len(results) == 1:
            get_anime_details(results[0]["id"])
    else:
        interactive_menu()