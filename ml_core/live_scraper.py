import requests

def fetch_live_elo(team_name):
    """
    Reaches out to the internet to get the latest Elo rating.
    Now uses a 2-step process: 
    1. Look up the 2-letter team code.
    2. Look up the rating using that code.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/plain, */*; q=0.01',
        'Referer': 'https://www.eloratings.net/'
    }
    
    try:
        # --- STEP 1: Find the 2-letter country code ---
        teams_url = "https://www.eloratings.net/en.teams.tsv"
        teams_response = requests.get(teams_url, headers=headers, timeout=5)
        teams_response.raise_for_status()
        
        team_code = None
        for line in teams_response.text.split('\n'):
            if not line.strip(): continue
            parts = line.split('\t')
            
            # Format is: 'AR' \t 'Argentina'
            if len(parts) >= 2:
                code = parts[0].strip()
                name = parts[1].strip()
                
                # Compare in lowercase to avoid capitalization issues
                if name.lower() == team_name.lower():
                    team_code = code
                    break
                    
        if not team_code:
            print(f"⚠️ Live Scrape: Could not find a 2-letter code for '{team_name}'.")
            return None

        # --- STEP 2: Fetch the live rating using the code ---
        world_url = "https://www.eloratings.net/World.tsv"
        world_response = requests.get(world_url, headers=headers, timeout=5)
        world_response.raise_for_status()
        
        for line in world_response.text.split('\n'):
            if not line.strip(): continue
            parts = line.split('\t')
            
            # Format is: Rank \t RankChange \t Code \t Rating
            # Example: '1 \t 1 \t AR \t 2151'
            if len(parts) >= 4:
                current_code = parts[2].strip()
                
                if current_code == team_code:
                    try:
                        rating = float(parts[3].strip())
                        print(f"🌐 LIVE SCRAPE SUCCESS: {team_name} ({team_code}) is currently at {rating} Elo.")
                        return rating
                    except ValueError:
                        continue # Skip if rating isn't a valid number
                        
        print(f"⚠️ Live Scrape: Found code '{team_code}' but no rating in World.tsv.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Live Scrape Network Error: {e}")
        return None

# --- LOCAL TEST BLOCK ---
if __name__ == "__main__":
    print("--- Testing Live Scraper ---")
    
    print("Fetching Argentina...")
    argentina_score = fetch_live_elo("Argentina")
    
    print("\nFetching Portugal...")
    portugal_score = fetch_live_elo("Portugal")
    
    print("\nFetching Atlantis (Fake Team)...")
    fake_score = fetch_live_elo("Atlantis")
    
    print("\n--- Test Complete ---")