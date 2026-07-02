import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import the match engine we trained in Phase 3
from phase3_primary_model import train_match_engine

def train_shootout_model(shootouts_file='./Dataset/shootouts.csv'):
    """
    Trains a secondary binary model to resolve knockout matches that end in a Draw.
    Features: Historical Shootout Win Rate of Team A vs Team B.
    """
    print("Training Secondary Shootout Model...")
    try:
        df = pd.read_csv(shootouts_file, parse_dates=['date'])
    except FileNotFoundError:
        print("shootouts.csv not found. Using coin-flip fallback for draws.")
        return None, defaultdict(lambda: 0.5)

    # Calculate historical shootout wins and total attempts per team
    wins = df['winner'].value_counts().to_dict()
    
    home_attempts = df['home_team'].value_counts().to_dict()
    away_attempts = df['away_team'].value_counts().to_dict()
    
    win_rates = {}
    all_teams = set(home_attempts.keys()).union(set(away_attempts.keys()))
    
    for team in all_teams:
        total_attempts = home_attempts.get(team, 0) + away_attempts.get(team, 0)
        total_wins = wins.get(team, 0)
        win_rates[team] = total_wins / total_attempts if total_attempts > 0 else 0.5
        
    # Create training data
    df['home_win_rate'] = df['home_team'].map(win_rates).fillna(0.5)
    df['away_win_rate'] = df['away_team'].map(win_rates).fillna(0.5)
    df['target'] = (df['winner'] == df['home_team']).astype(int)
    
    X = df[['home_win_rate', 'away_win_rate']]
    y = df['target']
    
    # Train a lightweight classifier
    shootout_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    shootout_model.fit(X, y)
    
    return shootout_model, win_rates

def extract_latest_stats(df, elo_file='./Dataset/eloratings.csv'):
    """
    Scans the historical dataset and Elo ratings to find the most recent 
    stats for every team so we can use them in the live simulation.
    """
    print("Extracting current team stats for simulation...")
    latest_stats = {}
    
    # Load raw Elo ratings to get the exact current strength of teams
    try:
        elo_df = pd.read_csv(elo_file, parse_dates=['date']).sort_values('date')
        elo_df['date'] = pd.to_datetime(elo_df['date'], errors='coerce')
        elo_df = elo_df.dropna(subset=['date'])
    except:
        elo_df = pd.DataFrame()
    
    df = df.sort_values('date')
    all_teams = set(df['home_team']).union(set(df['away_team']))
    
    for team in all_teams:
        # Get true absolute current Elo rating
        current_elo = 1500
        if not elo_df.empty:
            team_elo = elo_df[elo_df['team'] == team]
            if not team_elo.empty:
                current_elo = team_elo.iloc[-1]['rating']
                
        # Find their last match to approximate recent form/attack based on differentials
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
        if not team_matches.empty:
            last_match = team_matches.iloc[-1]
            if last_match['home_team'] == team:
                form = last_match['form_diff'] + 1.5
                attack = last_match['attack_diff'] + 1.0
            else:
                form = 1.5 - last_match['form_diff']
                attack = 1.0 - last_match['attack_diff']
        else:
            form, attack = 1.5, 1.0
            
        latest_stats[team] = {'elo': current_elo, 'form': form, 'attack': attack}
        
    return latest_stats

def simulate_match(team_A, team_B, match_model, shootout_model, win_rates, stats, is_neutral=1):
    """
    Simulates a single match between two teams probabilistically based on ML confidence.
    """
    stat_A = stats.get(team_A, {'elo': 1500, 'form': 1.5, 'attack': 1.0})
    stat_B = stats.get(team_B, {'elo': 1500, 'form': 1.5, 'attack': 1.0})
    
    features = pd.DataFrame([{
        'elo_diff': stat_A['elo'] - stat_B['elo'],
        'form_diff': stat_A['form'] - stat_B['form'],
        'attack_diff': stat_A['attack'] - stat_B['attack'],
        'is_home_advantage': 0 if is_neutral else 1
    }])
    
    # PROBABILISTIC FIX: Instead of getting the static prediction, get the probabilities!
    probs = match_model.predict_proba(features)[0]
    
    # Roll a weighted multi-sided dice based on those ML probabilities
    prediction = np.random.choice(match_model.classes_, p=probs)
    
    if prediction == 2:
        return team_A
    elif prediction == 0:
        return team_B
    else:
        # DRAW! We must go to penalty shootouts
        if shootout_model is None:
            return team_A if np.random.rand() > 0.5 else team_B
            
        shootout_features = pd.DataFrame([{
            'home_win_rate': win_rates.get(team_A, 0.5),
            'away_win_rate': win_rates.get(team_B, 0.5)
        }])
        
        # Make shootouts probabilistic too!
        pk_probs = shootout_model.predict_proba(shootout_features)[0]
        pk_pred = np.random.choice(shootout_model.classes_, p=pk_probs)
        return team_A if pk_pred == 1 else team_B

def run_monte_carlo(bracket, match_model, shootout_model, win_rates, stats, iterations=1000):
    """
    Simulates the knockout bracket thousands of times to find championship probabilities.
    """
    print(f"\n--- Running Monte Carlo Simulator ({iterations} Iterations) ---")
    championships = defaultdict(int)
    
    for i in range(iterations):
        current_round = bracket.copy()
        
        while len(current_round) > 1:
            next_round = []
            for j in range(0, len(current_round), 2):
                winner = simulate_match(current_round[j], current_round[j+1], match_model, shootout_model, win_rates, stats)
                next_round.append(winner)
            current_round = next_round
            
        championships[current_round[0]] += 1
        
        if (i + 1) % 250 == 0:
            print(f"Completed {i + 1} simulations...")
            
    results = {team: (wins / iterations) * 100 for team, wins in championships.items()}
    
    print("\n🏆 TOURNAMENT WIN PROBABILITIES 🏆")
    print("-" * 35)
    for team, prob in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{team:<15} : {prob:>5.1f}% chance to win")
    print("-" * 35)

if __name__ == "__main__":
    match_model, final_df = train_match_engine()
    team_stats = extract_latest_stats(final_df)
    shootout_model, win_rates = train_shootout_model()
    
    world_cup_bracket = [
        "Argentina", "Australia", 
        "Netherlands", "United States",
        "France", "Poland",
        "England", "Senegal",
        "Japan", "Croatia",
        "Brazil", "South Korea",
        "Morocco", "Spain",
        "Portugal", "Switzerland"
    ]
    
    run_monte_carlo(
        bracket=world_cup_bracket, 
        match_model=match_model, 
        shootout_model=shootout_model, 
        win_rates=win_rates, 
        stats=team_stats, 
        iterations=1000
    )