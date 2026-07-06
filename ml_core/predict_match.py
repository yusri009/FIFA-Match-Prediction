import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import the tools we built in Phase 3 and Phase 4
from phase3_primary_model import train_match_engine
from phase4_tournament_simulator import extract_latest_stats

def predict_tonights_match(home_team, away_team, is_neutral=0):
    """
    Predicts the exact probabilities of a single 90-minute match 
    using the trained Random Forest model.
    """
    print(f"--- GATHERING DATA FOR {home_team.upper()} VS {away_team.upper()} ---")
    
    # 1. Train the match engine and get the historical dataset
    # (This runs fast since the data is already cleaned)
    match_model, final_df = train_match_engine()
    
    # 2. Extract the absolute latest stats (Current Elo, Recent Form, Recent Attack)
    stats = extract_latest_stats(final_df)
    
    # 3. Retrieve stats specifically for Portugal and Croatia
    stat_home = stats.get(home_team, {'elo': 1500, 'form': 1.5, 'attack': 1.0})
    stat_away = stats.get(away_team, {'elo': 1500, 'form': 1.5, 'attack': 1.0})
    
    # 4. Calculate the Match Differentials (The context for the ML model)
    features = pd.DataFrame([{
        'elo_diff': stat_home['elo'] - stat_away['elo'],
        'form_diff': stat_home['form'] - stat_away['form'],
        'attack_diff': stat_home['attack'] - stat_away['attack'],
        'is_home_advantage': 0 if is_neutral else 1
    }])
    
    # 5. Get the Probabilities!
    # predict_proba returns an array of probabilities for classes [0, 1, 2]
    # Class 0 = Away Win, Class 1 = Draw, Class 2 = Home Win
    probs = match_model.predict_proba(features)[0]
    
    away_win_prob = probs[0] * 100
    draw_prob = probs[1] * 100
    home_win_prob = probs[2] * 100
    
    # 6. Display the Results
    print("\n" + "="*40)
    print("🔮 LIVE MATCH PREDICTION 🔮")
    print("="*40)
    print(f"Match Context:")
    print(f"Location: {'Neutral Venue' if is_neutral else f'{home_team} Home Advantage'}")
    print(f"Elo Gap:  {features['elo_diff'][0]:.1f} points in favor of {'Home' if features['elo_diff'][0] > 0 else 'Away'}")
    
    print("\n90-Minute Probabilities:")
    print(f"{home_team:<15} Win : {home_win_prob:>5.1f}%")
    print(f"{'Draw':<15}     : {draw_prob:>5.1f}%")
    print(f"{away_team:<15} Win : {away_win_prob:>5.1f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # If the match is happening in Portugal, is_neutral=0. 
    # If it's a Euro/World Cup match in Germany/USA, change is_neutral=1
    predict_tonights_match("Croatia","Portugal",  is_neutral=0)
    