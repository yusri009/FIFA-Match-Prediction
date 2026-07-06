import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import the harmonization function we built in Phase 1
from phase1_harmonization import harmonize_data

def calculate_team_history(df, goalscorers_file='./Dataset/goalscorers.csv'):
    """
    Transforms match data into a team-centric timeline to calculate rolling form,
    and merges in advanced stats from the goalscorers dataset.
    """
    print("Calculating rolling form and advanced attacking metrics...")
    
    # 1. Split match data into a team-by-team timeline
    home = df[['date', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'goals_for', 'away_score': 'goals_against'})
    away = df[['date', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'goals_for', 'home_score': 'goals_against'})
    
    team_history = pd.concat([home, away]).sort_values(by=['team', 'date'])
    
    # Calculate base points (3 for Win, 1 for Draw, 0 for Loss)
    team_history['points'] = np.where(team_history['goals_for'] > team_history['goals_against'], 3,
                             np.where(team_history['goals_for'] == team_history['goals_against'], 1, 0))
    
    # 2. Integrate Advanced Tactics (Goalscorers Dataset)
    # We want to know how many non-penalty goals a team is scoring to measure true attacking threat
    try:
        scorers_df = pd.read_csv(goalscorers_file, parse_dates=['date'])
        # Count non-penalty goals per team per match
        open_play_goals = scorers_df[scorers_df['penalty'] == False].groupby(['date', 'team']).size().reset_index(name='open_play_goals')
        team_history = pd.merge(team_history, open_play_goals, on=['date', 'team'], how='left')
        team_history['open_play_goals'] = team_history['open_play_goals'].fillna(0)
    except FileNotFoundError:
        print(f"Warning: {goalscorers_file} not found. Skipping advanced attacking metrics.")
        team_history['open_play_goals'] = team_history['goals_for'] # Fallback
        
    # 3. Calculate Rolling Averages (The "Form" context)
    # CRITICAL: .shift(1) prevents data leakage by excluding the current match from the average
    grouped = team_history.groupby('team')
    
    # Average points in the last 5 games
    team_history['form_5'] = grouped['points'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    # Total open play goals scored in the last 5 games
    team_history['attacking_momentum_5'] = grouped['open_play_goals'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum())
    
    return team_history[['date', 'team', 'form_5', 'attacking_momentum_5']]

def engineer_features(results_file='./Dataset/results.csv', names_file='./Dataset/former_names.csv', elo_file='./Dataset/eloratings.csv', scorers_file='./Dataset/goalscorers.csv'):
    """
    Executes Phase 2: Merges clean data with Elo and Form to create ML differentials.
    """
    print("--- Starting Phase 2: Feature Engineering ---")
    
    # Step 1: Get the clean baseline data from Phase 1
    df = harmonize_data(results_file, names_file, start_year=1990)
    
    # Filter out missing scores (e.g., future fixtures that haven't been played yet)
    df = df.dropna(subset=['home_score', 'away_score']).copy()
    
    # Target Variable: 0 = Away Win, 1 = Draw, 2 = Home Win
    conditions = [df['home_score'] > df['away_score'], df['home_score'] == df['away_score']]
    df['target'] = np.select(conditions, [2, 1], default=0)
    df['is_home_advantage'] = (~df['neutral']).astype(int)

    # Step 2: Calculate and merge rolling form/tactics
    history_df = calculate_team_history(df, scorers_file)
    
    df = pd.merge(df, history_df, left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').drop(columns=['team'])
    df = df.rename(columns={'form_5': 'home_form', 'attacking_momentum_5': 'home_attack'})
    
    df = pd.merge(df, history_df, left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').drop(columns=['team'])
    df = df.rename(columns={'form_5': 'away_form', 'attacking_momentum_5': 'away_attack'})

    # Step 3: Merge historical Elo Ratings
    print("Merging Historical Elo ratings...")
    elo_df = pd.read_csv(elo_file, parse_dates=['date']).sort_values('date')
    
    # FORCE DATETIME CONVERSION HERE to fix the dtype('O') error
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elo_df['date'] = pd.to_datetime(elo_df['date'], errors='coerce')
    
    # DROP THE NULL DATES caused by 'coerce' so merge_asof doesn't crash
    df = df.dropna(subset=['date'])
    elo_df = elo_df.dropna(subset=['date'])
    
    df = df.sort_values('date')
    elo_df = elo_df.sort_values('date')
    
    # Find the most recent Elo rating for both teams BEFORE this match happened
    df = pd.merge_asof(df, elo_df, left_on='date', left_by='home_team', right_on='date', right_by='team', direction='backward')
    df = df.rename(columns={'rating': 'home_elo'}).drop(columns=['team', 'change'])
    
    df = pd.merge_asof(df, elo_df, left_on='date', left_by='away_team', right_on='date', right_by='team', direction='backward')
    df = df.rename(columns={'rating': 'away_elo'}).drop(columns=['team', 'change'])
    
    df = df.dropna(subset=['home_elo', 'away_elo', 'home_form'])

    # Step 4: Calculate the DIFFERENTIALS (The features the ML model will actually learn from)
    print("Calculating final differentials...")
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['form_diff'] = df['home_form'] - df['away_form']
    df['attack_diff'] = df['home_attack'] - df['away_attack']
    
    # Keep only the features we need for Phase 3
    final_features = ['date', 'home_team', 'away_team', 'elo_diff', 'form_diff', 'attack_diff', 'is_home_advantage', 'target']
    final_df = df[final_features]
    
    print(f"Phase 2 Complete. Engineered features for {len(final_df)} matches.")
    print("--- ---------------------------------- ---\n")
    
    return final_df

if __name__ == "__main__":
    ml_dataset = engineer_features()
    print("Preview of Final ML Features:")
    print(ml_dataset.tail())