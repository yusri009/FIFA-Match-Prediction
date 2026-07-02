import pandas as pd

def harmonize_data(results_file='./Dataset/results.csv', names_file='./Dataset/former_names.csv', start_year=1990):
    """
    Executes Phase 1 of the ML Architecture:
    Cleans and standardizes historical football data for machine learning.
    """
    print("--- Starting Phase 1: Data Harmonization ---")
    
    # 1. Load the raw datasets
    print("Loading raw CSVs...")
    results_df = pd.read_csv(results_file, parse_dates=['date'])
    names_df = pd.read_csv(names_file)
    
    # 2. Filter out ancient eras
    # Football before 1990 had vastly different rules (e.g., back-pass rule introduced in 1992)
    # Training a model on 1920s matches will confuse it.
    initial_count = len(results_df)
    results_df = results_df[results_df['date'].dt.year >= start_year].copy()
    print(f"Filtered matches from {start_year} onwards: {initial_count} -> {len(results_df)} matches.")
    
    # 3. Create the translation dictionary
    # Maps 'former' names to 'current' names (e.g., 'Swaziland' -> 'Eswatini')
    name_mapping = dict(zip(names_df['former'], names_df['current']))
    print(f"Loaded {len(name_mapping)} historical name mappings.")
    
    # 4. Standardize the team names
    # If a team name exists in our mapping dictionary, replace it. Otherwise, leave it alone.
    results_df['home_team'] = results_df['home_team'].replace(name_mapping)
    results_df['away_team'] = results_df['away_team'].replace(name_mapping)
    
    # Also standardize the 'country' column just to keep the whole dataset clean
    results_df['country'] = results_df['country'].replace(name_mapping)
    
    # 5. Final validation checks
    # Let's verify a known change, like 'Zaïre' becoming 'DR Congo' in 1997
    if 'Zaïre' in results_df['home_team'].values:
        print("WARNING: Harmonization failed. Outdated names still exist.")
    else:
        print("SUCCESS: Team names successfully harmonized to modern standards.")
        
    print("--- Phase 1 Complete ---\n")
    
    return results_df

if __name__ == "__main__":
    # Run the harmonization and preview the cleaned data
    clean_results = harmonize_data()
    
    print("Preview of Cleaned Results:")
    print(clean_results[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head())