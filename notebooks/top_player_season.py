import pandas as pd
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

def analyze_season_leaders(season_years=None, save_visuals=True):
    """
    Analyze top players for each season and create visualizations.
    
    Parameters:
    season_years (list): List of years to analyze. If None, finds all available seasons.
    save_visuals (bool): Whether to save visualization images
    
    Returns:
    dict: Dictionary containing all season results
    """
    
    # -------------------------------
    # Setup Paths and Load Team Colors
    # -------------------------------
    path = Path(__file__).resolve().parent.parent
    team_colors = pd.read_csv(path / 'data' / 'team_colors.csv')
    
    # Clean hex codes
    team_colors['primary_color'] = team_colors['primary_color'].str.strip()
    team_colors['secondary_color'] = team_colors['secondary_color'].str.strip()
    
    # -------------------------------
    # Find Available Season Files
    # -------------------------------
    if season_years is None:
        # Find all available season files
        bat_files = glob.glob(str(path / 'data' / 'all_player_stats' / 'batting_stats' / 'player_bat_*.csv'))
        pitch_files = glob.glob(str(path / 'data' / 'all_player_stats' / 'pitching_stats' / 'player_pitch_*.csv'))
        
        # Extract years from filenames
        bat_years = [int(Path(f).stem.split('_')[2]) for f in bat_files]
        pitch_years = [int(Path(f).stem.split('_')[2]) for f in pitch_files]
        
        # Use intersection of available years
        season_years = sorted(list(set(bat_years) & set(pitch_years)))
        
        if not season_years:
            print("No matching season files found!")
            return {}
    
    print(f"Analyzing seasons: {season_years}")
    
    # -------------------------------
    # Initialize Results Storage
    # -------------------------------
    all_results = {}
    
    # -------------------------------
    # Process Each Season
    # -------------------------------
    for year in season_years:
        print(f"\nProcessing {year} season...")
        
        # Load season files
        bat_file = path / 'data' / 'all_player_stats' / 'batting_stats' / f'player_bat_{year}.csv'
        pitch_file = path / 'data' / 'all_player_stats' / 'pitching_stats' / f'player_pitch_{year}.csv'
        
        if not bat_file.exists() or not pitch_file.exists():
            print(f"Missing files for {year}, skipping...")
            continue
            
        bat_df = pd.read_csv(bat_file)
        pitch_df = pd.read_csv(pitch_file)
        
        # -------------------------------
        # Calculate Derived Statistics
        # -------------------------------
        # Batting statistics - season-specific calculations
        if 'PA' in bat_df.columns and 'G' in bat_df.columns:
            bat_df['PA/G'] = bat_df['PA'] / bat_df['G'].max()
        elif 'PA' in bat_df.columns:
            max_games_in_season = bat_df['G'].max()
            bat_df['PA/G'] = bat_df['PA'] / max_games_in_season  
        else:
            bat_df['PA/G'] = 0
        
        # Add OPS if not present
        if 'OPS' not in bat_df.columns and 'OBP' in bat_df.columns and 'SLG' in bat_df.columns:
            bat_df['OPS'] = bat_df['OBP'] + bat_df['SLG']
        
        # Pitching statistics
        if 'IP' in pitch_df.columns and 'G' in pitch_df.columns:
            pitch_df['IP/G'] = pitch_df['IP'] / pitch_df['G'].max()
        elif 'IP' in pitch_df.columns:
            max_games_in_season = pitch_df['G'].max()
            pitch_df['IP/G'] = pitch_df['IP'] / max_games_in_season  # Assume full season
        else:
            pitch_df['IP/G'] = 0
            
        if 'WHIP' not in pitch_df.columns and 'BB' in pitch_df.columns and 'H' in pitch_df.columns and 'IP' in pitch_df.columns:
            pitch_df['WHIP'] = (pitch_df['BB'] + pitch_df['H']) / pitch_df['IP']
            
        if 'SO/W' not in pitch_df.columns and 'SO' in pitch_df.columns and 'BB' in pitch_df.columns:
            pitch_df['SO/W'] = pitch_df['SO'] / pitch_df['BB'].replace(0, np.nan)
        
        # -------------------------------
        # Define Qualified Players (Season Standards)
        # -------------------------------
        # Season qualification standards (3.1 PA/G = ~502 PA, 1 IP/G = ~162 IP)
        qualified_hitters = bat_df[bat_df['PA/G'] >= 3.1] if 'PA/G' in bat_df.columns else bat_df
        qualified_pitchers = pitch_df[pitch_df['IP/G'] >= 1.0] if 'IP/G' in pitch_df.columns else pitch_df
        
        # -------------------------------
        # Get Top Players by Category
        # -------------------------------
        season_leaders = {}
        
        # Batting Leaders (only if we have qualified players)
        if len(qualified_hitters) > 0:
            available_bat_stats = []
            if 'WAR' in qualified_hitters.columns:
                season_leaders['bat_war'] = qualified_hitters.nlargest(10, 'WAR')[['Name', 'Team', 'WAR']]
                available_bat_stats.append('WAR')
            if 'AVG' in qualified_hitters.columns:
                season_leaders['bat_avg'] = qualified_hitters.nlargest(10, 'AVG')[['Name', 'Team', 'AVG']]
                available_bat_stats.append('AVG')
            if 'HR' in qualified_hitters.columns:
                season_leaders['bat_hr'] = qualified_hitters.nlargest(10, 'HR')[['Name', 'Team', 'HR']]
                available_bat_stats.append('HR')
            if 'RBI' in qualified_hitters.columns:
                season_leaders['bat_rbi'] = qualified_hitters.nlargest(10, 'RBI')[['Name', 'Team', 'RBI']]
                available_bat_stats.append('RBI')
            if 'SB' in qualified_hitters.columns:
                season_leaders['bat_sb'] = qualified_hitters.nlargest(10, 'SB')[['Name', 'Team', 'SB']]
                available_bat_stats.append('SB')
            if 'OPS' in qualified_hitters.columns:
                season_leaders['bat_ops'] = qualified_hitters.nlargest(10, 'OPS')[['Name', 'Team', 'OPS']]
                available_bat_stats.append('OPS')
        
        # Pitching Leaders (only if we have qualified players)
        if len(qualified_pitchers) > 0:
            available_pitch_stats = []
            if 'WAR' in qualified_pitchers.columns:
                season_leaders['pitch_war'] = qualified_pitchers.nlargest(10, 'WAR')[['Name', 'Team', 'WAR']]
                available_pitch_stats.append('WAR')
            if 'ERA' in qualified_pitchers.columns:
                season_leaders['pitch_era'] = qualified_pitchers.nsmallest(10, 'ERA')[['Name', 'Team', 'ERA']]
                available_pitch_stats.append('ERA')
            if 'WHIP' in qualified_pitchers.columns:
                season_leaders['pitch_whip'] = qualified_pitchers.nsmallest(10, 'WHIP')[['Name', 'Team', 'WHIP']]
                available_pitch_stats.append('WHIP')
            if 'SO' in qualified_pitchers.columns:
                season_leaders['pitch_so'] = qualified_pitchers.nlargest(10, 'SO')[['Name', 'Team', 'SO']]
                available_pitch_stats.append('SO')
            if 'SO/W' in qualified_pitchers.columns:
                season_leaders['pitch_so_w'] = qualified_pitchers.nlargest(10, 'SO/W')[['Name', 'Team', 'SO/W']]
                available_pitch_stats.append('SO/W')
        
        # Wins and Saves (all pitchers, not just qualified)
        if 'W' in pitch_df.columns:
            season_leaders['pitch_w'] = pitch_df.nlargest(10, 'W')[['Name', 'Team', 'W']]
            available_pitch_stats.append('W')
        if 'SV' in pitch_df.columns:
            season_leaders['pitch_sv'] = pitch_df.nlargest(10, 'SV')[['Name', 'Team', 'SV']]
            available_pitch_stats.append('SV')
        
        # -------------------------------
        # Add Team Colors to All DataFrames
        # -------------------------------
        def add_colors(df):
            # Merge existing team colors
            df = df.merge(
                team_colors[['abbrev', 'primary_color', 'secondary_color']],
                left_on='Team', right_on='abbrev', how='left'
            )
            
            # Define default MLB Player colors (MLB blue & red)
            default_primary = '#002D72'     # MLB Blue (Pantone 288 C) :contentReference[oaicite:1]{index=1}
            default_secondary = '#D50032'   # MLB Red (Pantone 199 C) :contentReference[oaicite:2]{index=2}
            
            # Apply defaults where colors are missing
            df['primary_color'] = df['primary_color'].fillna(default_primary)
            df['secondary_color'] = df['secondary_color'].fillna(default_secondary)
            
            return df
        
        # Apply colors to all season leader dataframes
        for key in season_leaders.keys():
            season_leaders[key] = add_colors(season_leaders[key])
        
        # Store results for this season
        all_results[year] = season_leaders
        
        # -------------------------------
        # Create Visuals for This Season
        # -------------------------------
        if save_visuals:
            print(f"Creating visualizations for {year}...")
            
            for stat_key, df in season_leaders.items():
                if len(df) > 0:
                    # Parse the stat type and column
                    stat_type, stat_col = stat_key.split('_', 1)
                    col_name = stat_col.upper().replace('_', '/')
                    
                    title = f"{year} {col_name} Leaders"
                    if stat_col in ['era', 'whip']:
                        title += " (Lowest)"
                    
                    create_season_visual(df, 'Name', col_name, title, year, save=save_visuals)
        
        # Print season summary
        print_season_summary(year, season_leaders)
    
    return all_results

def create_season_visual(df, x_col, y_col, title, year, figsize=(10,8), save=True):
    """
    Create a visual for season leaders (similar to the career visuals function)
    """
    # Create figure with proper spacing
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(style="whitegrid")
    
    df = df.copy().fillna({
        'primary_color':'#000089',
        'secondary_color':'#CD0001',
        'abbrev':'Traded Player'  # Free Agent/Traded Player
    })

    df[y_col] = df[y_col].fillna(0)  # Fill NaN values with 0 for plotting
    
    # Sort based on the statistic (ERA and WHIP should be ascending for "best")
    if y_col in ['ERA', 'WHIP']:
        df.sort_values(by=y_col, ascending=False, inplace=True)  # Reverse for display (best at top)
    else:
        df.sort_values(by=y_col, ascending=True, inplace=True)  # Normal sort (highest at top)
    
    y_labels = df[x_col].tolist()
    values = df[y_col].tolist()
    prim_cols = df['primary_color'].tolist()
    sec_cols = df['secondary_color'].tolist()
    teams = df['abbrev'].tolist()

    bar_height = 0.6
    overlay_height = bar_height * 0.5
    max_val = max(values)
    min_val = min(values)

    for i, (val, pcol, scol, t) in enumerate(zip(values, prim_cols, sec_cols, teams)):
        # Create the main bar
        ax.add_patch(Rectangle((0, i - bar_height/2), width=val, height=bar_height, color=scol, zorder=1))
        ax.add_patch(Rectangle((0, i - overlay_height/2), width=val, height=overlay_height, color=pcol, zorder=2))

        # Add team abbreviation
        ax.text(val * 0.02, i, t, va='center', ha='left', color='white', fontweight='bold', fontsize=8)
        
        # Add value with appropriate formatting
        if y_col in ['AVG', 'ERA', 'WHIP', 'OBP', 'SLG', 'OPS']:
            value_text = f"{val:.3f}"
        elif y_col in ['SO/W']:
            value_text = f"{val:.1f}"
        else:
            value_text = f"{int(val)}"
            
        ax.text(val + max_val*0.005, i, value_text, va='center', ha='left', color='black', fontsize=8)

    # Set proper limits and spacing
    ax.set_xlim(0, max_val * 1.1)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel(y_col, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)

    if save:
        safe_title = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        path = Path(__file__).resolve().parent.parent
        out = path / 'visuals' / 'season' / str(year) / f"{safe_title}.png"
        out.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()

def print_season_summary(year, season_leaders):
    """
    Print a summary of the season's top performers
    """
    print(f"\n{year} SEASON LEADERS:")
    print("="*30)
    
    # Batting leaders
    if 'bat_war' in season_leaders and len(season_leaders['bat_war']) > 0:
        leader = season_leaders['bat_war'].iloc[0]
        print(f"Batting WAR: {leader['Name']} ({leader['Team']}) - {leader['WAR']:.1f}")
    
    if 'bat_avg' in season_leaders and len(season_leaders['bat_avg']) > 0:
        leader = season_leaders['bat_avg'].iloc[0]
        print(f"Batting AVG: {leader['Name']} ({leader['Team']}) - {leader['AVG']:.3f}")
    
    if 'bat_hr' in season_leaders and len(season_leaders['bat_hr']) > 0:
        leader = season_leaders['bat_hr'].iloc[0]
        print(f"Home Runs: {leader['Name']} ({leader['Team']}) - {int(leader['HR'])}")
    
    # Pitching leaders
    if 'pitch_war' in season_leaders and len(season_leaders['pitch_war']) > 0:
        leader = season_leaders['pitch_war'].iloc[0]
        print(f"Pitching WAR: {leader['Name']} ({leader['Team']}) - {leader['WAR']:.1f}")
    
    if 'pitch_era' in season_leaders and len(season_leaders['pitch_era']) > 0:
        leader = season_leaders['pitch_era'].iloc[0]
        print(f"ERA: {leader['Name']} ({leader['Team']}) - {leader['ERA']:.3f}")
    
    if 'pitch_so' in season_leaders and len(season_leaders['pitch_so']) > 0:
        leader = season_leaders['pitch_so'].iloc[0]
        print(f"Strikeouts: {leader['Name']} ({leader['Team']}) - {int(leader['SO'])}")

def get_multi_season_comparison(results, stat_category, top_n=5):
    """
    Compare a specific stat across multiple seasons
    
    Parameters:
    results (dict): Results from analyze_season_leaders
    stat_category (str): e.g., 'bat_war', 'pitch_era', etc.
    top_n (int): Number of top players to show per season
    
    Returns:
    pandas.DataFrame: Comparison across seasons
    """
    comparison_data = []
    
    for year, season_data in results.items():
        if stat_category in season_data:
            season_leaders = season_data[stat_category].head(top_n)
            for _, player in season_leaders.iterrows():
                comparison_data.append({
                    'Year': year,
                    'Name': player['Name'],
                    'Team': player['Team'],
                    'Value': player[stat_category.split('_')[1].upper().replace('_', '/')]
                })
    
    return pd.DataFrame(comparison_data)

seasons_to_analyze = None  # Analyze all available seasons
save_visuals = True  # Save visualizations
results = analyze_season_leaders(seasons_to_analyze, save_visuals)

war_comparison = get_multi_season_comparison(results, 'bat_war', top_n=3)
print("\nTop 3 Batting WAR by Season:")
print(war_comparison.to_string(index=False))    

