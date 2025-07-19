import pandas as pd
from pathlib import Path
import glob

# Read all WAR CSV files in data folder
path = Path(__file__).resolve().parent.parent
bat_files = glob.glob(str(path / 'data' / 'all_player_stats' / 'batting_stats' / 'player_bat_*.csv'))
pitch_files = glob.glob(str(path / 'data' / 'all_player_stats' / 'pitching_stats' / 'player_pitch_*.csv'))
print(f"Found {len(bat_files)} batting files and {len(pitch_files)} pitching files.")
# Concatenate all into one DataFrame
bat_df = pd.concat((pd.read_csv(f) for f in bat_files), ignore_index=True)
pitch_df = pd.concat((pd.read_csv(f) for f in pitch_files), ignore_index=True)

# Assuming columns: 'team' and 'WAR'
bat_team_war_career = bat_df.groupby('Team', as_index=False)['WAR'].sum().round(2)
pitch_team_war_career = pitch_df.groupby('Team', as_index=False)['WAR'].sum().round(2)

# Season Groupings
bat_team_war_season = bat_df.groupby(['Team', 'Season'], as_index=False)['WAR'].sum().round(2)
pitch_team_war_season = pitch_df.groupby(['Team', 'Season'], as_index=False)['WAR'].sum().round(2)

# Merge batting and pitching WAR
team_war = pd.merge(bat_team_war_career, pitch_team_war_career, on='Team', suffixes=('_bat', '_pitch'))

# Calculate total WAR
team_war['WAR_total'] = (team_war['WAR_bat'] + team_war['WAR_pitch']).round(2)
# Rename columns for clarity
team_war = team_war.rename(columns={
    'Team': 'team',
    'WAR_bat': 'batting_WAR',
    'WAR_pitch': 'pitching_WAR',
    'WAR_total': 'total_WAR'
})

# Season Team WAR
season_team_war = pd.merge(bat_team_war_season, pitch_team_war_season, on=['Team', 'Season'], suffixes=('_bat', '_pitch'))

# Calculate total WAR for each season
season_team_war['WAR_total'] = (season_team_war['WAR_bat'] + season_team_war['WAR_pitch']).round(2)

# Rename columns for clarity
season_team_war = season_team_war.rename(columns={
    'Team': 'team',
    'Season': 'season',
    'WAR_bat': 'batting_WAR',
    'WAR_pitch': 'pitching_WAR',
    'WAR_total': 'total_WAR'
})

season_team_war = season_team_war.sort_values(by='season').reset_index(drop=True)
# Save
team_war.to_csv(path / 'data' / 'all_player_stats' / 'totals' / 'team_war_career.csv', index=False)
season_team_war.to_csv(path / 'data' / 'all_player_stats' / 'season' / 'team_war_season_by_season.csv', index=False)