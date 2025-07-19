import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Read all player totals CSV files in data folder
path = Path(__file__).resolve().parent.parent
bat_files = glob.glob(str(path / 'data' / 'all_player_stats' / 'batting_stats' / 'player_bat_*.csv'))
pitch_files = glob.glob(str(path / 'data' / 'all_player_stats' / 'pitching_stats' / 'player_pitch_*.csv'))

# Concatenate all batting stats into one DataFrame
bat_df = pd.concat((pd.read_csv(f) for f in bat_files), ignore_index=True)
# Concatenate all pitching stats into one DataFrame
pitch_df = pd.concat((pd.read_csv(f) for f in pitch_files), ignore_index=True)

# Batting
# Perform Career Totals Calculations
# After your existing aggregation...
bat_totals = bat_df.groupby(['IDfg', 'Name'], as_index=False).agg(
    Team = pd.NamedAgg(column='Team', aggfunc=lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
    rookie_season = pd.NamedAgg(column='Season', aggfunc='min'),
    final_season  = pd.NamedAgg(column='Season', aggfunc='max'),
    years_played  = pd.NamedAgg(column='Season', aggfunc='size'),
    G             = pd.NamedAgg(column='G', aggfunc='sum'),
    AB            = pd.NamedAgg(column='AB', aggfunc='sum'),
    PA            = pd.NamedAgg(column='PA', aggfunc='sum'),
    **{col: pd.NamedAgg(column=col, aggfunc='sum') for col in [
        '1B','2B','3B','H','HR','R','RBI','BB','IBB','SO','HBP',
        'SH','SF','GDP','SB','CS','GB','FB','LD','IFFB',
        'Pitches','Balls','Strikes','IFH','BU','BUH',
        'Barrels','HardHit','Events','WAR','L-WAR','RAR'
    ]}
)

# Add calculated stats
bat_totals['TB'] = (
    bat_totals['1B'] 
    + 2 * bat_totals['2B'] 
    + 3 * bat_totals['3B'] 
    + 4 * bat_totals['HR']
)

bat_totals['BIP'] = (
    bat_totals['AB'] 
    - bat_totals['SO'] 
    - bat_totals['HR'] 
    + bat_totals['SF']
)

# Batting average
bat_totals['AVG'] = round(bat_totals['H'] / bat_totals['AB'], 3).fillna(0)

# On-base percentage
bat_totals['OBP'] = round((
    bat_totals['H'] 
    + bat_totals['BB'] 
    + bat_totals['HBP']
) / (
    bat_totals['AB'] 
    + bat_totals['BB'] 
    + bat_totals['HBP'] 
    + bat_totals['SF']
), 3).fillna(0)

# Slugging percentage
bat_totals['SLG'] = round(bat_totals['TB'] / bat_totals['AB'], 3).fillna(0)

# OPS
bat_totals['OPS'] = round(bat_totals['OBP'] + bat_totals['SLG'], 3).fillna(0)

# Isolated Power (ISO)
bat_totals['ISO'] = round(bat_totals['SLG'] - bat_totals['AVG'], 3).fillna(0)

# BABIP
bat_totals['BABIP'] = round((
    bat_totals['H'] - bat_totals['HR']
) / (
    bat_totals['AB'] 
    - bat_totals['SO'] 
    - bat_totals['HR'] 
    + bat_totals['SF']
), 3).fillna(0)

# Rates using plate appearances
bat_totals['BB%'] = round(bat_totals['BB'] / bat_totals['PA'], 3).fillna(0)
bat_totals['K%']  = round(bat_totals['SO'] / bat_totals['PA'], 3).fillna(0)

# Home run to fly ball rate (est, only if FB exists)
if 'FB' in bat_totals.columns:
    bat_totals['HR/FB'] = round(bat_totals['HR'] / bat_totals['FB'], 3).fillna(0)

# Stolen base success rate
bat_totals['SB%'] = round(bat_totals['SB'] / (bat_totals['SB'] + bat_totals['CS']), 3).fillna(0)

# -------------------------------
# Rates for Ground Balls, Fly Balls, and Line Drives
# -------------------------------
for col in ['GB', 'FB', 'LD']:
    if col not in bat_totals.columns:
        print(f"⚠️ Column '{col}' not in data; cannot compute related rates.")

# Compute percentages
bat_totals['GB%'] = round(bat_totals['GB'] / bat_totals['BIP'], 3).fillna(0)
bat_totals['FB%'] = round(bat_totals['FB'] / bat_totals['BIP'], 3).fillna(0)
bat_totals['LD%'] = round(bat_totals['LD'] / bat_totals['BIP'], 3).fillna(0)

# Ground ball to fly ball ratio
bat_totals['GB/FB'] = round(bat_totals['GB'] / bat_totals['FB'], 3).fillna(0)

# Infield fly-ball % (IFFB%), if 'IFFB' is in dataset:
if 'IFFB' in bat_totals.columns:
    bat_totals['IFFB%'] = round(bat_totals['IFFB'] / bat_totals['FB'], 3).fillna(0)

print(bat_totals.columns)

bat_totals = bat_totals.sort_values(by='Name')
bat_totals.to_csv(path / 'data' / 'all_player_stats' / 'totals' / 'player_bat_career_totals.csv', index=False)

# Pitching
# Perform Career Totals Calculations
pitch_totals = pitch_df.groupby(['IDfg', 'Name'], as_index=False).agg(
    Team = pd.NamedAgg(column='Team', aggfunc=lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
    rookie_season = pd.NamedAgg(column='Season', aggfunc='min'),
    final_season  = pd.NamedAgg(column='Season', aggfunc='max'),
    years_played  = pd.NamedAgg(column='Season', aggfunc='size'),
    G             = pd.NamedAgg(column='G', aggfunc='sum'),
    GS            = pd.NamedAgg(column='GS', aggfunc='sum'),
    IP            = pd.NamedAgg(column='IP', aggfunc='sum'),
    **{col: pd.NamedAgg(column=col, aggfunc='sum') for col in [
        'W', 'L', 'SV', 'ShO', 'CG', 'BS', 'TBF', 'H','R','ER','HR','BB','IBB','HBP','WP','BK',
        'SO', 'GB', 'FB', 'LD', 'IFFB', 'Pitches','Balls','Strikes', 'RS',
        'Barrels','HardHit','Events','WAR', 'RAR', 'HardHit'
    ]}
)

# Add calculated stats
pitch_totals['W/L'] = round(pitch_totals['W'] / pitch_totals['L'], 2).fillna(0)
pitch_totals['ERA'] = round(9 * pitch_totals['ER'] / pitch_totals['IP'], 2).fillna(0)
pitch_totals['AVG'] = round(pitch_totals['H'] / pitch_totals['TBF'], 3).fillna(0)
pitch_totals['OBP'] = round((pitch_totals['H'] + pitch_totals['BB'] + pitch_totals['HBP']) / pitch_totals['TBF'], 3).fillna(0)
pitch_totals['WHIP'] = round((pitch_totals['H'] + pitch_totals['BB']) / pitch_totals['IP'], 2).fillna(0)
pitch_totals['FIP'] = round(
    (3 * pitch_totals['HR'] + 13 * pitch_totals['BB'] + 3 * pitch_totals['HBP'] - 2 * pitch_totals['SO']) / pitch_totals['IP'] + 3.1,
    2               
).fillna(0)
pitch_totals['K/9'] = round(9 * pitch_totals['SO'] / pitch_totals['IP'], 2).fillna(0)
pitch_totals['BB/9'] = round(9 * pitch_totals['BB'] / pitch_totals['IP'], 2).fillna(0)
pitch_totals['K/BB'] = round(pitch_totals['SO'] / pitch_totals['BB'], 2).fillna(0)
pitch_totals['LOB%'] = (
    (pitch_totals['H'] + pitch_totals['BB'] + pitch_totals['HBP'] - pitch_totals['R']) /
    (pitch_totals['H'] + pitch_totals['BB'] + pitch_totals['HBP'] - 1.4 * pitch_totals['HR'])
).round(3).fillna(0)
pitch_totals['GB%'] = round(pitch_totals['GB'] / (pitch_totals['GB'] + pitch_totals['FB'] + pitch_totals['LD']), 3).fillna(0)
pitch_totals['FB%'] = round(pitch_totals['FB'] / (pitch_totals['GB'] + pitch_totals['FB'] + pitch_totals['LD']), 3).fillna(0)
pitch_totals['LD%'] = round(pitch_totals['LD'] / (pitch_totals['GB'] + pitch_totals['FB'] + pitch_totals['LD']), 3).fillna(0)
pitch_totals['GB/FB'] = round(pitch_totals['GB'] / pitch_totals['FB'], 3).fillna(0)
pitch_totals['LD/FB'] = round(pitch_totals['LD'] / pitch_totals['FB'], 3).fillna(0)
pitch_totals['GB/LD'] = round(pitch_totals['GB'] / pitch_totals['LD'], 3).fillna(0)
pitch_totals['IFFB%'] = round(pitch_totals['IFFB'] / pitch_totals['FB'], 3).fillna(0)
pitch_totals['Barrel%'] = round(pitch_totals['Barrels'] / pitch_totals['Events'], 3).fillna(0)
pitch_totals['HardHit%'] = round(pitch_totals['HardHit'] / pitch_totals['Events'], 3).fillna(0)
pitch_totals['Barrel/PA'] = round(pitch_totals['Barrels'] / pitch_totals['Pitches'], 3).fillna(0)
pitch_totals['HardHit/PA'] = round(pitch_totals['HardHit'] / pitch_totals['Pitches'], 3).fillna(0)

pitch_totals = pitch_totals.sort_values(by='Name')
pitch_totals.to_csv(path / 'data' / 'all_player_stats' / 'totals' / 'player_pitch_career_totals.csv', index=False)