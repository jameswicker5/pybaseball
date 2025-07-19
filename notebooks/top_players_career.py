import pandas as pd
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# -------------------------------
# Load Files
# -------------------------------
path = Path(__file__).resolve().parent.parent
bat_file = glob.glob(str(path / 'data' / 'all_player_stats' / 'totals' / 'player_bat_career_totals.csv'))
pitch_file = glob.glob(str(path / 'data' / 'all_player_stats' / 'totals' / 'player_pitch_career_totals.csv'))

bat_df = pd.read_csv(bat_file[0])
pitch_df = pd.read_csv(pitch_file[0])
team_colors = pd.read_csv(path / 'data' / 'team_colors.csv')

# Clean hex codes
team_colors['primary_color'] = team_colors['primary_color'].str.strip()
team_colors['secondary_color'] = team_colors['secondary_color'].str.strip()

# -------------------------------
# Calculate Derived Statistics
# -------------------------------
# Batting statistics
bat_df['PA/G'] = bat_df['PA'] / (bat_df['years_played'] * 162)


# Pitching statistics
pitch_df['IP/G'] = pitch_df['IP'] / (pitch_df['years_played'] * 162)
pitch_df['WHIP'] = (pitch_df['BB'] + pitch_df['H']) / pitch_df['IP']
pitch_df['SO/W'] = pitch_df['SO'] / pitch_df['BB']


# -------------------------------
# Define Qualified Players
# -------------------------------
qualified_hitters = bat_df[bat_df['PA/G'] >= 3.1]  # Minimum ~502 PA per season
qualified_pitchers = pitch_df[pitch_df['IP/G'] >= (1/9)]  # Minimum ~162 IP per season

# -------------------------------
# Top Players by Category
# -------------------------------
# Batting Leaders
top_bat_war = qualified_hitters.nlargest(10, 'WAR')[['Name', 'Team', 'WAR']]
top_bat_avg = qualified_hitters.nlargest(10, 'AVG')[['Name', 'Team', 'AVG']]
top_bat_hr = qualified_hitters.nlargest(10, 'HR')[['Name', 'Team', 'HR']]
top_bat_rbi = qualified_hitters.nlargest(10, 'RBI')[['Name', 'Team', 'RBI']]
top_bat_sb = qualified_hitters.nlargest(10, 'SB')[['Name', 'Team', 'SB']]

# Pitching Leaders
top_pitch_war = qualified_pitchers.nlargest(10, 'WAR')[['Name', 'Team', 'WAR']]
top_pitch_era = qualified_pitchers.nsmallest(10, 'ERA')[['Name', 'Team', 'ERA']]
top_pitch_whip = qualified_pitchers.nsmallest(10, 'WHIP')[['Name', 'Team', 'WHIP']]
top_pitch_so = qualified_pitchers.nlargest(10, 'SO')[['Name', 'Team', 'SO']]
top_pitch_so_w = qualified_pitchers.nlargest(10, 'SO/W')[['Name', 'Team', 'SO/W']]

# Wins and Saves (all pitchers, not just qualified)
top_pitch_w = pitch_df.nlargest(10, 'W')[['Name', 'Team', 'W']]
top_pitch_sv = pitch_df.nlargest(10, 'SV')[['Name', 'Team', 'SV']]

# -------------------------------
# Merge Team Colors
# -------------------------------
def add_colors(df):
    return df.merge(
        team_colors[['abbrev', 'primary_color', 'secondary_color']],
        left_on='Team', right_on='abbrev', how='left'
    )

# Apply colors to all dataframes
batting_dfs = [top_bat_war, top_bat_avg, top_bat_hr, top_bat_rbi, top_bat_sb]
pitching_dfs = [top_pitch_war, top_pitch_era, top_pitch_whip, 
               top_pitch_so, top_pitch_so_w, top_pitch_w, top_pitch_sv]

# Add colors to all dataframes
for i, df in enumerate(batting_dfs):
    batting_dfs[i] = add_colors(df)

for i, df in enumerate(pitching_dfs):
    pitching_dfs[i] = add_colors(df)

# Update the individual dataframe references
top_bat_war, top_bat_avg, top_bat_hr, top_bat_rbi, top_bat_sb = batting_dfs
top_pitch_war, top_pitch_era, top_pitch_whip, top_pitch_so, top_pitch_so_w, top_pitch_w, top_pitch_sv = pitching_dfs

# -------------------------------
# Enhanced Visuals Function
# -------------------------------
def visuals(df, x_col, y_col, title=None, figsize=(10,8), save=True, ascending_sort=True):
    # Create figure with proper spacing
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(style="whitegrid")
    
    df = df.copy().fillna({
        'primary_color':'#000089',
        'secondary_color':'#CD0001',
        'abbrev':'FA'  # Free Agent/Traded Player
    })
    
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
        if y_col in ['AVG', 'ERA', 'WHIP', 'OBP', 'SLG']:
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
    ax.set_title(title or f"Top 10 by {y_col}", fontsize=14, pad=20)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)

    if save:
        safe_title = (title or f"top_10_{y_col}").lower().replace(' ', '_').replace('/', '_')
        out = path / 'visuals' / 'career' / f"{safe_title}.png"
        out.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out, dpi=200, bbox_inches='tight')
    
    plt.show()

# -------------------------------
# Create All Visuals
# -------------------------------
print("Creating Batting Leader Visualizations...")

# Batting Leaders
visuals(top_bat_war, 'Name', 'WAR', title='Career Batting WAR Leaders')
visuals(top_bat_avg, 'Name', 'AVG', title='Career Batting Average Leaders')
visuals(top_bat_hr, 'Name', 'HR', title='Career Home Run Leaders')
visuals(top_bat_rbi, 'Name', 'RBI', title='Career RBI Leaders')
visuals(top_bat_sb, 'Name', 'SB', title='Career Stolen Base Leaders')

print("Creating Pitching Leader Visualizations...")

# Pitching Leaders
visuals(top_pitch_war, 'Name', 'WAR', title='Career Pitching WAR Leaders')
visuals(top_pitch_era, 'Name', 'ERA', title='Career ERA Leaders (Lowest)')
visuals(top_pitch_whip, 'Name', 'WHIP', title='Career WHIP Leaders (Lowest)')
visuals(top_pitch_so, 'Name', 'SO', title='Career Strikeout Leaders')
visuals(top_pitch_so_w, 'Name', 'SO/W', title='Career SO/W Ratio Leaders')
visuals(top_pitch_w, 'Name', 'W', title='Career Win Leaders')
visuals(top_pitch_sv, 'Name', 'SV', title='Career Save Leaders')

print("All visualizations complete! Check the 'visuals' folder for saved images.")

# -------------------------------
# Summary Statistics
# -------------------------------
print("\n" + "="*50)
print("SUMMARY OF TOP PERFORMERS")
print("="*50)

print(f"\nBATTING LEADERS:")
print(f"WAR Leader: {top_bat_war.iloc[0]['Name']} ({top_bat_war.iloc[0]['Team']}) - {top_bat_war.iloc[0]['WAR']:.1f}")
print(f"AVG Leader: {top_bat_avg.iloc[0]['Name']} ({top_bat_avg.iloc[0]['Team']}) - {top_bat_avg.iloc[0]['AVG']:.3f}")
print(f"HR Leader: {top_bat_hr.iloc[0]['Name']} ({top_bat_hr.iloc[0]['Team']}) - {int(top_bat_hr.iloc[0]['HR'])}")
print(f"RBI Leader: {top_bat_rbi.iloc[0]['Name']} ({top_bat_rbi.iloc[0]['Team']}) - {int(top_bat_rbi.iloc[0]['RBI'])}")

print(f"\nPITCHING LEADERS:")
print(f"WAR Leader: {top_pitch_war.iloc[0]['Name']} ({top_pitch_war.iloc[0]['Team']}) - {top_pitch_war.iloc[0]['WAR']:.1f}")
print(f"ERA Leader: {top_pitch_era.iloc[0]['Name']} ({top_pitch_era.iloc[0]['Team']}) - {top_pitch_era.iloc[0]['ERA']:.3f}")
print(f"SO Leader: {top_pitch_so.iloc[0]['Name']} ({top_pitch_so.iloc[0]['Team']}) - {int(top_pitch_so.iloc[0]['SO'])}")
print(f"Win Leader: {top_pitch_w.iloc[0]['Name']} ({top_pitch_w.iloc[0]['Team']}) - {int(top_pitch_w.iloc[0]['W'])}")
