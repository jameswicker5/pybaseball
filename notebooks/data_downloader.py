import pybaseball
import pandas as pd
from pathlib import Path
from time import sleep

# Base directory for output files (modify if needed)
base_dir = Path(__file__).resolve().parent.parent / 'data'
bats_dir = base_dir / 'batting_stats'
pitches_dir = base_dir / 'pitching_stats'
bats_dir.mkdir(parents=True, exist_ok=True)
pitches_dir.mkdir(parents=True, exist_ok=True)

# Loop through all years
for year in range(1871, 2025):
    print(f"Fetching data for year: {year}")

    # Batting stats with qualification threshold (0 = include all players)
    try:
        df_bat = pybaseball.batting_stats(year, qual=0)
        df_bat.to_csv(bats_dir / f'player_bat_{year}.csv', index=False)
        print(f"✅ Batting stats saved for {year}")
    except Exception as e:
        print(f"❌ Failed to fetch batting stats for {year}: {e}")

    # Pitching stats with qualification threshold (0 = include all players)
    try:
        df_pitch = pybaseball.pitching_stats(year, qual=0)
        df_pitch.to_csv(pitches_dir / f'player_pitch_{year}.csv', index=False)
        print(f"✅ Pitching stats saved for {year}")
    except Exception as e:
        print(f"❌ Failed to fetch pitching stats for {year}: {e}")

    # Short pause to avoid server overload
    sleep(1)
