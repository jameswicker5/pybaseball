import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from top_player_season import get_multi_season_comparison

era = get_multi_season_comparison(years=[2021, 2022, 2023, 2024], stat='ERA')
print(era)
    
