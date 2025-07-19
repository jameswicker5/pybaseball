import pandas as pd
from pathlib import Path
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

# Set style for better visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -------------------------------
# Load Files
# -------------------------------
path = Path(__file__).resolve().parent.parent
bat_file = glob.glob(str(path / 'data' / 'mlb_bat_*.csv'))
pitch_file = glob.glob(str(path / 'data' / 'mlb_pitch_*.csv'))
bat_df = pd.read_csv(bat_file[0])
pitch_df = pd.read_csv(pitch_file[0])
team_colors = pd.read_csv(path / 'data' / 'team_colors.csv')

# -------------------------------
# Enhanced PCA Analysis Function
# -------------------------------
def enhanced_pca_analysis(df, features, title, save=True, filter_outliers=True):
    """
    Enhanced PCA analysis with better visualization options
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Filter outliers if requested (remove extreme values)
    if filter_outliers:
        df_filtered = df.copy()
        for feature in features:
            Q1 = df_filtered[feature].quantile(0.25)
            Q3 = df_filtered[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_filtered = df_filtered[(df_filtered[feature] >= lower_bound) & 
                                    (df_filtered[feature] <= upper_bound)]
    else:
        df_filtered = df
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_filtered[features])
    
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    # Add team info and colors
    if 'Team' in df_filtered.columns:
        pca_df['Team'] = df_filtered['Team'].values
        pca_df = pca_df.merge(
            team_colors[['abbrev', 'primary_color']],
            left_on='Team', right_on='abbrev', how='left'
        )
        pca_df['color'] = pca_df['primary_color'].fillna('#2E86AB')
    else:
        pca_df['color'] = '#2E86AB'
    
    # -------------------------------
    # Subplot 1: Main PCA Scatter Plot
    # -------------------------------
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # Create scatter plot with improved aesthetics
    scatter = ax1.scatter(
        pca_df['PC1'], pca_df['PC2'],
        c=pca_df['color'],
        alpha=0.7,
        s=60,
        edgecolors='white',
        linewidth=1.5
    )
    
    # Add team labels with better positioning
    for _, row in pca_df.iterrows():
        ax1.annotate(row['Team'], 
                    (row['PC1'], row['PC2']), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8, 
                    fontweight='bold',
                    ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             alpha=0.7,
                             edgecolor='none'))
    
    ax1.set_title(f'{title}\n(Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%})', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add confidence ellipses for clusters
    try:
        # Calculate 95% confidence ellipse
        cov = np.cov(pca_df['PC1'], pca_df['PC2'])
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        
        ellipse = Ellipse(
            xy=(np.mean(pca_df['PC1']), np.mean(pca_df['PC2'])),
            width=lambda_[0]*2*2.57,  # 95% confidence
            height=lambda_[1]*2*2.57,
            angle=np.rad2deg(np.arccos(v[0, 0])),
            facecolor='none',
            edgecolor='red',
            linewidth=2,
            linestyle='--',
            alpha=0.5
        )
        ax1.add_patch(ellipse)
    except:
        pass
    
    # -------------------------------
    # Subplot 2: Feature Contributions (Biplot)
    # -------------------------------
    ax2 = plt.subplot(2, 3, 3)
    
    # Feature loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Plot feature vectors
    for i, feature in enumerate(features):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 head_width=0.05, head_length=0.05, 
                 fc='red', ec='red', alpha=0.8)
        ax2.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, 
                feature, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel('PC1 Loading', fontsize=12)
    ax2.set_ylabel('PC2 Loading', fontsize=12)
    ax2.set_title('Feature Contributions\n(Biplot)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # -------------------------------
    # Subplot 3: Explained Variance
    # -------------------------------
    ax3 = plt.subplot(2, 3, 4)
    
    # Full PCA to show all components
    pca_full = PCA()
    pca_full.fit(scaled_features)
    
    components = range(1, len(pca_full.explained_variance_ratio_) + 1)
    ax3.bar(components, pca_full.explained_variance_ratio_, 
           alpha=0.7, color='steelblue')
    ax3.set_xlabel('Principal Component', fontsize=12)
    ax3.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax3.set_title('Scree Plot\n(All Components)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Highlight first two components
    ax3.bar([1, 2], pca_full.explained_variance_ratio_[:2], 
           alpha=0.9, color='red', label='Selected PCs')
    ax3.legend()
    
    # -------------------------------
    # Subplot 4: Density Plot
    # -------------------------------
    ax4 = plt.subplot(2, 3, 5)
    
    # Create density plot
    x = pca_df['PC1']
    y = pca_df['PC2']
    
    # Calculate point density
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    
    scatter = ax4.scatter(x, y, c=density, s=50, alpha=0.7, cmap='viridis')
    ax4.set_xlabel('Principal Component 1', fontsize=12)
    ax4.set_ylabel('Principal Component 2', fontsize=12)
    ax4.set_title('Point Density\n(Darker = More Dense)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax4, label='Density')
    
    # -------------------------------
    # Subplot 5: Top/Bottom Performers
    # -------------------------------
    ax5 = plt.subplot(2, 3, 6)
    
    # Identify extreme performers (top/bottom 10% in PC1)
    pc1_top = pca_df.nlargest(int(len(pca_df) * 0.1), 'PC1')
    pc1_bottom = pca_df.nsmallest(int(len(pca_df) * 0.1), 'PC1')
    
    # Plot all points in light gray
    ax5.scatter(pca_df['PC1'], pca_df['PC2'], 
               c='lightgray', alpha=0.3, s=40)
    
    # Highlight top performers
    ax5.scatter(pc1_top['PC1'], pc1_top['PC2'], 
               c='green', alpha=0.8, s=80, label='Top 10% (PC1)')
    
    # Highlight bottom performers
    ax5.scatter(pc1_bottom['PC1'], pc1_bottom['PC2'], 
               c='red', alpha=0.8, s=80, label='Bottom 10% (PC1)')
    
    # Add labels for extreme performers
    for _, row in pc1_top.iterrows():
        ax5.annotate(row['Team'], (row['PC1'], row['PC2']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color='green')
    
    for _, row in pc1_bottom.iterrows():
        ax5.annotate(row['Team'], (row['PC1'], row['PC2']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color='red')
    
    ax5.set_xlabel('Principal Component 1', fontsize=12)
    ax5.set_ylabel('Principal Component 2', fontsize=12)
    ax5.set_title('Extreme Performers\n(Top/Bottom 10%)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # -------------------------------
    # Overall styling and save
    # -------------------------------
    plt.tight_layout()
    plt.suptitle(f'{title} - Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save:
        path.joinpath('visuals').mkdir(parents=True, exist_ok=True)
        plt.savefig(path / 'visuals' / f'{title.lower().replace(" ", "_")}_enhanced.png', 
                   dpi=300, bbox_inches='tight')
        
        # Save enhanced data
        enhanced_data = pca_df.copy()
        enhanced_data['PC1_percentile'] = enhanced_data['PC1'].rank(pct=True)
        enhanced_data['PC2_percentile'] = enhanced_data['PC2'].rank(pct=True)
        enhanced_data.to_csv(path / 'data' / f'{title.lower().replace(" ", "_")}_enhanced_pca.csv', 
                           index=False)
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{title} - PCA Summary:")
    print(f"Total variance explained by PC1 + PC2: {sum(pca.explained_variance_ratio_):.2%}")
    print(f"Number of data points: {len(pca_df)}")
    print(f"Features analyzed: {', '.join(features)}")
    
    return pca_df, pca

# -------------------------------
# Alternative: Division-based Analysis
# -------------------------------
def division_based_pca(df, features, title, save=True):
    """
    PCA analysis grouped by divisions for clearer visualization
    """
    # Define divisions (you may need to adjust based on your data)
    divisions = {
        'AL East': ['NYY', 'BOS', 'TOR', 'TB', 'BAL'],
        'AL Central': ['MIN', 'CWS', 'CLE', 'DET', 'KC'],
        'AL West': ['HOU', 'LAA', 'SEA', 'TEX', 'OAK'],
        'NL East': ['ATL', 'PHI', 'NYM', 'WSH', 'MIA'],
        'NL Central': ['MIL', 'STL', 'CHC', 'CIN', 'PIT'],
        'NL West': ['LAD', 'SD', 'SF', 'COL', 'ARI']
    }
    
    # Add division column
    df_with_div = df.copy()
    df_with_div['Division'] = 'Other'
    
    for division, teams in divisions.items():
        mask = df_with_div['Team'].isin(teams)
        df_with_div.loc[mask, 'Division'] = division
    
    # Perform PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_with_div[features])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Plot each division separately
    for i, (division, teams) in enumerate(divisions.items()):
        ax = axes[i]
        
        # Filter data for this division
        mask = df_with_div['Division'] == division
        div_data = df_with_div[mask]
        
        if len(div_data) > 0:
            div_pcs = principal_components[mask]
            
            ax.scatter(div_pcs[:, 0], div_pcs[:, 1], 
                      c=colors[i], alpha=0.7, s=80, edgecolors='white')
            
            # Add team labels
            for j, team in enumerate(div_data['Team']):
                ax.annotate(team, (div_pcs[j, 0], div_pcs[j, 1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
            
            ax.set_title(f'{division}', fontsize=12, fontweight='bold')
            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'{title} - By Division', fontsize=16, fontweight='bold', y=0.98)
    
    if save:
        path.joinpath('visuals').mkdir(parents=True, exist_ok=True)
        plt.savefig(path / 'visuals' / f'{title.lower().replace(" ", "_")}_by_division.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

# -------------------------------
# Execute Enhanced Analysis
# -------------------------------
print("Performing Enhanced PCA Analysis...")

# Batting analysis
bat_features = ['AVG', 'OBP', 'SLG', 'WAR']
bat_pca_df, bat_pca = enhanced_pca_analysis(bat_df, bat_features, 'Enhanced Batting PCA')

# Pitching analysis
pitch_features = ['ERA', 'WHIP', 'K/9', 'WAR']
pitch_pca_df, pitch_pca = enhanced_pca_analysis(pitch_df, pitch_features, 'Enhanced Pitching PCA')

# Division-based analysis
print("\nPerforming Division-based Analysis...")
division_based_pca(bat_df, bat_features, 'Batting PCA')
division_based_pca(pitch_df, pitch_features, 'Pitching PCA')
