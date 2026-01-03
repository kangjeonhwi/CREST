import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
INPUT_FILE = "./src/analysis/results/error_grid/steering_experiment_MATH_Robust_20260102_050952_merged.jsonl"
OUTPUT_DIR = "./results/figures/error_grid_analysis"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    """Load JSONL data into a Pandas DataFrame."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            row = entry["config"]
            row["accuracy"] = entry["metrics"]["accuracy"]
            data.append(row)
    return pd.DataFrame(data)

def analyze_best_configs(df):
    """Find and save top performing configurations."""
    # Global best
    best_idx = df["accuracy"].idxmax()
    best_config = df.iloc[best_idx]
    
    # Top 10 configurations
    top_10 = df.sort_values(by="accuracy", ascending=False).head(10)
    
    print(f"=== Best Configuration ===")
    print(best_config)
    
    csv_path = os.path.join(OUTPUT_DIR, "top_10_configs.csv")
    top_10.to_csv(csv_path, index=False)
    print(f"\nTop 10 configurations saved to: {csv_path}")

def plot_heatmap_per_vector(df):
    """Generate heatmaps: Layer vs Alpha for each Vector."""
    unique_vectors = sorted(df["vector"].unique())
    
    # Create subplots based on number of vectors
    fig, axes = plt.subplots(1, len(unique_vectors), figsize=(6 * len(unique_vectors), 6), sharey=True)
    if len(unique_vectors) == 1: axes = [axes]
    
    # Find global min/max for consistent colorbar
    vmin, vmax = df["accuracy"].min(), df["accuracy"].max()
    
    for i, vec_idx in enumerate(unique_vectors):
        subset = df[df["vector"] == vec_idx]
        pivot = subset.pivot(index="layer", columns="alpha", values="accuracy")
        
        sns.heatmap(
            pivot, 
            ax=axes[i], 
            cmap="RdYlGn", 
            annot=True, 
            fmt=".2f", 
            vmin=vmin, 
            vmax=vmax,
            cbar=(i == len(unique_vectors) - 1) # Only show colorbar on last plot
        )
        axes[i].set_title(f"Vector {vec_idx} Impact")
        axes[i].set_xlabel("Alpha (Steering Strength)")
        axes[i].set_ylabel("Layer")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "heatmap_layer_alpha_per_vector.png")
    plt.savefig(save_path)
    print(f"Heatmap saved to: {save_path}")

def plot_layer_sensitivity(df):
    """Analyze which layer is most sensitive/effective (Max Accuracy per Layer)."""
    # Max accuracy achievable per layer (across all alphas/vectors)
    layer_stats = df.groupby("layer")["accuracy"].max().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="layer", y="accuracy", data=layer_stats, palette="viridis")
    plt.title("Max Accuracy Achievable per Layer")
    plt.ylim(0, 1.0)
    
    for index, row in layer_stats.iterrows():
        plt.text(index, row.accuracy + 0.01, f"{row.accuracy:.2f}", ha="center", color="black")
        
    save_path = os.path.join(OUTPUT_DIR, "max_accuracy_per_layer.png")
    plt.savefig(save_path)
    print(f"Layer sensitivity plot saved to: {save_path}")

def plot_alpha_trend(df):
    """Line plot showing global trend of Alpha vs Accuracy."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="alpha", y="accuracy", hue="vector", style="layer", markers=True, palette="tab10")
    
    plt.title("Steering Effect: Accuracy vs Alpha")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axvline(0, color="red", linestyle=":", label="Baseline")
    
    save_path = os.path.join(OUTPUT_DIR, "alpha_accuracy_trend.png")
    plt.savefig(save_path)
    print(f"Alpha trend plot saved to: {save_path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found at {INPUT_FILE}")
        return

    print("Loading data...")
    df = load_data(INPUT_FILE)
    
    print("Analyzing best configurations...")
    analyze_best_configs(df)
    
    print("Generating visualizations...")
    plot_heatmap_per_vector(df)     # Detailed grid view
    plot_layer_sensitivity(df)      # Layer effectiveness
    plot_alpha_trend(df)            # Overall trend
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()