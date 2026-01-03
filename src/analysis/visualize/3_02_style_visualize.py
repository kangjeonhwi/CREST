import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# RAPIDS required
try:
    import cupy as cp
    import rmm
    from cuml.cluster import KMeans
    from cuml.manifold import UMAP
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError("RAPIDS environment (cuML, cuPy, RMM) is required.")

PROCESSED_DIR = "./steering_vector/styles/processed_vectors"
OUTPUT_DIR = "./results/style_vector_analysis"

GRID_LAYERS = list(range(0, 30, 3))
FOCUS_LAYERS = [15, 18, 21, 24]

N_CLUSTERS = 10
RANDOM_STATE = 42

sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = False


def setup_rmm():
    """Initialize RMM pool allocator if available."""
    try:
        rmm.reinitialize(pool_allocator=True, initial_pool_size=None, managed_memory=False)
    except Exception:
        pass


def free_gpu():
    """Release CuPy memory pool to avoid fragmentation across repeated fits."""
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


def load_layer_data(metadata, layer_idx):
    """Load vectors for a single layer, then center and L2-normalize per sample."""
    X_list, y_list = [], []
    for meta in metadata:
        arr = np.load(meta["path"], mmap_mode="r")
        layer_vecs = arr[:, layer_idx, :].astype(np.float32)
        X_list.append(layer_vecs)
        y_list.extend([meta["name"]] * meta["count"])
        del arr

    if not X_list:
        return None, None

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)

    X = X - np.mean(X, axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-8)
    return X, y


def plot_layer_evolution_grid(metadata, layers_idx):
    print(f"Figure 1: UMAP grid ({len(layers_idx)} layers)")
    n_cols = 3
    n_rows = (len(layers_idx) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True
    )
    axes = axes.flatten()

    styles = sorted({m["name"] for m in metadata})
    palette = sns.color_palette("tab10", n_colors=len(styles))
    style_color_map = dict(zip(styles, palette))

    for idx, layer in enumerate(tqdm(layers_idx, desc="Layers")):
        ax = axes[idx]
        free_gpu()

        X_cpu, y_labels = load_layer_data(metadata, layer)
        if X_cpu is None:
            ax.axis("off")
            continue

        X_gpu = cp.asarray(X_cpu)

        umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=RANDOM_STATE)
        embedding = cp.asnumpy(umap.fit_transform(X_gpu))

        kmeans = KMeans(n_clusters=len(styles), random_state=RANDOM_STATE)
        preds = kmeans.fit_predict(X_gpu)
        ari = adjusted_rand_score(y_labels, cp.asnumpy(preds))

        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=y_labels,
            palette=style_color_map,
            s=15,
            alpha=0.6,
            edgecolor=None,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"Layer {layer} (ARI: {ari:.3f})", fontsize=14, fontweight="bold")
        ax.axis("off")

    for i in range(len(layers_idx), len(axes)):
        axes[i].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=l)
        for l, c in style_color_map.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(styles)),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "Figure_1_Layer_Evolution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_similarity_heatmap(metadata, target_layer):
    print(f"Figure 2: cosine similarity heatmap (layer={target_layer})")
    free_gpu()

    X_cpu, y_labels = load_layer_data(metadata, target_layer)
    if X_cpu is None:
        return

    df = pd.DataFrame(X_cpu)
    df["label"] = y_labels
    centroids = df.groupby("label").mean()

    sim_matrix = cosine_similarity(centroids.values)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=1.0,
        xticklabels=centroids.index,
        yticklabels=centroids.index,
        square=True,
    )
    plt.title(f"Style vector cosine similarity (Layer {target_layer})", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"Figure_2_Heatmap_L{target_layer}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_kmeans_visual_check(metadata, target_layer, k=10):
    print(f"Figure 3: GT vs K-Means (layer={target_layer}, k={k})")
    free_gpu()

    X_cpu, y_true = load_layer_data(metadata, target_layer)
    if X_cpu is None:
        return

    X_gpu = cp.asarray(X_cpu)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    y_pred = cp.asnumpy(kmeans.fit_predict(X_gpu))

    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=RANDOM_STATE)
    embedding = cp.asnumpy(umap.fit_transform(X_gpu))

    ari_score = adjusted_rand_score(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=y_true,
        palette="tab10",
        s=20,
        alpha=0.7,
        edgecolor=None,
        ax=axes[0],
        legend="full",
    )
    axes[0].set_title(f"(A) Ground truth (Layer {target_layer})", fontsize=18, fontweight="bold")
    axes[0].axis("off")
    axes[0].legend(loc="lower right", title="Style", frameon=True)

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=y_pred,
        palette="tab10",
        s=20,
        alpha=0.7,
        edgecolor=None,
        ax=axes[1],
        legend="full",
    )
    axes[1].set_title(f"(B) K-Means (k={k}, ARI={ari_score:.3f})", fontsize=18, fontweight="bold")
    axes[1].axis("off")
    axes[1].legend(loc="lower right", title="Cluster", frameon=True)

    plt.suptitle(
        f"Layer {target_layer}: ground truth vs unsupervised clustering",
        fontsize=20,
        fontweight="bold",
        y=1.05,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"Figure_3_KMeans_Visual_Check_L{target_layer}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def run_analysis():
    setup_rmm()

    meta_path = os.path.join(PROCESSED_DIR, "metadata.npy")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    metadata = np.load(meta_path, allow_pickle=True)

    plot_layer_evolution_grid(metadata, GRID_LAYERS)

    print(f"Deep analysis layers: {FOCUS_LAYERS}")
    for layer in FOCUS_LAYERS:
        plot_similarity_heatmap(metadata, layer)
        plot_kmeans_visual_check(metadata, layer, k=N_CLUSTERS)


if __name__ == "__main__":
    run_analysis()