import os

import torch
import torch.utils.dlpack
import cupy as cp

import cuml
from cuml.cluster import KMeans
from cuml.decomposition import PCA
from cuml.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

FILE_PATH = "./steering_vector/error_vectors.pt"
OUTPUT_DIR = "./results/figures/error_vector_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LAYERS = [str(i) for i in range(28)]
N_CLUSTERS = 4
RANDOM_STATE = 42


def load_vectors(file_path):
    return torch.load(file_path)


def extract_layer_vectors_gpu(data, layer_name):
    vectors_list = []
    valid_indices = []

    for idx, sample in enumerate(data):
        sv = sample.get("steering_vectors", {})
        if layer_name in sv:
            vectors_list.append(sv[layer_name].float())
            valid_indices.append(idx)

    if not vectors_list:
        return None, []

    stacked = torch.stack(vectors_list)
    if not stacked.is_cuda:
        stacked = stacked.cuda()
    return stacked, valid_indices


def to_cpu(x):
    if hasattr(x, "get"):
        return x.get()
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return x


def visualize_clustering_gpu(vectors_gpu, labels_gpu, layer_name, output_dir, expl_var_score):
    # PCA (2D) for quick view
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    pca2_gpu = pca2.fit_transform(vectors_gpu)

    # t-SNE on PCA-50 compressed features
    pca50 = PCA(n_components=50, random_state=RANDOM_STATE)
    vec50_gpu = pca50.fit_transform(vectors_gpu)

    tsne = TSNE(
        n_components=2,
        perplexity=50,
        n_neighbors=150,
        n_iter=2000,
        learning_rate=200,
        random_state=RANDOM_STATE,
    )
    tsne_gpu = tsne.fit_transform(vec50_gpu)

    pca2_cpu = to_cpu(pca2_gpu)
    tsne_cpu = to_cpu(tsne_gpu)
    labels_cpu = to_cpu(labels_gpu)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    sns.scatterplot(
        x=pca2_cpu[:, 0],
        y=pca2_cpu[:, 1],
        hue=labels_cpu,
        palette="viridis",
        s=15,
        alpha=0.6,
        ax=axes[0],
        legend="full",
    )
    axes[0].set_title(f"PCA - {layer_name} (Expl. Var: {expl_var_score:.2f}%)", fontsize=14)

    sns.scatterplot(
        x=tsne_cpu[:, 0],
        y=tsne_cpu[:, 1],
        hue=labels_cpu,
        palette="viridis",
        s=15,
        alpha=0.6,
        ax=axes[1],
        legend="full",
    )
    axes[1].set_title(f"t-SNE - {layer_name} (Top 5% clipped)", fontsize=14)

    plt.suptitle(f"Error Vector Analysis (K={N_CLUSTERS}) : {layer_name}", fontsize=18)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"cluster_vis_{layer_name}_k{N_CLUSTERS}.png")
    plt.savefig(save_path)
    plt.close(fig)


def main():
    data = load_vectors(FILE_PATH)

    clustering_results = {}
    representative_vectors_dict = {}

    for layer_name in TARGET_LAYERS:
        tensor_gpu, indices = extract_layer_vectors_gpu(data, layer_name)
        if tensor_gpu is None or tensor_gpu.numel() == 0:
            continue

        # Torch GPU tensor -> CuPy (zero-copy via DLPack)
        X_cupy = cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor_gpu))

        # Clip top 5% by L2 norm (outlier removal)
        norms = cp.linalg.norm(X_cupy, axis=1)
        cutoff = cp.percentile(norms, 95)
        mask = norms <= cutoff
        X_clean = X_cupy[mask]
        mask_cpu = mask.get()
        indices_clean = [indices[i] for i in range(len(indices)) if mask_cpu[i]]

        # Standardize then cluster
        scaler = cuml.preprocessing.StandardScaler()
        X_scaled_gpu = scaler.fit_transform(X_clean)

        # PCA variance (2D) for reference
        pca_check = PCA(n_components=2)
        pca_check.fit(X_scaled_gpu)
        expl_var = float(cp.sum(pca_check.explained_variance_ratio_) * 100)

        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_scaled_gpu)

        labels_gpu = kmeans.labels_
        centroids_gpu = kmeans.cluster_centers_

        # Representative vector per cluster: mean of original (clean) then L2-normalize
        layer_rep_vectors = []
        for k in range(N_CLUSTERS):
            cluster_samples = X_clean[labels_gpu == k]
            if cluster_samples.shape[0] == 0:
                layer_rep_vectors.append(torch.zeros(X_clean.shape[1]))
                continue

            mean_vec = cp.mean(cluster_samples, axis=0)
            vnorm = cp.linalg.norm(mean_vec)
            norm_vec = mean_vec / vnorm if vnorm > 0 else mean_vec
            layer_rep_vectors.append(torch.from_numpy(cp.asnumpy(norm_vec)))

        representative_vectors_dict[layer_name] = torch.stack(layer_rep_vectors)

        clustering_results[layer_name] = {
            "centroids": to_cpu(centroids_gpu),
            "labels": to_cpu(labels_gpu),
            "indices": indices_clean,
        }

        visualize_clustering_gpu(X_scaled_gpu, labels_gpu, layer_name, OUTPUT_DIR, expl_var)

        del tensor_gpu, X_cupy, X_clean, X_scaled_gpu, norms, mask, labels_gpu
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    torch.save(clustering_results, "./steering_vector/clustering_metadata.pt")
    torch.save(representative_vectors_dict, "./steering_vector/error_vectors_norm.pt")


if __name__ == "__main__":
    main()