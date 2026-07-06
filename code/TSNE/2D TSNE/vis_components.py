import os
import argparse
import warnings
import time
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sys.path.append("../")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP is not installed. Install it with: pip install umap-learn")

try:
    from sklearn.manifold import Isomap
    ISOMAP_AVAILABLE = True
except ImportError:
    ISOMAP_AVAILABLE = False

from networks.species.Distangle_knowledge import ImprovedCompositeModelWithAttention
from networks.species.dataloader_knowledge import build_dataloaders


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def extract_components_and_original(
    model,
    dataset,
    device,
    batch_size=64,
    disease_start=None,
    disease_dim=None,
):
    """Extract the original input and three model components."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    x_clr_list = []
    x_base_list = []
    delta_d_list = []
    delta_o_list = []
    disease_labels_list = []
    subject_ids_list = []

    print("\nExtracting original input and model components...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch["x_clr"].to(device)
            c = batch["c"].to(device)
            idx = batch["idx"].cpu().numpy()

            output = model(
                x,
                c,
                disease_start=disease_start,
                disease_dim=disease_dim,
                training_stage="full",
            )

            x_clr_list.append(x.cpu())
            x_base_list.append(output["x_base"].cpu())
            delta_d_list.append(output["delta_d"].cpu())
            delta_o_list.append(output["delta_o"].cpu())

            batch_c = batch["c"].cpu().numpy()
            if disease_start is not None and disease_dim is not None:
                disease_onehot = batch_c[:, disease_start:disease_start + disease_dim]
                disease_label = np.argmax(disease_onehot, axis=1)
            else:
                disease_label = (batch_c[:, 0] > 0.5).astype(int)

            disease_labels_list.append(disease_label)
            subject_ids_list.append(idx)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches")

    x_clr_all = torch.cat(x_clr_list, dim=0).numpy()
    x_base_all = torch.cat(x_base_list, dim=0).numpy()
    delta_d_all = torch.cat(delta_d_list, dim=0).numpy()
    delta_o_all = torch.cat(delta_o_list, dim=0).numpy()
    disease_labels_all = np.concatenate(disease_labels_list, axis=0)
    subject_ids_all = np.concatenate(subject_ids_list, axis=0)

    n_asd = int(np.sum(disease_labels_all))
    n_nt = int(len(disease_labels_all) - n_asd)

    print("Extraction completed:")
    print(f"  x_clr shape: {x_clr_all.shape}")
    print(f"  x_base shape: {x_base_all.shape}")
    print(f"  delta_d shape: {delta_d_all.shape}")
    print(f"  delta_o shape: {delta_o_all.shape}")
    print(f"  Label distribution: ASD={n_asd}, NT={n_nt}")

    return x_clr_all, x_base_all, delta_d_all, delta_o_all, disease_labels_all, subject_ids_all


def reduce_dimension(data, method="TSNE", **kwargs):
    data_scaled = data

    if method == "PCA":
        n_components = kwargs.get("n_components", 2)
        pca = PCA(n_components=n_components, random_state=42)
        data_reduced = pca.fit_transform(data_scaled)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {sum(pca.explained_variance_ratio_):.3f}")
        return data_reduced, pca

    if method == "TSNE":
        perplexity = kwargs.get("perplexity", 30)
        learning_rate = kwargs.get("learning_rate", 200)

        if data.shape[1] > 50:
            print("  High-dimensional input detected. Applying PCA to 50 dimensions before t-SNE.")
            pca_pre = PCA(n_components=min(50, data.shape[1]), random_state=42)
            data_scaled = pca_pre.fit_transform(data_scaled)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            learning_rate=learning_rate,
            max_iter=1000,
        )
        data_reduced = tsne.fit_transform(data_scaled)
        return data_reduced, tsne

    if method == "UMAP":
        if not UMAP_AVAILABLE:
            print("UMAP is unavailable. Falling back to PCA.")
            return reduce_dimension(data, method="PCA", **kwargs)

        n_neighbors = kwargs.get("n_neighbors", 15)
        min_dist = kwargs.get("min_dist", 0.1)
        metric = kwargs.get("metric", "euclidean")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )
        data_reduced = reducer.fit_transform(data_scaled)
        return data_reduced, reducer

    if method == "ISOMAP":
        if not ISOMAP_AVAILABLE:
            print("ISOMAP is unavailable. Falling back to PCA.")
            return reduce_dimension(data, method="PCA", **kwargs)

        n_neighbors = kwargs.get("n_neighbors", 5)
        n_components = kwargs.get("n_components", 2)

        isomap = Isomap(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_jobs=-1,
        )
        data_reduced = isomap.fit_transform(data_scaled)
        return data_reduced, isomap

    print(f"Unknown dimensionality reduction method: {method}. Falling back to PCA.")
    return reduce_dimension(data, method="PCA", **kwargs)


def plot_single_component(data, disease_labels, component_name, output_dir, method="TSNE", timestamp=None, **kwargs):
    print(f"\nRunning {method} for {component_name}...")
    data_2d, _ = reduce_dimension(data, method=method, **kwargs)

    plt.figure(figsize=(8, 6))

    colors = ["#0066FF", "#FF0000"]
    asd_indices = disease_labels == 1
    nt_indices = disease_labels == 0

    if np.any(asd_indices):
        plt.scatter(
            data_2d[asd_indices, 0],
            data_2d[asd_indices, 1],
            c=colors[1],
            label="ASD",
            alpha=0.7,
            s=60,
            edgecolors=None,
            linewidth=0.5,
        )

    if np.any(nt_indices):
        plt.scatter(
            data_2d[nt_indices, 0],
            data_2d[nt_indices, 1],
            c=colors[0],
            label="NT",
            alpha=0.7,
            s=60,
            edgecolors=None,
            linewidth=0.5,
        )

    plt.title(f"{component_name} - {method}", fontsize=14, fontweight="bold")
    plt.xlabel(f"{method} Component 1")
    plt.ylabel(f"{method} Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = os.path.join(output_dir, f"{component_name}_{method.lower()}_{timestamp}.pdf")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the original microbiome input and disentangled model components."
    )

    parser.add_argument("--model_path", type=str, default="XXX")
    parser.add_argument("--abundance_csv", type=str, default="XXX")
    parser.add_argument("--meta_csv", type=str, default="XXX")
    parser.add_argument("--metadata_txt", type=str, default="XXX")
    parser.add_argument("--species_prior_csv", type=str, default="XXX")

    parser.add_argument(
        "--method",
        type=str,
        default="TSNE",
        choices=["PCA", "TSNE", "UMAP", "ISOMAP"],
        help="Dimensionality reduction method.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./component_visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity.")
    parser.add_argument("--learning_rate", type=int, default=200, help="t-SNE learning rate.")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP or ISOMAP.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP minimum distance.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)

    print("Loading model and dataset...")
    _, val_loader, _ = build_dataloaders(
        abundance_csv=args.abundance_csv,
        meta_csv=args.meta_csv,
        metadata_txt=args.metadata_txt,
        species_prior_csv=args.species_prior_csv,
        save_species_csv=None,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        val_split=0.1,
        seed=42,
        transform_clr=True,
        subject_intersection_only=True,
        prevalence_threshold=0.1,
        min_nonzero_species=20,
        standardize_continuous=True,
        include_cont_missing_indicator=True,
        numeric_discrete_mode="minmax",
        device=None,
    )

    dataset_for_analysis = val_loader.dataset

    checkpoint = torch.load(args.model_path, map_location=device)
    info_ckpt = checkpoint["info"]
    model_args = checkpoint.get("args", {})

    disease_start = info_ckpt.get("disease_start", 0)
    disease_dim = info_ckpt.get("disease_dim", 2)

    prior_weight = torch.from_numpy(
        info_ckpt.get("prior_weight", np.zeros(info_ckpt["X_dim"], dtype=np.float32))
    ).to(device)
    prior_info = {
        "prior_weight": prior_weight,
        "pos_mask": info_ckpt.get("prior_pos_mask", np.zeros(info_ckpt["X_dim"], dtype=np.float32)),
        "neg_mask": info_ckpt.get("prior_neg_mask", np.zeros(info_ckpt["X_dim"], dtype=np.float32)),
        "neu_mask": info_ckpt.get("prior_neu_mask", np.zeros(info_ckpt["X_dim"], dtype=np.float32)),
    }

    model = ImprovedCompositeModelWithAttention(
        x_dim=info_ckpt["X_dim"],
        c_dim=info_ckpt["C_dim"],
        disease_dim=disease_dim,
        others_dim=info_ckpt.get("others_dim", info_ckpt["C_dim"] - disease_dim),
        z_dim=model_args.get("z_dim", 512),
        u_dim=model_args.get("u_dim", 32),
        enc_hidden_dims=[model_args.get("hid", 1024), model_args.get("hid", 1024) // 2],
        dec_hidden_dims=[model_args.get("hid", 1024) // 2, model_args.get("hid", 1024)],
        dropout=model_args.get("dropout", 0.3),
        shift_dropout=model_args.get("shift_dropout", 0.4),
        grl_lambda=model_args.get("grl_lambda", 0.1),
        adversary=None,
        prior_info=prior_info,
        use_attention=model_args.get("use_prior_attention", True),
        attention_heads=model_args.get("attention_heads", 5),
        attn_beta=model_args.get("attn_beta", 0.7),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    x_clr, x_base, delta_d, delta_o, disease_labels, _ = extract_components_and_original(
        model,
        dataset_for_analysis,
        device,
        args.batch_size,
        disease_start,
        disease_dim,
    )

    reduction_kwargs = {}
    if args.method == "TSNE":
        reduction_kwargs["perplexity"] = args.perplexity
        reduction_kwargs["learning_rate"] = args.learning_rate
    elif args.method == "UMAP":
        reduction_kwargs["n_neighbors"] = args.n_neighbors
        reduction_kwargs["min_dist"] = args.min_dist
    elif args.method == "ISOMAP":
        reduction_kwargs["n_neighbors"] = args.n_neighbors

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    components = {
        "original": x_clr,
        "x_base": x_base,
        "delta_d": delta_d,
        "delta_o": delta_o,
    }

    for name, data in components.items():
        plot_single_component(
            data,
            disease_labels,
            name,
            args.output_dir,
            method=args.method,
            timestamp=timestamp,
            **reduction_kwargs,
        )

    print("\n" + "=" * 60)
    print(f"Visualization completed with {args.method}.")
    print(f"Generated {len(components)} figures in: {args.output_dir}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
