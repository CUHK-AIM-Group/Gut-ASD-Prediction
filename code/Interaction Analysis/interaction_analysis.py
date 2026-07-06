#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute species-level interaction matrices for a trained PyTorch classifier using Captum GradientShap.

This script estimates pairwise microbial interactions by measuring how the SHAP attribution of one
species changes when another species is fixed to its background mean. It then generates consensus,
group-specific, and ASD-vs-NT difference interaction visualizations.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from tqdm import tqdm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from captum.attr import GradientShap

from cls import ASCClassifier
from dataloader_cls import ASCClassificationDataset


# ==================== Configuration ====================

DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
SEED = 42

MODEL_CHECKPOINT_PATH = "XXX"
ABUNDANCE_CSV = "XXX"
META_CSV = "XXX"
IMPORTANT_SPECIES_TXT = "XXX"
METADATA_SELECT_TXT = "XXX"

BACKGROUND_SAMPLES = 50
CONSISTENCY_THRESHOLD = 0.7
CLASS_OF_INTEREST = 1

BATCH_SIZE = 64
GS_N_SAMPLES = 200


# ==================== Utility Functions ====================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_data():
    """
    Load the trained model, dataset, species abundance matrix, metadata matrix,
    labels, species names, and metadata usage flag.
    """
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "args" in checkpoint:
        ckpt_args = checkpoint["args"]
        gut_latent_dim = ckpt_args.get("gut_latent_dim", 128)
        metadata_latent_dim = ckpt_args.get("metadata_latent_dim", 64)
        classifier_hidden_dims = ckpt_args.get("classifier_hidden_dims", [128, 64, 32])
        dropout = ckpt_args.get("dropout", 0.3)
        use_batch_norm = ckpt_args.get("use_batch_norm", False)
        use_metadata = ckpt_args.get("use_metadata", True)
    else:
        gut_latent_dim = 128
        metadata_latent_dim = 64
        classifier_hidden_dims = [128, 64, 32]
        dropout = 0.3
        use_batch_norm = False
        use_metadata = True

    print(f"Model configuration: use_metadata={use_metadata}")

    dataset = ASCClassificationDataset(
        abundance_csv=ABUNDANCE_CSV,
        meta_csv=META_CSV,
        important_species_txt=IMPORTANT_SPECIES_TXT,
        metadata_select_txt=METADATA_SELECT_TXT if use_metadata else None,
        disease_col_name="group_x",
        disease_mapping={"2": 0, "1": 1},
        transform_clr=True,
        clr_eps=1e-6,
        remove_missing_disease=True,
        remove_disease_3=True,
        use_metadata=use_metadata,
        device=None,
    )

    species_names = dataset.info["species_names"]
    X = dataset.X_gut.numpy()
    y = dataset.y.numpy().astype(np.float32)
    X_meta = dataset.X_metadata.numpy() if use_metadata else None

    metadata_dim = X_meta.shape[1] if X_meta is not None else 0

    print(
        f"Data loaded: {X.shape[0]} samples, "
        f"{X.shape[1]} species, metadata dimension={metadata_dim}"
    )

    model = ASCClassifier(
        gut_dim=dataset.info["gut_dim"],
        metadata_dim=dataset.info["metadata_dim"],
        gut_latent_dim=gut_latent_dim,
        metadata_latent_dim=metadata_latent_dim,
        classifier_hidden_dims=classifier_hidden_dims,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        use_metadata=use_metadata,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "best_state" in checkpoint:
            state_dict = checkpoint["best_state"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model, X, X_meta, y, species_names, use_metadata


# ==================== Captum-Based Interaction Estimation ====================

class WrappedModel(nn.Module):
    """
    Wrap the original classifier so Captum can receive a single concatenated input tensor.
    """

    def __init__(self, original_model, gut_dim, meta_dim, use_metadata):
        super().__init__()
        self.original_model = original_model
        self.gut_dim = gut_dim
        self.meta_dim = meta_dim
        self.use_metadata = use_metadata

    def forward(self, x):
        gut_part = x[:, :self.gut_dim]

        if self.use_metadata:
            meta_part = x[:, self.gut_dim:]
        else:
            meta_part = torch.zeros(gut_part.size(0), 0, device=x.device)

        logits = self.original_model(gut_part, meta_part)["logits"]

        return logits.squeeze(-1)


def compute_shap_main_effect(
    model_wrapped,
    data_combined,
    baselines,
    batch_size=32,
    n_samples=50,
):
    """
    Compute GradientShap attributions for all input features.
    """
    gs = GradientShap(model_wrapped)

    n_total = data_combined.shape[0]
    all_shap = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        x_batch = data_combined[start:end]

        shap_batch = gs.attribute(
            x_batch,
            baselines=baselines,
            n_samples=n_samples,
            return_convergence_delta=False,
        )

        all_shap.append(shap_batch.detach().cpu().numpy())

    shap_all = np.concatenate(all_shap, axis=0)

    return shap_all


def captum_interaction_values(
    model,
    X_gut,
    X_meta,
    background_gut,
    background_meta,
    use_metadata,
    batch_size=16,
    gs_n_samples=50,
):
    """
    Estimate pairwise species interaction values by recomputing SHAP values after
    fixing each species feature to its background mean.
    """
    device = next(model.parameters()).device

    gut_dim = X_gut.shape[1]
    meta_dim = X_meta.shape[1] if use_metadata else 0

    wrapped_model = WrappedModel(
        original_model=model,
        gut_dim=gut_dim,
        meta_dim=meta_dim,
        use_metadata=use_metadata,
    ).to(device)

    wrapped_model.eval()

    background_gut_t = torch.from_numpy(background_gut.astype(np.float32)).to(device)

    if use_metadata:
        background_meta_t = torch.from_numpy(background_meta.astype(np.float32)).to(device)
        meta_mean = background_meta_t.mean(dim=0, keepdim=True)

        background_combined = torch.cat(
            [
                background_gut_t,
                meta_mean.repeat(background_gut_t.size(0), 1),
            ],
            dim=1,
        )
    else:
        background_combined = background_gut_t
        meta_mean = None

    test_gut_t = torch.from_numpy(X_gut.astype(np.float32)).to(device)

    if use_metadata:
        test_meta_t = meta_mean.repeat(test_gut_t.size(0), 1).to(device)
        test_combined = torch.cat([test_gut_t, test_meta_t], dim=1)
    else:
        test_combined = test_gut_t
        test_meta_t = None

    print("Computing original SHAP main effects...")

    shap_original_all = compute_shap_main_effect(
        wrapped_model,
        test_combined,
        background_combined,
        batch_size=batch_size,
        n_samples=gs_n_samples,
    )

    shap_original_gut = shap_original_all[:, :gut_dim]

    n_samples = X_gut.shape[0]
    interaction_vals = np.zeros((n_samples, gut_dim, gut_dim), dtype=np.float32)

    for j in tqdm(range(gut_dim), desc="Computing interaction matrix"):
        X_gut_fixed = X_gut.copy()

        fixed_val = np.mean(background_gut[:, j])
        X_gut_fixed[:, j] = fixed_val

        X_gut_fixed_t = torch.from_numpy(X_gut_fixed.astype(np.float32)).to(device)

        if use_metadata:
            test_combined_fixed = torch.cat([X_gut_fixed_t, test_meta_t], dim=1)
        else:
            test_combined_fixed = X_gut_fixed_t

        shap_cond_all = compute_shap_main_effect(
            wrapped_model,
            test_combined_fixed,
            background_combined,
            batch_size=batch_size,
            n_samples=gs_n_samples,
        )

        shap_cond_gut = shap_cond_all[:, :gut_dim]

        diff = shap_original_gut - shap_cond_gut
        interaction_vals[:, :, j] += diff

    interaction_vals = (
        interaction_vals + np.transpose(interaction_vals, axes=(0, 2, 1))
    ) / 2.0

    return interaction_vals


# ==================== Network Visualization ====================

def plot_group_network(
    interaction_subset,
    group_name,
    species_names,
    threshold=0.01,
    figsize=(12, 10),
    show_edge_labels=False,
    edge_labels_fmt="{:.3f}",
):
    """
    Plot a group-specific microbial interaction network and save the edge list as CSV.
    """
    mean_interaction = interaction_subset.mean(axis=0)
    np.fill_diagonal(mean_interaction, 0)

    graph = nx.Graph()
    n_species = len(species_names)

    for i, name in enumerate(species_names):
        graph.add_node(i, label=name)

    edge_values = {}

    for i in range(n_species):
        for j in range(i + 1, n_species):
            val = mean_interaction[i, j]

            if abs(val) >= threshold:
                graph.add_edge(i, j, weight=abs(val), sign=np.sign(val))
                edge_values[(i, j)] = val

    if graph.number_of_edges() == 0:
        print(f"No edges passed the threshold {threshold} for group {group_name}.")
        return

    plt.figure(figsize=figsize)

    pos = nx.circular_layout(graph)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="lightblue",
        node_size=2000,
        alpha=0.9,
    )

    positive_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d["sign"] == 1
    ]

    negative_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d["sign"] == -1
    ]

    if positive_edges:
        positive_weights = [graph[u][v]["weight"] for u, v in positive_edges]
        max_weight = max(positive_weights)
        widths = [0.5 + 2.5 * w / max_weight for w in positive_weights]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=positive_edges,
            edge_color="green",
            width=widths,
            alpha=0.7,
        )

    if negative_edges:
        negative_weights = [graph[u][v]["weight"] for u, v in negative_edges]
        max_weight = max(negative_weights)
        widths = [0.5 + 2.5 * w / max_weight for w in negative_weights]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=negative_edges,
            edge_color="red",
            width=widths,
            alpha=0.7,
        )

    labels = {i: name for i, name in enumerate(species_names)}

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=10,
    )

    if show_edge_labels:
        edge_labels = {
            edge: edge_labels_fmt.format(edge_values[edge])
            for edge in graph.edges()
        }

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
        )

    legend_elements = [
        Patch(facecolor="green", label="Positive interaction"),
        Patch(facecolor="red", label="Negative interaction"),
    ]

    plt.legend(handles=legend_elements, loc="upper right")
    plt.title(f"{group_name} Microbial Interaction Network (|mean| >= {threshold})")
    plt.axis("off")
    plt.tight_layout()

    png_filename = f"interaction_network_{group_name}.png"
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(
        f"{group_name} network plot saved to {png_filename}. "
        f"Number of edges: {graph.number_of_edges()}."
    )

    edges_data = []

    for (u, v), val in edge_values.items():
        edges_data.append(
            {
                "Species_A": species_names[u],
                "Species_B": species_names[v],
                "Mean_interaction_strength": val,
                "Absolute_weight": abs(val),
                "Direction": "positive" if val > 0 else "negative",
            }
        )

    df_edges = pd.DataFrame(edges_data)

    csv_filename = f"interaction_network_{group_name}.csv"
    df_edges.to_csv(csv_filename, index=False)

    print(f"{group_name} edge list saved to {csv_filename}.")


def plot_difference_network(
    interaction_vals_asd,
    interaction_vals_nt,
    species_names,
    threshold=0.01,
    figsize=(12, 10),
    show_edge_labels=False,
    edge_labels_fmt="{:.3f}",
):
    """
    Plot the ASD-vs-NT interaction difference network.
    Edge value = mean interaction in ASD - mean interaction in NT.
    """
    mean_asd = interaction_vals_asd.mean(axis=0)
    mean_nt = interaction_vals_nt.mean(axis=0)

    diff = mean_asd - mean_nt
    np.fill_diagonal(diff, 0)

    graph = nx.Graph()
    n_species = len(species_names)

    for i, name in enumerate(species_names):
        graph.add_node(i, label=name)

    edge_values = {}

    for i in range(n_species):
        for j in range(i + 1, n_species):
            val = diff[i, j]

            if abs(val) >= threshold:
                graph.add_edge(i, j, weight=abs(val), sign=np.sign(val))
                edge_values[(i, j)] = val

    if graph.number_of_edges() == 0:
        print(
            f"No edges passed the difference threshold {threshold}. "
            "Consider lowering the threshold or checking the input data."
        )
        return

    plt.figure(figsize=figsize)

    pos = nx.circular_layout(graph)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="lightgray",
        node_size=2000,
        alpha=0.9,
    )

    asd_stronger_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d["sign"] > 0
    ]

    nt_stronger_edges = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d["sign"] < 0
    ]

    if asd_stronger_edges:
        weights_asd = [graph[u][v]["weight"] for u, v in asd_stronger_edges]
        max_weight = max(weights_asd)
        widths = [0.5 + 2.5 * w / max_weight for w in weights_asd]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=asd_stronger_edges,
            edge_color="blue",
            width=widths,
            alpha=0.7,
            label="ASD > NT",
        )

    if nt_stronger_edges:
        weights_nt = [graph[u][v]["weight"] for u, v in nt_stronger_edges]
        max_weight = max(weights_nt)
        widths = [0.5 + 2.5 * w / max_weight for w in weights_nt]

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=nt_stronger_edges,
            edge_color="red",
            width=widths,
            alpha=0.7,
            label="NT > ASD",
        )

    labels = {i: name for i, name in enumerate(species_names)}

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=10,
    )

    if show_edge_labels:
        edge_labels = {
            edge: edge_labels_fmt.format(edge_values[edge])
            for edge in graph.edges()
        }

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
        )

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="ASD stronger"),
        Line2D([0], [0], color="red", lw=2, label="NT stronger"),
    ]

    plt.legend(handles=legend_elements, loc="upper right")
    plt.title(f"ASD vs NT Interaction Difference Network (|diff| >= {threshold})")
    plt.axis("off")
    plt.tight_layout()

    png_filename = "interaction_network_difference.png"
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(
        f"Difference network plot saved to {png_filename}. "
        f"Number of edges: {graph.number_of_edges()}."
    )

    edges_data = []

    for (u, v), val in edge_values.items():
        edges_data.append(
            {
                "Species_A": species_names[u],
                "Species_B": species_names[v],
                "Difference_ASD_minus_NT": val,
                "Absolute_weight": abs(val),
                "Stronger_group": "ASD" if val > 0 else "NT",
            }
        )

    df_edges = pd.DataFrame(edges_data)

    csv_filename = "interaction_network_difference.csv"
    df_edges.to_csv(csv_filename, index=False)

    print(f"Difference network edge list saved to {csv_filename}.")


# ==================== Main Analysis ====================

def main():
    set_seed(SEED)

    model, X, X_meta, y, species_names, use_metadata = load_model_and_data()

    gut_dim = X.shape[1]
    n_total = X.shape[0]

    background_indices = np.random.choice(
        n_total,
        size=min(BACKGROUND_SAMPLES, n_total),
        replace=False,
    )

    background_gut = X[background_indices]
    background_meta = X_meta[background_indices] if use_metadata else None

    print("Computing interaction matrix with Captum. This may take a long time.")

    interaction_vals = captum_interaction_values(
        model=model,
        X_gut=X,
        X_meta=X_meta,
        background_gut=background_gut,
        background_meta=background_meta,
        use_metadata=use_metadata,
        batch_size=BATCH_SIZE,
        gs_n_samples=GS_N_SAMPLES,
    )

    print(f"Interaction matrix shape: {interaction_vals.shape}")

    print("\nBuilding consensus graph...")

    pair_info = {}

    for i in range(gut_dim):
        for j in range(i + 1, gut_dim):
            interaction_ij = interaction_vals[:, i, j]

            positive_ratio = np.mean(interaction_ij > 0)
            negative_ratio = np.mean(interaction_ij < 0)

            mean_positive = (
                np.mean(interaction_ij[interaction_ij > 0])
                if np.any(interaction_ij > 0)
                else 0.0
            )

            mean_negative = (
                np.mean(np.abs(interaction_ij[interaction_ij < 0]))
                if np.any(interaction_ij < 0)
                else 0.0
            )

            pair_info[(i, j)] = {
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "mean_positive": mean_positive,
                "mean_negative": mean_negative,
                "consensus_direction": None,
                "consensus_strength": None,
            }

            if positive_ratio >= CONSISTENCY_THRESHOLD:
                pair_info[(i, j)]["consensus_direction"] = "positive"
                pair_info[(i, j)]["consensus_strength"] = mean_positive

            elif negative_ratio >= CONSISTENCY_THRESHOLD:
                pair_info[(i, j)]["consensus_direction"] = "negative"
                pair_info[(i, j)]["consensus_strength"] = mean_negative

    consensus_graph = nx.Graph()

    for k in range(gut_dim):
        consensus_graph.add_node(k, label=species_names[k])

    for (i, j), info in pair_info.items():
        if info["consensus_direction"] is not None:
            consensus_graph.add_edge(
                i,
                j,
                direction=info["consensus_direction"],
                weight=info["consensus_strength"],
            )

    if consensus_graph.edges():
        plt.figure(figsize=(8, 7))

        pos = nx.circular_layout(consensus_graph)

        nx.draw_networkx_nodes(
            consensus_graph,
            pos,
            node_color="lightblue",
            node_size=2000,
        )

        edges = consensus_graph.edges()

        edge_colors = [
            "green" if consensus_graph[u][v]["direction"] == "positive" else "red"
            for u, v in edges
        ]

        weights = [
            consensus_graph[u][v]["weight"]
            for u, v in edges
        ]

        if max(weights) > 0:
            edge_widths = [0.5 + 2 * w / max(weights) for w in weights]
        else:
            edge_widths = [1.0] * len(weights)

        nx.draw_networkx_edges(
            consensus_graph,
            pos,
            edgelist=edges,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.8,
        )

        labels = {
            k: species_names[k]
            for k in range(gut_dim)
        }

        nx.draw_networkx_labels(
            consensus_graph,
            pos,
            labels=labels,
            font_size=12,
        )

        legend_elements = [
            Patch(facecolor="green", label="Positive consensus (>=70%)"),
            Patch(facecolor="red", label="Negative consensus (>=70%)"),
        ]

        plt.legend(handles=legend_elements, loc="upper right")
        plt.title(f"Consensus Graph (consistency >= {CONSISTENCY_THRESHOLD * 100:.0f}%)")
        plt.axis("off")
        plt.tight_layout()

        consensus_png = "consensus_graph.png"
        plt.savefig(consensus_png, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"Consensus graph saved to {consensus_png}.")

    else:
        print(
            f"No edges reached the consistency threshold "
            f"{CONSISTENCY_THRESHOLD * 100:.0f}%."
        )

    results = []

    for (i, j), info in pair_info.items():
        results.append(
            {
                "Species_A": species_names[i],
                "Species_B": species_names[j],
                "Positive_ratio": info["positive_ratio"],
                "Negative_ratio": info["negative_ratio"],
                "Mean_positive_interaction": info["mean_positive"],
                "Mean_negative_interaction_strength": info["mean_negative"],
                "Consensus_direction": (
                    info["consensus_direction"]
                    if info["consensus_direction"]
                    else "no_consensus"
                ),
                "Consensus_strength": (
                    info["consensus_strength"]
                    if info["consensus_strength"]
                    else np.nan
                ),
            }
        )

    df_results = pd.DataFrame(results)

    consensus_csv = "shap_interaction_consensus.csv"
    df_results.to_csv(consensus_csv, index=False)

    print(f"\nConsensus statistics saved to {consensus_csv}.")
    print(df_results.to_string(index=False))

    asd_mask = y == 1
    nt_mask = y == 0

    asd_mean = interaction_vals[asd_mask].mean(axis=0)
    nt_mean = interaction_vals[nt_mask].mean(axis=0)
    diff_mean = asd_mean - nt_mean

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(
        asd_mean,
        xticklabels=species_names,
        yticklabels=species_names,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".4f",
    )
    plt.title("ASD Mean SHAP Interaction")

    plt.subplot(1, 3, 2)
    sns.heatmap(
        nt_mean,
        xticklabels=species_names,
        yticklabels=species_names,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".4f",
    )
    plt.title("NT Mean SHAP Interaction")

    plt.subplot(1, 3, 3)
    sns.heatmap(
        diff_mean,
        xticklabels=species_names,
        yticklabels=species_names,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".4f",
    )
    plt.title("Difference (ASD - NT)")

    plt.tight_layout()

    heatmap_png = "shap_interaction_heatmap.png"
    plt.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Interaction heatmap saved to {heatmap_png}.")

    if np.any(asd_mask):
        plot_group_network(
            interaction_vals[asd_mask],
            "ASD",
            species_names,
            threshold=0.0,
            show_edge_labels=True,
            edge_labels_fmt="{:.6f}",
        )
    else:
        print("No ASD samples found. Skipping ASD network plot.")

    if np.any(nt_mask):
        plot_group_network(
            interaction_vals[nt_mask],
            "NT",
            species_names,
            threshold=0.0,
            show_edge_labels=True,
            edge_labels_fmt="{:.6f}",
        )
    else:
        print("No NT samples found. Skipping NT network plot.")

    if np.any(asd_mask) and np.any(nt_mask):
        plot_difference_network(
            interaction_vals[asd_mask],
            interaction_vals[nt_mask],
            species_names,
            threshold=0.0,
            show_edge_labels=True,
            edge_labels_fmt="{:.6f}",
        )
    else:
        print("ASD or NT samples are missing. Skipping difference network plot.")

    print("Analysis completed.")


if __name__ == "__main__":
    main()