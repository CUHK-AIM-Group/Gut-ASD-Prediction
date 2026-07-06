import os
import re
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)


# ============================================================
# Data loading: same logic as original code
# ============================================================

def parse_group_mapping(mapping_str: str) -> Dict[str, int]:
    mapping = {}
    for item in mapping_str.split(","):
        if ":" not in item:
            continue
        k, v = item.split(":")
        mapping[str(k).strip()] = int(v.strip())
    return mapping


def map_labels(group_series: pd.Series, group_mapping: Dict[str, int]) -> np.ndarray:
    mapped = []
    lower_mapping = {str(k).strip().lower(): v for k, v in group_mapping.items()}

    for value in group_series:
        raw = str(value).strip()
        raw_lower = raw.lower()

        if raw in group_mapping:
            mapped.append(group_mapping[raw])
        elif raw_lower in lower_mapping:
            mapped.append(lower_mapping[raw_lower])
        else:
            try:
                num = float(raw)
                if num in [0.0, 1.0]:
                    mapped.append(int(num))
                else:
                    mapped.append(np.nan)
            except Exception:
                mapped.append(np.nan)

    return np.asarray(mapped, dtype=np.float32)


def find_dataset_pairs(datasets_dir: str) -> List[Tuple[str, str, str]]:
    files = os.listdir(datasets_dir)
    metadata_pattern = re.compile(r"^(.+?)metadata\.csv$", re.IGNORECASE)

    pairs = []
    for f in files:
        m = metadata_pattern.match(f)
        if m is None:
            continue

        prefix = m.group(1)
        metadata_path = os.path.join(datasets_dir, f)
        species_path = os.path.join(datasets_dir, f"{prefix}species.csv")

        if os.path.exists(species_path):
            pairs.append((prefix, species_path, metadata_path))
        else:
            print(f"WARNING: found {f}, but missing {prefix}species.csv, skipped.")

    return sorted(pairs, key=lambda x: x[0])


def load_species_list(species_list_path: str) -> List[str]:
    with open(species_list_path, "r", encoding="utf-8") as f:
        species = [line.strip() for line in f if line.strip()]
    species = list(dict.fromkeys(species))
    if len(species) == 0:
        raise ValueError("species_list file is empty.")
    return species


@dataclass
class CohortData:
    name: str
    X_raw: np.ndarray
    y: np.ndarray
    sample_ids: List[str]
    species_path: str
    metadata_path: str


def normalize_dataset_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", str(name)).lower()


def load_one_dataset(
    species_path: str,
    metadata_path: str,
    important_species: List[str],
    group_mapping: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    abund = pd.read_csv(species_path, index_col=0)
    abund.index = abund.index.astype(str)
    abund.columns = abund.columns.astype(str)

    abund = abund.T
    abund = abund.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    meta = pd.read_csv(metadata_path)
    if "clade_name" not in meta.columns:
        raise ValueError(f"{metadata_path} missing clade_name column.")
    if "group" not in meta.columns:
        raise ValueError(f"{metadata_path} missing group column.")

    meta["clade_name"] = meta["clade_name"].astype(str)
    meta = meta.set_index("clade_name")

    common_ids = abund.index.intersection(meta.index)
    if len(common_ids) == 0:
        raise ValueError(
            f"{os.path.basename(species_path)} and "
            f"{os.path.basename(metadata_path)} have no matched samples."
        )

    abund = abund.loc[common_ids]
    meta = meta.loc[common_ids]

    y = map_labels(meta["group"], group_mapping)
    valid_mask = ~np.isnan(y)

    abund = abund.iloc[valid_mask]
    y = y[valid_mask].astype(np.float32)

    X = np.zeros((len(abund), len(important_species)), dtype=np.float32)
    available_species = set(abund.columns)

    missing_count = 0
    for i, sp in enumerate(important_species):
        if sp in available_species:
            X[:, i] = abund[sp].values.astype(np.float32)
        else:
            missing_count += 1

    if missing_count > 0:
        print(
            f"  Note: {os.path.basename(species_path)} missing "
            f"{missing_count}/{len(important_species)} key species, filled with 0."
        )

    sample_ids = abund.index.astype(str).tolist()
    return X, y, sample_ids


def load_all_data(args) -> List[CohortData]:
    group_mapping = parse_group_mapping(args.group_mapping)
    important_species = load_species_list(args.species_list)
    dataset_pairs = find_dataset_pairs(args.datasets_dir)

    all_data = []
    for prefix, species_path, metadata_path in dataset_pairs:
        try:
            X_raw, y, sample_ids = load_one_dataset(
                species_path=species_path,
                metadata_path=metadata_path,
                important_species=important_species,
                group_mapping=group_mapping,
            )

            if len(np.unique(y)) < 2:
                print(f"WARNING: {prefix} has only one class, skipped.")
                continue

            all_data.append(
                CohortData(
                    name=prefix,
                    X_raw=X_raw,
                    y=y,
                    sample_ids=sample_ids,
                    species_path=species_path,
                    metadata_path=metadata_path,
                )
            )

            print(
                f"Loaded {prefix}: n={len(y)}, "
                f"TD={(y == 0).sum()}, ASD={(y == 1).sum()}"
            )

        except Exception as e:
            print(f"Failed to load {prefix}: {e}")

    if len(all_data) < 2:
        raise RuntimeError("Fewer than 2 valid datasets are available.")

    return all_data


# ============================================================
# Transformations: same abundance transform logic as original code
# ============================================================

def transform_abundance(
    X_raw: np.ndarray,
    method: str = "relative_log1p",
    eps: float = 1e-6,
) -> np.ndarray:

    X = np.asarray(X_raw, dtype=np.float32)
    X = np.clip(X, a_min=0.0, a_max=None)

    if method == "none":
        return X.astype(np.float32)

    if method == "log1p":
        return np.log1p(X).astype(np.float32)

    if method == "relative_log1p":
        row_sum = X.sum(axis=1, keepdims=True)
        X_rel = X / (row_sum + eps)
        return np.log1p(1e4 * X_rel).astype(np.float32)

    if method == "hellinger":
        row_sum = X.sum(axis=1, keepdims=True)
        X_rel = X / (row_sum + eps)
        return np.sqrt(X_rel).astype(np.float32)

    if method == "clr":
        row_sum = X.sum(axis=1, keepdims=True)
        X_rel = X / (row_sum + eps)
        log_x = np.log(X_rel + eps)
        clr = log_x - log_x.mean(axis=1, keepdims=True)
        return clr.astype(np.float32)

    if method == "arcsinh":
        return np.arcsinh(X / eps).astype(np.float32)

    raise ValueError(f"Unknown transform method: {method}")


# ============================================================
# PCA + GMM fitting logic
# ============================================================

def parse_gmm_components(s: str) -> List[int]:
    comps = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            comps.append(int(x))
    if len(comps) == 0:
        raise ValueError("--gmm_components cannot be empty.")
    return comps


def fit_pca_gmm_for_target(
    target_idx: int,
    all_data: List[CohortData],
    args,
):

    target = all_data[target_idx]
    sources = [all_data[i] for i in range(len(all_data)) if i != target_idx]

    if args.exclude_pretrain_from_sources:
        sources = [
            d for d in sources
            if normalize_dataset_name(d.name) != normalize_dataset_name(args.pretrain_dataset)
        ]

    if len(sources) == 0:
        raise RuntimeError(
            f"Target {target.name}: no source cohorts after excluding pretrain dataset."
        )

    source_names = [d.name for d in sources]

    print("\n" + "=" * 80)
    print(f"Target: {target.name}")
    print(f"Sources: {source_names}")
    print("=" * 80)

    # ------------------------------------------------------------
    # Same as original Step 1:
    # X_raw_list -> transform_abundance -> merge -> StandardScaler
    # -> PCA -> GMM with BIC selection
    # ------------------------------------------------------------
    X_raw_list = [d.X_raw for d in sources]
    X_trans_list = [
        transform_abundance(x, method=args.base_transform)
        for x in X_raw_list
    ]
    X_merged = np.vstack(X_trans_list).astype(np.float32)

    source_cohort_labels = []
    source_class_labels = []
    source_sample_ids = []

    for d in sources:
        source_cohort_labels.extend([d.name] * len(d.y))
        source_class_labels.extend(d.y.astype(int).tolist())
        source_sample_ids.extend(d.sample_ids)

    source_cohort_labels = np.asarray(source_cohort_labels, dtype=object)
    source_class_labels = np.asarray(source_class_labels, dtype=int)

    scaler_proto = StandardScaler()
    X_scaled = scaler_proto.fit_transform(X_merged)

    n_pca = min(
        int(args.pca_components),
        X_scaled.shape[1],
        X_scaled.shape[0] - 1,
    )

    if n_pca < 1:
        raise RuntimeError("PCA cannot be fitted because n_pca < 1.")

    pca = PCA(n_components=n_pca, random_state=args.seed)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)

    n_components_range = parse_gmm_components(args.gmm_components)

    best_gmm = None
    best_bic = np.inf
    best_n_comp = None

    for n_comp in n_components_range:
        if n_comp > X_pca.shape[0]:
            continue

        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                random_state=args.seed,
                max_iter=200,
                n_init=5,
                reg_covar=1e-6,
            )
            gmm.fit(X_pca)
            bic = gmm.bic(X_pca)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n_comp = n_comp

        except ValueError as e:
            print(f"  GMM with {n_comp} components failed: {e}. Skipping.")

    if best_gmm is None:
        fallback_n = min(2, X_pca.shape[0])
        print(
            f"  All GMM attempts failed; using fallback "
            f"n_components={fallback_n}, reg_covar=1e-3."
        )
        best_gmm = GaussianMixture(
            n_components=fallback_n,
            covariance_type="full",
            random_state=args.seed,
            reg_covar=1e-3,
        )
        best_gmm.fit(X_pca)
        best_n_comp = fallback_n
        best_bic = best_gmm.bic(X_pca)

    posteriors_source = best_gmm.predict_proba(X_pca)
    source_hard_component = np.argmax(posteriors_source, axis=1)

    # ------------------------------------------------------------
    # Target projection: same scaler + PCA + GMM posterior
    # ------------------------------------------------------------
    X_target_trans = transform_abundance(target.X_raw, method=args.base_transform)
    X_target_scaled = scaler_proto.transform(X_target_trans)
    X_target_pca = pca.transform(X_target_scaled)
    posteriors_target = best_gmm.predict_proba(X_target_pca)
    target_hard_component = np.argmax(posteriors_target, axis=1)

    print(f"  PCA components used: {n_pca}")
    print(f"  GMM components discovered: {best_n_comp}")
    print(f"  Best GMM BIC: {best_bic:.2f}")

    return {
        "target": target,
        "sources": sources,
        "source_names": source_names,
        "pca": pca,
        "gmm": best_gmm,
        "best_bic": best_bic,
        "X_source_pca": X_pca,
        "X_target_pca": X_target_pca,
        "posteriors_source": posteriors_source,
        "posteriors_target": posteriors_target,
        "source_hard_component": source_hard_component,
        "target_hard_component": target_hard_component,
        "source_cohort_labels": source_cohort_labels,
        "source_class_labels": source_class_labels,
        "source_sample_ids": source_sample_ids,
    }


# ============================================================
# Plotting utilities
# ============================================================

def ensure_2d_pca_coords(X_pca: np.ndarray) -> np.ndarray:
    X_pca = np.asarray(X_pca)
    if X_pca.shape[1] >= 2:
        return X_pca[:, :2]
    zeros = np.zeros((X_pca.shape[0], 1), dtype=X_pca.dtype)
    return np.hstack([X_pca[:, :1], zeros])


def get_2d_covariance(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov)

    if cov.ndim == 0:
        return np.eye(2) * float(cov)

    if cov.ndim == 1:
        if len(cov) == 1:
            return np.diag([cov[0], 1e-6])
        return np.diag(cov[:2])

    if cov.shape[0] >= 2 and cov.shape[1] >= 2:
        return cov[:2, :2]

    out = np.eye(2) * 1e-6
    out[0, 0] = cov[0, 0]
    return out


def draw_cov_ellipse(
    ax,
    mean_2d: np.ndarray,
    cov_2d: np.ndarray,
    edgecolor,
    label: str,
    n_std: float = 2.0,
    linewidth: float = 2.0,
):

    cov_2d = get_2d_covariance(cov_2d)

    vals, vecs = np.linalg.eigh(cov_2d)
    vals = np.clip(vals, 1e-12, None)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    width, height = 2.0 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        xy=mean_2d,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle="-",
        alpha=0.95,
        label=label,
    )

    ax.add_patch(ellipse)


def make_component_colors(M: int):
    cmap = plt.get_cmap("tab10" if M <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(M)]


def save_pca_coordinates_csv(result: dict, out_csv: str):
    target = result["target"]

    Xs = ensure_2d_pca_coords(result["X_source_pca"])
    Xt = ensure_2d_pca_coords(result["X_target_pca"])

    rows = []

    for i in range(Xs.shape[0]):
        row = {
            "set": "source",
            "dataset": result["source_cohort_labels"][i],
            "sample_id": result["source_sample_ids"][i],
            "label": int(result["source_class_labels"][i]),
            "PC1": float(Xs[i, 0]),
            "PC2": float(Xs[i, 1]),
            "hard_gmm_component": int(result["source_hard_component"][i]),
        }

        for m in range(result["posteriors_source"].shape[1]):
            row[f"posterior_proto_{m}"] = float(result["posteriors_source"][i, m])

        rows.append(row)

    for i in range(Xt.shape[0]):
        row = {
            "set": "target",
            "dataset": target.name,
            "sample_id": target.sample_ids[i],
            "label": int(target.y[i]),
            "PC1": float(Xt[i, 0]),
            "PC2": float(Xt[i, 1]),
            "hard_gmm_component": int(result["target_hard_component"][i]),
        }

        for m in range(result["posteriors_target"].shape[1]):
            row[f"posterior_proto_{m}"] = float(result["posteriors_target"][i, m])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)


def plot_pca_gmm_schematic(result: dict, args, out_path: str):
    target = result["target"]
    pca = result["pca"]
    gmm = result["gmm"]

    Xs = ensure_2d_pca_coords(result["X_source_pca"])
    Xt = ensure_2d_pca_coords(result["X_target_pca"])

    source_comp = result["source_hard_component"]
    target_comp = result["target_hard_component"]

    M = gmm.n_components
    colors = make_component_colors(M)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.8),
        gridspec_kw={"width_ratios": [2.1, 1.0]},
    )

    ax = axes[0]

    # -----------------------------
    # Source points by GMM component
    # -----------------------------
    for m in range(M):
        mask = source_comp == m
        if mask.sum() == 0:
            continue

        ax.scatter(
            Xs[mask, 0],
            Xs[mask, 1],
            s=args.source_point_size,
            color=colors[m],
            alpha=args.source_alpha,
            linewidths=0,
            label=f"Source assigned to prototype {m}",
        )

    # -----------------------------
    # Target points, projected but not used to fit PCA/GMM
    # -----------------------------
    for m in range(M):
        mask = target_comp == m
        if mask.sum() == 0:
            continue

        ax.scatter(
            Xt[mask, 0],
            Xt[mask, 1],
            s=args.target_point_size,
            marker="^",
            color=colors[m],
            edgecolor="black",
            linewidths=0.2,
            alpha=args.target_alpha,
            label=f"Target routed to prototype {m}",
        )

    # -----------------------------
    # GMM centers and covariance ellipses
    # -----------------------------
    means_2d = ensure_2d_pca_coords(gmm.means_)

    for m in range(M):
        cov_2d = get_2d_covariance(gmm.covariances_[m])

        draw_cov_ellipse(
            ax=ax,
            mean_2d=means_2d[m],
            cov_2d=cov_2d,
            edgecolor=colors[m],
            label=f"Prototype {m} covariance",
            n_std=args.ellipse_n_std,
            linewidth=2.0,
        )

        ax.scatter(
            means_2d[m, 0],
            means_2d[m, 1],
            s=180,
            marker="X",
            color=colors[m],
            edgecolor="black",
            linewidths=1.0,
            zorder=10,
        )

        ax.text(
            means_2d[m, 0],
            means_2d[m, 1],
            f"  P{m}",
            fontsize=11,
            weight="bold",
            va="center",
        )

    evr = pca.explained_variance_ratio_
    pc1_var = evr[0] * 100.0 if len(evr) >= 1 else 0.0
    pc2_var = evr[1] * 100.0 if len(evr) >= 2 else 0.0

    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    ax.set_title(
        f"PCA + GMM prototype discovery\n"
        f"Target = {target.name}; GMM fitted on source only"
    )

    ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
    ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    # Keep legend compact
    legend_handles = [

    # Source samples (empty circle)
        Line2D(
            [0],
            [0],
            marker='o',
            linestyle='None',
            markersize=8,
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=1.5,
            label='Source samples'
        ),

        # Target samples (empty triangle)
        Line2D(
            [0],
            [0],
            marker='^',
            linestyle='None',
            markersize=9,
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=1.5,
            label='Target samples'
        )
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=10,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1,
        handlelength=2.0,
        borderpad=0.6,
        labelspacing=0.6
    )

    # -----------------------------
    # Right panel: mean posterior routing weight for target
    # -----------------------------
    ax2 = axes[1]

    mean_target_post = result["posteriors_target"].mean(axis=0)
    std_target_post = result["posteriors_target"].std(axis=0)

    x = np.arange(M)

    ax2.bar(
        x,
        mean_target_post,
        yerr=std_target_post,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    ax2.set_ylim(0, max(1.0, float(mean_target_post.max() + std_target_post.max()) * 1.2))
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{m}" for m in range(M)])
    ax2.set_ylabel("Mean target posterior")
    ax2.set_title("Target routing weights\nmean ± std")

    for m in range(M):
        ax2.text(
            x[m],
            mean_target_post[m] + 0.02,
            f"{mean_target_post[m]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle(
        "Prototype-based Soft MoE: PCA + GMM schematic",
        fontsize=15,
        weight="bold",
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Draw PCA+GMM schematic figures for prototype-based Soft MoE."
    )

    # Data
    parser.add_argument("--datasets_dir", type=str, default="XXX", help="Directory containing datasets with species and metadata CSVs.")
    parser.add_argument("--species_list", type=str, default="XXX", help="Path to the text file listing important species.")
    parser.add_argument("--output_dir", type=str, default="./pca_gmm_schematic")
    parser.add_argument("--group_mapping", type=str, default="XXX", help="Mapping of group labels to integers, e.g., 'TD:0,ASD:1'.")

    # Same prototype-discovery options as original code
    parser.add_argument(
        "--base_transform",
        type=str,
        default="arcsinh",
        choices=["none", "log1p", "relative_log1p", "hellinger", "clr", "arcsinh"],
    )
    parser.add_argument("--pca_components", type=int, default=20)
    parser.add_argument("--gmm_components", type=str, default="4")
    parser.add_argument("--seed", type=int, default=421)

    # Same source-exclusion option as original code
    parser.add_argument("--pretrain_dataset", type=str, default="Inhouse")
    parser.add_argument(
        "--exclude_pretrain_from_sources",
        action="store_true",
        default=True,
        help="Match original setting: remove pretrain dataset from supervised source pool.",
    )
    parser.add_argument(
        "--include_pretrain_as_source",
        action="store_false",
        dest="exclude_pretrain_from_sources",
        help="Include pretrain dataset as source when drawing LODO figures.",
    )

    # Plot options
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        help="Target dataset name. Use 'all' to draw all LODO target figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ellipse_n_std", type=float, default=2.0)
    parser.add_argument("--source_point_size", type=float, default=32)
    parser.add_argument("--target_point_size", type=float, default=58)
    parser.add_argument("--source_alpha", type=float, default=0.65)
    parser.add_argument("--target_alpha", type=float, default=0.95)
    parser.add_argument(
        "--save_coordinates",
        action="store_true",
        default=True,
        help="Save PCA coordinates and GMM posterior weights as CSV.",
    )
    parser.add_argument(
        "--no_save_coordinates",
        action="store_false",
        dest="save_coordinates",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    all_data = load_all_data(args)

    if args.target.lower() == "all":
        target_indices = list(range(len(all_data)))
    else:
        target_key = normalize_dataset_name(args.target)
        target_indices = [
            i for i, d in enumerate(all_data)
            if normalize_dataset_name(d.name) == target_key
        ]

        if len(target_indices) == 0:
            available = [d.name for d in all_data]
            raise ValueError(
                f"Target {args.target} not found. Available datasets: {available}"
            )

    for target_idx in target_indices:
        result = fit_pca_gmm_for_target(
            target_idx=target_idx,
            all_data=all_data,
            args=args,
        )

        target_name = result["target"].name
        safe_target_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", target_name)

        fig_path = os.path.join(
            args.output_dir,
            f"pca_gmm_schematic_target_{safe_target_name}.pdf",
        )

        plot_pca_gmm_schematic(
            result=result,
            args=args,
            out_path=fig_path,
        )

        print(f"  Figure saved to: {fig_path}")

        if args.save_coordinates:
            csv_path = os.path.join(
                args.output_dir,
                f"pca_gmm_coordinates_target_{safe_target_name}.csv",
            )
            save_pca_coordinates_csv(result, csv_path)
            print(f"  Coordinates saved to: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()