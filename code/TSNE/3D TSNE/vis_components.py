import os
import sys
import argparse
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.append("../")

from networks.species.Disentangle_knowledge import ImprovedCompositeModelWithAttention
from networks.species.dataloader_knowledge import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and visualize base, metadata-shift, and disease-shift components."
    )
    parser.add_argument("--model_path", type=str, default="XXX")
    parser.add_argument("--abundance_csv", type=str, default="XXX")
    parser.add_argument("--meta_csv", type=str, default="XXX")
    parser.add_argument("--metadata_txt", type=str, default="XXX")
    parser.add_argument("--species_prior_csv", type=str, default="XXX")
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="XXX")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--delta_noise_std", type=float, default=0.0)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--gif_frames", type=int, default=36)
    parser.add_argument("--gif_duration", type=float, default=0.1)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def zscore_features(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def save_component_csv(
    data: np.ndarray,
    name: str,
    output_dir: str,
    selected_species,
    sample_ids: np.ndarray,
    disease_labels: np.ndarray,
) -> None:
    df = pd.DataFrame(data, columns=selected_species)
    df.insert(0, "sample_id", sample_ids)
    df.insert(1, "disease", disease_labels)
    out_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {data.shape}.")


def build_model_from_checkpoint(checkpoint, device: torch.device):
    model_args = checkpoint.get("args", {})
    info_ckpt = checkpoint["info"]

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

    hidden_dim = model_args.get("hid", 1024)
    model = ImprovedCompositeModelWithAttention(
        x_dim=info_ckpt["X_dim"],
        c_dim=info_ckpt["C_dim"],
        disease_dim=disease_dim,
        others_dim=info_ckpt.get("others_dim", info_ckpt["C_dim"] - disease_dim),
        z_dim=model_args.get("z_dim", 512),
        u_dim=model_args.get("u_dim", 32),
        enc_hidden_dims=[hidden_dim, hidden_dim // 2],
        dec_hidden_dims=[hidden_dim // 2, hidden_dim],
        dropout=model_args.get("dropout", 0.3),
        shift_dropout=model_args.get("shift_dropout", 0.4),
        grl_lambda=model_args.get("grl_lambda", 0.1),
        adversary=None,
        prior_info=prior_info,
        use_attention=model_args.get("use_prior_attention", True),
        attention_heads=model_args.get("attention_heads", 5),
        attn_beta=model_args.get("attn_beta", 0.7),
        use_unknown=model_args.get("use_unknown", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, info_ckpt


def extract_components(model, dataset, batch_size, device, disease_start, disease_dim):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    base_list, delta_o_list, delta_d_list = [], [], []
    sample_ids, disease_labels = [], []

    print("Extracting model components...")
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x_clr"].to(device)
            c = batch["c"].to(device)
            out = model(x, c, disease_start=disease_start, disease_dim=disease_dim, training_stage="full")

            base_list.append(out["x_base"].cpu())
            delta_o_list.append(out["delta_o"].cpu())
            delta_d_list.append(out["delta_d"].cpu())
            sample_ids.extend(batch["idx"].numpy())

            c_np = c.cpu().numpy()
            disease_onehot = c_np[:, disease_start:disease_start + disease_dim]
            disease_labels.extend(np.argmax(disease_onehot, axis=1))

    base = torch.cat(base_list, dim=0).numpy()
    delta_o = torch.cat(delta_o_list, dim=0).numpy()
    delta_d = torch.cat(delta_d_list, dim=0).numpy()

    return (
        zscore_features(base),
        zscore_features(delta_o),
        zscore_features(delta_d),
        np.asarray(sample_ids),
        np.asarray(disease_labels),
    )


def sample_components(base, delta_o, delta_d, sample_num):
    n_total = len(base)
    if sample_num > 0 and sample_num < n_total:
        rng = np.random.default_rng(42)
        idx_base = rng.choice(n_total, sample_num, replace=False)
        idx_delta_o = rng.choice(n_total, sample_num, replace=False)
        idx_delta_d = rng.choice(n_total, sample_num, replace=False)
        print(f"Using {sample_num} samples per component.")
        return base[idx_base], delta_o[idx_delta_o], delta_d[idx_delta_d]

    return base, delta_o, delta_d


def run_tsne(base, delta_o, delta_d, perplexity):
    combined = np.vstack([base, delta_o, delta_d])
    combined_scaled = StandardScaler().fit_transform(combined)
    max_perplexity = max(5.0, min(perplexity, (combined_scaled.shape[0] - 1) / 3))
    tsne = TSNE(n_components=3, perplexity=max_perplexity, random_state=42)
    combined_3d = tsne.fit_transform(combined_scaled)

    n_sample = len(base)
    return (
        combined_3d[:n_sample],
        combined_3d[n_sample:2 * n_sample],
        combined_3d[2 * n_sample:],
    )


def plot_components_3d(base_3d, delta_o_3d, delta_d_3d, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(0.5)
        axis.pane.set_facecolor("white")

    ax.scatter(
        base_3d[:, 0], base_3d[:, 1], base_3d[:, 2],
        c="#D8EEF8", alpha=1.0, s=50, edgecolors="black", linewidths=0.15, label="Base"
    )
    ax.scatter(
        delta_o_3d[:, 0], delta_o_3d[:, 1], delta_o_3d[:, 2],
        c="#84ADDC", alpha=1.0, s=50, edgecolors="black", linewidths=0.15, label="Metadata shift"
    )
    ax.scatter(
        delta_d_3d[:, 0], delta_d_3d[:, 1], delta_d_3d[:, 2],
        c="#377EB8", alpha=1.0, s=50, edgecolors="black", linewidths=0.15, label="Disease shift"
    )

    all_x = np.concatenate([base_3d[:, 0], delta_o_3d[:, 0], delta_d_3d[:, 0]])
    all_y = np.concatenate([base_3d[:, 1], delta_o_3d[:, 1], delta_d_3d[:, 1]])
    all_z = np.concatenate([base_3d[:, 2], delta_o_3d[:, 2], delta_d_3d[:, 2]])

    ax.set_xlim(all_x.min(), all_x.max())
    ax.set_ylim(all_y.min(), all_y.max())
    ax.set_zlim(all_z.min(), all_z.max())
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_zlabel("t-SNE3")
    ax.grid(True, linestyle="--", linewidth=0.01, alpha=0.1)
    ax.legend(frameon=False)

    out_img = os.path.join(output_dir, "components_separation_3d.svg")
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    print(f"Saved 3D component plot to {out_img}.")
    return fig, ax


def create_rotating_gif(fig, ax, output_path, n_frames=36, duration=0.1):
    temp_dir = Path(output_path).parent / "temp_frames"
    temp_dir.mkdir(exist_ok=True)

    init_elev = ax.elev
    init_azim = ax.azim
    filenames = []

    for i in range(n_frames):
        azim = init_azim + i * (360 / n_frames)
        ax.view_init(elev=init_elev, azim=azim)
        frame_path = temp_dir / f"frame_{i:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches="tight")
        filenames.append(frame_path)

    with imageio.get_writer(output_path, mode="I", duration=duration, loop=0) as writer:
        for frame_path in filenames:
            writer.append_data(imageio.imread(frame_path))

    shutil.rmtree(temp_dir)
    print(f"Saved rotating GIF to {output_path}.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)

    _, val_loader, _ = build_dataloaders(
        abundance_csv=args.abundance_csv,
        meta_csv=args.meta_csv,
        metadata_txt=args.metadata_txt,
        species_prior_csv=args.species_prior_csv,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        val_split=0.2,
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

    checkpoint = torch.load(args.model_path, map_location=device)
    model, info_ckpt = build_model_from_checkpoint(checkpoint, device)

    disease_start = info_ckpt.get("disease_start", 0)
    disease_dim = info_ckpt.get("disease_dim", 2)
    selected_species = info_ckpt.get(
        "selected_species",
        [f"f{i}" for i in range(info_ckpt["X_dim"])],
    )

    base, delta_o, delta_d, sample_ids, disease_labels = extract_components(
        model=model,
        dataset=val_loader.dataset,
        batch_size=args.batch_size,
        device=device,
        disease_start=disease_start,
        disease_dim=disease_dim,
    )

    save_component_csv(base, "base", args.output_dir, selected_species, sample_ids, disease_labels)
    save_component_csv(delta_o, "delta_o", args.output_dir, selected_species, sample_ids, disease_labels)
    save_component_csv(delta_d, "delta_d", args.output_dir, selected_species, sample_ids, disease_labels)

    base_sampled, delta_o_sampled, delta_d_sampled = sample_components(
        base, delta_o, delta_d, args.sample_num
    )

    if args.delta_noise_std > 0:
        rng = np.random.default_rng(42)
        delta_d_sampled = delta_d_sampled + rng.normal(
            0, args.delta_noise_std, size=delta_d_sampled.shape
        )
        print(f"Added Gaussian noise to disease-shift points with std={args.delta_noise_std}.")

    base_3d, delta_o_3d, delta_d_3d = run_tsne(
        base_sampled,
        delta_o_sampled,
        delta_d_sampled,
        args.perplexity,
    )

    fig, ax = plot_components_3d(base_3d, delta_o_3d, delta_d_3d, args.output_dir)
    gif_path = os.path.join(args.output_dir, "components_rotation.gif")
    create_rotating_gif(fig, ax, gif_path, n_frames=args.gif_frames, duration=args.gif_duration)
    plt.close(fig)


if __name__ == "__main__":
    main()
