#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Note: The number of prototypes depends on the distribution of the source datasets and should be selected carefully."""

import os
import re
import json
import argparse
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

from scipy.stats import rankdata, norm
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics
# ============================================================

def safe_auc(y, score):
    y = np.asarray(y).astype(int).reshape(-1)
    score = np.asarray(score).reshape(-1)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return 0.5


def safe_prauc(y, score):
    y = np.asarray(y).astype(int).reshape(-1)
    score = np.asarray(score).reshape(-1)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return 0.5


def calculate_metrics(probs, targets, threshold: float = 0.5):
    probs = np.asarray(probs).reshape(-1)
    targets = np.asarray(targets).astype(int).reshape(-1)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)

    return {
        "auc": safe_auc(targets, probs),
        "prauc": safe_prauc(targets, probs),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


# ============================================================
# Data loading
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
            print(f"WARNING: found {f}, but {prefix}species.csv is missing. Skipping.")

    return sorted(pairs, key=lambda x: x[0])


def load_species_list(species_list_path: str) -> List[str]:
    with open(species_list_path, "r", encoding="utf-8") as f:
        species = [line.strip() for line in f if line.strip()]
    species = list(dict.fromkeys(species))
    if len(species) == 0:
        raise ValueError("The species list file is empty.")
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


def find_dataset_by_name(all_data: List[CohortData], name: str) -> Optional[CohortData]:
    target_key = normalize_dataset_name(name)
    for d in all_data:
        if normalize_dataset_name(d.name) == target_key:
            return d
    return None


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
        raise ValueError(f"{metadata_path} is missing the clade_name column.")
    if "group" not in meta.columns:
        raise ValueError(f"{metadata_path} is missing the group column.")

    meta["clade_name"] = meta["clade_name"].astype(str)
    meta = meta.set_index("clade_name")

    common_ids = abund.index.intersection(meta.index)
    if len(common_ids) == 0:
        raise ValueError(
            f"{os.path.basename(species_path)} and {os.path.basename(metadata_path)} have no matched samples."
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
        print(f"  NOTE: {os.path.basename(species_path)} is missing {missing_count}/{len(important_species)} selected species; missing values were filled with 0.")

    sample_ids = abund.index.astype(str).tolist()
    return X, y, sample_ids


# ============================================================
# Transformations
# ============================================================

def transform_abundance(X_raw: np.ndarray, method: str = "relative_log1p", eps: float = 1e-6) -> np.ndarray:
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


def rank_gaussian_transform(X: np.ndarray, clip: float = 3.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    Z = np.zeros_like(X, dtype=np.float64)

    if n <= 1:
        return Z.astype(np.float32)

    for j in range(p):
        ranks = rankdata(X[:, j], method="average")
        u = (ranks - 0.5) / n
        u = np.clip(u, 1e-4, 1.0 - 1e-4)
        z = norm.ppf(u)
        z = np.clip(z, -clip, clip)
        Z[:, j] = z

    return Z.astype(np.float32)


def preprocess_blocks(
    X_raw_list: List[np.ndarray],
    mode: str,
    base_transform: str = "relative_log1p",
    rank_clip: float = 3.0,
    query_raw: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if mode == "cohort_rank_gauss":
        X_train = np.vstack([rank_gaussian_transform(x, clip=rank_clip) for x in X_raw_list]).astype(np.float32)
        X_query = None
        if query_raw is not None:
            X_query = rank_gaussian_transform(query_raw, clip=rank_clip).astype(np.float32)
        return X_train, X_query

    X_base_list = [transform_abundance(x, method=base_transform) for x in X_raw_list]
    X_train_base = np.vstack(X_base_list).astype(np.float32)

    if mode == "pooled_robust_zscore":
        scaler = RobustScaler()
    elif mode == "pooled_standard_zscore":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    X_train = scaler.fit_transform(X_train_base).astype(np.float32)

    X_query = None
    if query_raw is not None:
        Xq = transform_abundance(query_raw, method=base_transform)
        X_query = scaler.transform(Xq).astype(np.float32)

    return X_train, X_query


# ============================================================
# SupCon Loss
# ============================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        sim = features @ features.T / self.temperature

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()

        N = features.shape[0]
        eye = torch.eye(N, device=features.device, dtype=torch.bool)
        pos_mask = pos_mask.masked_fill(eye, 0.0)

        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return features.sum() * 0.0

        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim).masked_fill(eye, 0.0)

        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sum[valid] / (all_sum[valid] + 1e-8) + 1e-8)
        return loss.mean()


# ============================================================
# Models and training
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.sample_weights = torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)) if sample_weights is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.sample_weights is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx]
        return self.X[idx], self.y[idx]


def parse_dims(s: str) -> List[int]:
    if s is None or str(s).strip() == "":
        return [32]
    dims = [int(x) for x in str(s).split(",") if str(x).strip()]
    return dims if len(dims) > 0 else [32]


class SmallMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        use_layernorm: bool = True,
        use_bn: bool = False,
    ):
        super().__init__()
        self.use_bn = use_bn
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            elif use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = h
        self.feature_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x, return_feature=False):
        feat = self.feature_layers(x)
        logits = self.classifier(feat).squeeze(1)
        probs = torch.sigmoid(logits)
        if return_feature:
            return logits, probs, feat
        return logits, probs


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int).reshape(-1)
    weights = np.zeros(len(y), dtype=np.float64)
    for c in [0, 1]:
        mask = y == c
        n = int(mask.sum())
        if n > 0:
            weights[mask] = 1.0 / n
    weights = weights / (weights.mean() + 1e-12)
    return weights.astype(np.float64)


def clone_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_pretrained_mlp_weights(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    mode: str = "encoder",
) -> None:
    """Load compatible pretrained encoder weights into the MLP classifier."""
    if state_dict is None:
        return

    current = model.state_dict()
    filtered = {}

    for k, v in state_dict.items():
        if k not in current:
            continue
        if tuple(v.shape) != tuple(current[k].shape):
            continue

        if mode == "encoder":
            if k.startswith("feature_layers."):
                filtered[k] = v
        elif mode == "first_layer":
            # With the current MLP definition, feature_layers.0 is the first Linear layer.
            if k.startswith("feature_layers.0."):
                filtered[k] = v
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown ae_transfer_mode: {mode}")

    current.update(filtered)
    model.load_state_dict(current)


class MaskedAEDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class MaskedDenoisingAutoencoder(nn.Module):
    """Masked denoising autoencoder with a classifier-compatible encoder."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        use_layernorm: bool = True,
        use_bn: bool = False,
    ):
        super().__init__()
        self.use_bn = use_bn

        enc_layers = []
        dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(dim, h))
            if use_bn:
                enc_layers.append(nn.BatchNorm1d(h))
            elif use_layernorm:
                enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(nn.GELU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            dim = h
        self.feature_layers = nn.Sequential(*enc_layers)

        dec_layers = []
        dec_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        for j, h in enumerate(dec_dims):
            dec_layers.append(nn.Linear(dim, h))
            is_last = (j == len(dec_dims) - 1)
            if not is_last:
                if use_bn:
                    dec_layers.append(nn.BatchNorm1d(h))
                elif use_layernorm:
                    dec_layers.append(nn.LayerNorm(h))
                dec_layers.append(nn.GELU())
                if dropout > 0:
                    dec_layers.append(nn.Dropout(dropout))
            dim = h
        self.decoder = nn.Sequential(*dec_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        z = self.feature_layers(x)
        recon = self.decoder(z)
        return recon, z


def apply_random_mask(
    x: torch.Tensor,
    mask_ratio: float,
    mask_value: float = 0.0,
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask feature entries and return the corrupted tensor plus mask."""
    if mask_ratio <= 0:
        mask = torch.zeros_like(x, dtype=torch.bool)
        x_corrupt = x
    else:
        mask = torch.rand_like(x) < float(mask_ratio)
        if x.shape[1] > 0:
            no_mask = mask.sum(dim=1) == 0
            if no_mask.any():
                rand_idx = torch.randint(0, x.shape[1], (int(no_mask.sum().item()),), device=x.device)
                rows = torch.where(no_mask)[0]
                mask[rows, rand_idx] = True
        x_corrupt = x.masked_fill(mask, float(mask_value))

    if noise_std > 0:
        x_corrupt = x_corrupt + torch.randn_like(x_corrupt) * float(noise_std)
    return x_corrupt, mask


def fit_inhouse_masked_ae_pretrain(
    all_data: List[CohortData],
    args,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[str]]:
    """Pretrain a masked denoising autoencoder on the selected In-house dataset."""
    if (not args.use_inhouse_ae_pretrain) or args.model_type != "mlp":
        return None, None

    pretrain_data = find_dataset_by_name(all_data, args.pretrain_dataset)
    if pretrain_data is None:
        print(f"WARNING: pretrain_dataset={args.pretrain_dataset} not found; disable In-house AE pretraining.")
        return None, None

    print("\n" + "=" * 90)
    print(
        f"Masked denoising AE pretraining on {pretrain_data.name}: "
        f"n={len(pretrain_data.y)}, TD={(pretrain_data.y == 0).sum()}, ASD={(pretrain_data.y == 1).sum()}"
    )
    print("Labels are ignored during AE pretraining.")
    print("=" * 90)

    X_pretrain, _ = preprocess_blocks(
        [pretrain_data.X_raw],
        mode=args.preprocess,
        base_transform=args.base_transform,
        rank_clip=args.rank_clip,
        query_raw=None,
    )

    set_seed(args.seed + 7777)
    device = torch.device(args.device)

    dataset = MaskedAEDataset(X_pretrain)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = MaskedDenoisingAutoencoder(
        input_dim=X_pretrain.shape[1],
        hidden_dims=parse_dims(args.hidden_dims),
        dropout=args.ae_dropout if args.ae_dropout >= 0 else args.dropout,
        use_layernorm=args.use_layernorm,
        use_bn=args.use_bn,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(args.ae_pretrain_lr),
        weight_decay=float(args.ae_weight_decay),
    )

    best_state = None
    best_loss = float("inf")
    model.train()
    for epoch in range(1, int(args.ae_pretrain_epochs) + 1):
        losses = []
        for xb in loader:
            xb = xb.to(device)
            x_corrupt, mask = apply_random_mask(
                xb,
                mask_ratio=args.ae_mask_ratio,
                mask_value=args.ae_mask_value,
                noise_std=args.ae_noise_std,
            )
            recon, _ = model(x_corrupt)

            if args.ae_loss_on_masked_only and mask.any():
                loss = F.mse_loss(recon[mask], xb[mask])
            else:
                loss = F.mse_loss(recon, xb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        epoch_loss = float(np.mean(losses)) if len(losses) > 0 else float("inf")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = clone_state_dict_cpu(model)

        if args.ae_verbose and (epoch == 1 or epoch % args.ae_print_every == 0 or epoch == int(args.ae_pretrain_epochs)):
            print(f"  AE epoch {epoch:03d}/{int(args.ae_pretrain_epochs)} | recon_loss={epoch_loss:.6f}")

    if best_state is None:
        print("WARNING: In-house AE pretraining failed; disable pretrained initialization.")
        return None, None

    print(
        f"Finished In-house masked AE pretraining. "
        f"best_recon_loss={best_loss:.6f}, transfer_mode={args.ae_transfer_mode}."
    )
    return best_state, pretrain_data.name


def train_mlp_classifier(
    X: np.ndarray,
    y: np.ndarray,
    args,
    seed: int = 42,
    supcon_weight: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    init_mode: str = "encoder",
) -> Optional[nn.Module]:
    y_int = np.asarray(y).astype(int).reshape(-1)
    if len(np.unique(y_int)) < 2:
        return None

    set_seed(seed)
    device = torch.device(args.device)

    # ----------------------------
    # Build sampler
    # ----------------------------
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        sample_weights = np.clip(sample_weights, 0.0, None)
        sample_weights = sample_weights / (sample_weights.mean() + 1e-12)

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        if args.balanced_sampling:
            bal_weights = make_sample_weights(y_int)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(bal_weights, dtype=torch.double),
                num_samples=len(bal_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

    dataset = TabularDataset(X, y_int, sample_weights)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = SmallMLPClassifier(
        input_dim=X.shape[1],
        hidden_dims=parse_dims(args.hidden_dims),
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
        use_bn=args.use_bn,
    ).to(device)

    if init_state_dict is not None:
        load_pretrained_mlp_weights(model, init_state_dict, mode=init_mode)

    # Keep per-sample classification losses for optional sample weighting.
    criterion_cls = nn.BCEWithLogitsLoss(reduction="none")
    criterion_supcon = SupConLoss(temperature=0.1) if supcon_weight > 0 else None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    swa_model = None
    if args.use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start_epoch = int(args.epochs * args.swa_start_ratio)

    model.train()

    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            if len(batch) == 3:
                xb, yb, wb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)

                # Normalize weights for stable optimization.
                wb = torch.clamp(wb, min=0.0)
                wb = wb / (wb.mean() + 1e-8)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                wb = None

            logits, probs, feat = model(xb, return_feature=True)

            # Per-sample BCE loss.
            loss_vec = criterion_cls(logits, yb)

            # Apply sample weights to the BCE loss.
            if wb is not None:
                loss_cls = (loss_vec * wb).sum() / (wb.sum() + 1e-8)
            else:
                loss_cls = loss_vec.mean()

            if supcon_weight > 0 and criterion_supcon is not None:
                loss_sup = criterion_supcon(feat, yb)
                loss = loss_cls + supcon_weight * loss_sup
            else:
                loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )
            optimizer.step()

        if args.use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

    if args.use_swa and swa_model is not None:
        swa_model.module.eval()
        return swa_model.module
    else:
        model.eval()
        return model


@torch.no_grad()
def predict_mlp(model: nn.Module, X: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray]:
    if model is None:
        probs = np.full(X.shape[0], 0.5, dtype=np.float32)
        logits = np.zeros(X.shape[0], dtype=np.float32)
        return logits, probs

    device = torch.device(args.device)
    model.eval()
    dataset = TabularDataset(X, np.zeros(X.shape[0], dtype=np.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logits_list = []
    probs_list = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits, probs = model(xb)
        logits_list.append(logits.detach().cpu().numpy().reshape(-1))
        probs_list.append(probs.detach().cpu().numpy().reshape(-1))

    logits = np.concatenate(logits_list).astype(np.float32)
    probs = np.concatenate(probs_list).astype(np.float32)
    return logits, probs


def adapt_bn(model: nn.Module, X: np.ndarray, batch_size: int = 32, device: str = 'cpu') -> nn.Module:
    """Update BatchNorm running statistics with unlabeled target data."""
    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    if not has_bn:
        return model

    model.train()
    device = torch.device(device)
    dataset = TabularDataset(X, np.zeros(len(X)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            model(xb)

    model.eval()
    return model


def fit_classifier(X: np.ndarray, y: np.ndarray, args, seed: int = 42,
                   supcon_weight: float = 0.0, sample_weights: Optional[np.ndarray] = None,
                   init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
                   init_mode: str = "encoder"):
    if args.model_type == "logistic":
        return {"type": "logistic", "model": fit_logistic(X, y, C=args.C, seed=seed)}
    if args.model_type == "mlp":
        model = train_mlp_classifier(
            X, y, args, seed, supcon_weight, sample_weights,
            init_state_dict=init_state_dict,
            init_mode=init_mode,
        )
        return {"type": "mlp", "model": model}
    raise ValueError(f"Unknown model_type: {args.model_type}")


def predict_classifier(model_obj, X: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray]:
    if model_obj is None:
        probs = np.full(X.shape[0], 0.5, dtype=np.float32)
        logits = np.zeros(X.shape[0], dtype=np.float32)
        return logits, probs

    model_type = model_obj.get("type", "logistic")
    model = model_obj.get("model", None)

    if model_type == "logistic":
        return predict_logistic(model, X)
    if model_type == "mlp":
        return predict_mlp(model, X, args)

    raise ValueError(f"Unknown model_type: {model_type}")


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 42):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(
        penalty="l2", C=float(C), solver="liblinear",
        class_weight="balanced", max_iter=2000, random_state=seed,
    )
    model.fit(X, y)
    return model


def predict_logistic(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if model is None:
        probs = np.full(X.shape[0], 0.5, dtype=np.float32)
        logits = np.zeros(X.shape[0], dtype=np.float32)
        return logits, probs
    probs = model.predict_proba(X)[:, 1].astype(np.float32)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = logit(probs).astype(np.float32)
    return logits, probs


# ============================================================
# Fusion utilities
# ============================================================

def weighted_logit_fusion(logits_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits_matrix = np.asarray(logits_matrix, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    weights = weights / (weights.sum() + 1e-12)
    final_logits = logits_matrix @ weights
    final_probs = expit(final_logits).astype(np.float32)
    return final_probs


def simplex_weights(M: int, step: float = 0.05, n_random: int = 100) -> List[np.ndarray]:
    if M <= 1:
        return [np.ones(1, dtype=np.float32)]

    weights = []
    # Equal weights.
    weights.append(np.ones(M, dtype=np.float32) / M)

    # One-hot weights.
    for k in range(M):
        w = np.zeros(M, dtype=np.float32)
        w[k] = 1.0
        weights.append(w)

    # Full grid search for small expert counts.
    if M == 2:
        vals = np.arange(0.0, 1.0 + 1e-9, step)
        for a in vals:
            weights.append(np.asarray([a, 1.0 - a], dtype=np.float32))
    elif M == 3:
        vals = np.arange(0.0, 1.0 + 1e-9, step)
        for a in vals:
            for b in vals:
                c = 1.0 - a - b
                if c >= -1e-9:
                    weights.append(np.asarray([a, b, max(0.0, c)], dtype=np.float32))
    else:
        # Random Dirichlet weights for larger expert counts.
        np.random.seed(42)
        for _ in range(n_random):
            w = np.random.dirichlet(np.ones(M), size=1).flatten().astype(np.float32)
            weights.append(w)

    # Remove near-duplicate candidates.
    unique_weights = []
    for w in weights:
        is_dup = False
        for u in unique_weights:
            if np.max(np.abs(w - u)) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_weights.append(w)

    return unique_weights


def choose_oracle_fusion(logits_matrix: np.ndarray, y: np.ndarray, step: float = 0.05) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    M = logits_matrix.shape[1]
    best_auc = -np.inf
    best_probs = None
    best_w = None

    for w in simplex_weights(M, step=step):
        probs = weighted_logit_fusion(logits_matrix, w)
        auc = safe_auc(y, probs)
        prauc = safe_prauc(y, probs)
        score = (auc, prauc)
        if score > (best_auc, -np.inf if best_probs is None else safe_prauc(y, best_probs)):
            best_auc = auc
            best_probs = probs
            best_w = w

    info = {
        "oracle_fusion_auc": float(best_auc),
        "oracle_fusion_weights": ",".join([f"{x:.4f}" for x in best_w]),
    }
    return best_w, best_probs, info


# ============================================================
# Target-label direction-bias routing utilities
# ============================================================

def make_target_calibration_split(
    y: np.ndarray,
    calib_ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:

    y = np.asarray(y).astype(int).reshape(-1)
    n = len(y)
    all_idx = np.arange(n)

    calib_ratio = float(calib_ratio)
    if calib_ratio <= 0:
        return np.asarray([], dtype=int), all_idx.astype(int)
    if calib_ratio >= 1:
        raise ValueError("--target_calib_ratio must be < 1.0 because a held-out test split is required.")

    # Use at least one sample from each class in calibration when possible.
    n_calib = int(round(n * calib_ratio))
    n_calib = max(2, min(n - 2, n_calib))

    try:
        calib_idx, test_idx = train_test_split(
            all_idx,
            train_size=n_calib,
            random_state=seed,
            stratify=y,
            shuffle=True,
        )
    except Exception:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(all_idx)
        calib_idx = perm[:n_calib]
        test_idx = perm[n_calib:]

    calib_idx = np.asarray(sorted(calib_idx.tolist()), dtype=int)
    test_idx = np.asarray(sorted(test_idx.tolist()), dtype=int)
    return calib_idx, test_idx


def softmax_np(a: np.ndarray, axis: int = 1) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = a - np.max(a, axis=axis, keepdims=True)
    e = np.exp(a)
    return (e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)).astype(np.float32)


def compute_expert_direction_info(
    logits_calib: np.ndarray,
    y_calib: np.ndarray,
    kappa: float = 30.0,
    bias_mode: str = "reward_only",
) -> Dict[str, np.ndarray]:

    logits_calib = np.asarray(logits_calib, dtype=np.float32)
    y_calib = np.asarray(y_calib).astype(int).reshape(-1)
    M = logits_calib.shape[1]
    n = len(y_calib)

    raw_auc = np.zeros(M, dtype=np.float32)
    for m in range(M):
        raw_auc[m] = safe_auc(y_calib, logits_calib[:, m])

    if kappa < 0:
        kappa = 0.0
    rho = n / (n + float(kappa)) if (n + float(kappa)) > 0 else 1.0
    shrunk_auc = rho * raw_auc + (1.0 - rho) * 0.5

    centered_direction_score = 2.0 * (shrunk_auc - 0.5)

    bias_mode = str(bias_mode).strip().lower()
    if bias_mode == "reward_only":
        # Conservative default for tiny target calibration sets:
        # aligned experts receive positive routing bias;
        # below-0.5 experts are not flipped and not strongly penalized.
        direction_score = np.maximum(centered_direction_score, 0.0)
    elif bias_mode == "centered":
        # More aggressive: below-0.5 experts receive negative routing bias.
        # Still no logit flip.
        direction_score = centered_direction_score
    else:
        raise ValueError(f"Unknown direction_bias_mode: {bias_mode}")

    return {
        "raw_auc": raw_auc.astype(np.float32),
        "shrunk_auc": shrunk_auc.astype(np.float32),
        "centered_direction_score": centered_direction_score.astype(np.float32),
        "direction_score": direction_score.astype(np.float32),
        "rho": np.asarray([rho], dtype=np.float32),
    }


def apply_direction_aware_routing(
    base_routing_weights: np.ndarray,
    expert_logits: np.ndarray,
    direction_score: np.ndarray,
    beta: float = 0.5,
    routing_temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply target-calibrated direction bias to posterior routing weights."""
    base = np.asarray(base_routing_weights, dtype=np.float32)
    logits = np.asarray(expert_logits, dtype=np.float32)
    direction_score = np.asarray(direction_score, dtype=np.float32).reshape(1, -1)

    base = np.clip(base, 1e-12, None)
    base = base / (base.sum(axis=1, keepdims=True) + 1e-12)

    T = max(float(routing_temperature), 1e-6)
    route_logits = np.log(base + 1e-12) / T + float(beta) * direction_score
    direction_routing_weights = softmax_np(route_logits, axis=1)

    final_logits = np.sum(logits * direction_routing_weights, axis=1)
    final_probs = expit(final_logits).astype(np.float32)
    return final_logits.astype(np.float32), final_probs, direction_routing_weights


def parse_float_grid(grid_str: str) -> List[float]:
    """Parse comma-separated float grid, preserving order and removing duplicates."""
    values = []
    if grid_str is None:
        return values
    for item in str(grid_str).split(','):
        item = item.strip()
        if item == '':
            continue
        try:
            values.append(float(item))
        except ValueError:
            raise ValueError(f"Invalid float in grid: {item}")

    unique = []
    for v in values:
        if not any(abs(v - u) < 1e-12 for u in unique):
            unique.append(v)
    return unique


def select_direction_beta_by_calibration(
    base_routing_weights: np.ndarray,
    expert_logits: np.ndarray,
    direction_score: np.ndarray,
    y: np.ndarray,
    calib_idx: np.ndarray,
    beta_grid: List[float],
    routing_temperature: float,
    select_margin: float,
    threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Select a nonzero direction-bias beta only when calibration performance improves."""
    y = np.asarray(y).astype(int).reshape(-1)
    calib_idx = np.asarray(calib_idx, dtype=int).reshape(-1)

    base_routing_weights = np.asarray(base_routing_weights, dtype=np.float32)
    expert_logits = np.asarray(expert_logits, dtype=np.float32)

    posterior_logits = np.sum(expert_logits * base_routing_weights, axis=1).astype(np.float32)
    posterior_probs = expit(posterior_logits).astype(np.float32)

    if len(calib_idx) > 0 and len(np.unique(y[calib_idx])) >= 2:
        posterior_calib_metrics = calculate_metrics(posterior_probs[calib_idx], y[calib_idx], threshold=threshold)
    else:
        posterior_calib_metrics = None

    # Ensure beta=0 and user-supplied values are represented once.
    clean_grid = []
    for b in beta_grid:
        if not any(abs(float(b) - float(u)) < 1e-12 for u in clean_grid):
            clean_grid.append(float(b))
    if not any(abs(b) < 1e-12 for b in clean_grid):
        clean_grid = [0.0] + clean_grid

    candidates = []
    for beta in clean_grid:
        logits_b, probs_b, weights_b = apply_direction_aware_routing(
            base_routing_weights,
            expert_logits,
            direction_score,
            beta=beta,
            routing_temperature=routing_temperature,
        )
        if posterior_calib_metrics is not None:
            calib_metrics_b = calculate_metrics(probs_b[calib_idx], y[calib_idx], threshold=threshold)
        else:
            calib_metrics_b = None
        candidates.append({
            "beta": float(beta),
            "logits": logits_b,
            "probs": probs_b,
            "weights": weights_b,
            "calib_metrics": calib_metrics_b,
        })

    # Default: raw posterior fallback, not temperature-adjusted beta=0 unless it clears the margin.
    selected = {
        "method": "posterior_routed_fallback",
        "beta": 0.0,
        "logits": posterior_logits,
        "probs": posterior_probs,
        "weights": base_routing_weights,
        "calib_metrics": posterior_calib_metrics,
        "posterior_calib_metrics": posterior_calib_metrics,
        "select_reason": "no_valid_calibration",
    }

    if posterior_calib_metrics is None:
        return selected, candidates

    nonzero_candidates = [c for c in candidates if abs(float(c["beta"])) > 1e-12 and c["calib_metrics"] is not None]
    if len(nonzero_candidates) == 0:
        selected["select_reason"] = "no_nonzero_beta_candidate"
        return selected, candidates

    # Select by AUC first, then PRAUC as tie breaker.
    best_nonzero = max(
        nonzero_candidates,
        key=lambda c: (c["calib_metrics"]["auc"], c["calib_metrics"]["prauc"]),
    )

    base_auc = posterior_calib_metrics["auc"]
    best_auc = best_nonzero["calib_metrics"]["auc"]
    margin = float(select_margin)

    if best_auc > base_auc + margin:
        selected = dict(best_nonzero)
        selected["method"] = "direction_beta_selected"
        selected["posterior_calib_metrics"] = posterior_calib_metrics
        selected["select_reason"] = f"calib_auc_gain={best_auc - base_auc:.4f}>margin={margin:.4f}"
    else:
        selected["select_reason"] = f"calib_auc_gain={best_auc - base_auc:.4f}<=margin={margin:.4f}"

    return selected, candidates


def binary_cross_entropy_with_logits_np(logits: np.ndarray, y: np.ndarray) -> float:
    """Compute stable mean BCE from logits using NumPy."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(logits) == 0:
        return float("inf")
    # max(x, 0) - x*y + log(1 + exp(-abs(x)))
    loss = np.maximum(logits, 0.0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return float(np.mean(loss))


def fit_prototype_logit_offset_adapter(
    routing_weights: np.ndarray,
    posterior_logits: np.ndarray,
    y: np.ndarray,
    calib_idx: np.ndarray,
    l2: float = 1.0,
    lr: float = 0.05,
    epochs: int = 300,
    max_abs_delta: float = 0.75,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit a small prototype-level logit offset using target calibration labels."""
    routing_weights = np.asarray(routing_weights, dtype=np.float32)
    posterior_logits = np.asarray(posterior_logits, dtype=np.float32).reshape(-1)
    y = np.asarray(y).astype(np.float32).reshape(-1)
    calib_idx = np.asarray(calib_idx, dtype=int).reshape(-1)

    if routing_weights.ndim != 2:
        raise ValueError("routing_weights must be a 2D array [N, M].")
    N, M = routing_weights.shape
    if len(posterior_logits) != N or len(y) != N:
        raise ValueError("posterior_logits, y, and routing_weights must have the same N.")

    posterior_probs = expit(posterior_logits).astype(np.float32)
    invalid = len(calib_idx) == 0 or len(np.unique(y[calib_idx].astype(int))) < 2
    if invalid:
        return {
            "delta": np.zeros(M, dtype=np.float32),
            "adapted_logits": posterior_logits.copy().astype(np.float32),
            "adapted_probs": posterior_probs,
            "used": False,
            "reason": "invalid_calibration",
            "posterior_calib_bce": float("inf"),
            "offset_calib_bce": float("inf"),
            "bce_improvement": 0.0,
            "best_loss": float("inf"),
        }

    set_seed(seed)
    device = torch.device("cpu")

    W_calib = torch.from_numpy(routing_weights[calib_idx]).float().to(device)
    base_calib = torch.from_numpy(posterior_logits[calib_idx]).float().to(device)
    y_calib = torch.from_numpy(y[calib_idx]).float().to(device)

    delta = torch.zeros(M, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))

    best_state = np.zeros(M, dtype=np.float32)
    best_loss = float("inf")
    max_abs_delta = max(float(max_abs_delta), 0.0)

    for _ in range(int(epochs)):
        correction = W_calib @ delta
        logits = base_calib + correction
        bce = F.binary_cross_entropy_with_logits(logits, y_calib)
        reg = float(l2) * torch.mean(delta ** 2)
        loss = bce + reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if max_abs_delta > 0:
                delta.clamp_(min=-max_abs_delta, max=max_abs_delta)

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = delta.detach().cpu().numpy().astype(np.float32).copy()

    adapted_logits = posterior_logits + routing_weights @ best_state
    adapted_probs = expit(adapted_logits).astype(np.float32)

    posterior_calib_bce = binary_cross_entropy_with_logits_np(posterior_logits[calib_idx], y[calib_idx])
    offset_calib_bce = binary_cross_entropy_with_logits_np(adapted_logits[calib_idx], y[calib_idx])

    return {
        "delta": best_state.astype(np.float32),
        "adapted_logits": adapted_logits.astype(np.float32),
        "adapted_probs": adapted_probs.astype(np.float32),
        "used": True,
        "reason": "fitted",
        "posterior_calib_bce": float(posterior_calib_bce),
        "offset_calib_bce": float(offset_calib_bce),
        "bce_improvement": float(posterior_calib_bce - offset_calib_bce),
        "best_loss": float(best_loss),
    }



def fit_prototype_ranking_offset_adapter(
    routing_weights: np.ndarray,
    posterior_logits: np.ndarray,
    y: np.ndarray,
    calib_idx: np.ndarray,
    l2: float = 1.0,
    lr: float = 0.03,
    epochs: int = 300,
    max_abs_delta: float = 0.50,
    temperature: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit a small prototype-level offset with a pairwise ranking loss."""
    routing_weights = np.asarray(routing_weights, dtype=np.float32)
    posterior_logits = np.asarray(posterior_logits, dtype=np.float32).reshape(-1)
    y = np.asarray(y).astype(np.float32).reshape(-1)
    calib_idx = np.asarray(calib_idx, dtype=int).reshape(-1)

    if routing_weights.ndim != 2:
        raise ValueError("routing_weights must be a 2D array [N, M].")
    N, M = routing_weights.shape
    if len(posterior_logits) != N or len(y) != N:
        raise ValueError("posterior_logits, y, and routing_weights must have the same N.")

    posterior_probs = expit(posterior_logits).astype(np.float32)
    invalid = len(calib_idx) == 0 or len(np.unique(y[calib_idx].astype(int))) < 2
    if invalid:
        return {
            "delta": np.zeros(M, dtype=np.float32),
            "adapted_logits": posterior_logits.copy().astype(np.float32),
            "adapted_probs": posterior_probs,
            "used": False,
            "reason": "invalid_calibration",
            "posterior_calib_auc": 0.5,
            "rank_offset_calib_auc": 0.5,
            "auc_improvement": 0.0,
            "posterior_pairwise_loss": float("inf"),
            "rank_offset_pairwise_loss": float("inf"),
            "pairwise_loss_improvement": 0.0,
            "best_loss": float("inf"),
        }

    pos_idx = calib_idx[y[calib_idx] == 1]
    neg_idx = calib_idx[y[calib_idx] == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "delta": np.zeros(M, dtype=np.float32),
            "adapted_logits": posterior_logits.copy().astype(np.float32),
            "adapted_probs": posterior_probs,
            "used": False,
            "reason": "calibration_missing_pos_or_neg",
            "posterior_calib_auc": 0.5,
            "rank_offset_calib_auc": 0.5,
            "auc_improvement": 0.0,
            "posterior_pairwise_loss": float("inf"),
            "rank_offset_pairwise_loss": float("inf"),
            "pairwise_loss_improvement": 0.0,
            "best_loss": float("inf"),
        }

    set_seed(seed)
    device = torch.device("cpu")
    W_pos = torch.from_numpy(routing_weights[pos_idx]).float().to(device)
    W_neg = torch.from_numpy(routing_weights[neg_idx]).float().to(device)
    base_pos = torch.from_numpy(posterior_logits[pos_idx]).float().to(device)
    base_neg = torch.from_numpy(posterior_logits[neg_idx]).float().to(device)

    delta = torch.zeros(M, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))

    best_state = np.zeros(M, dtype=np.float32)
    best_loss = float("inf")
    max_abs_delta = max(float(max_abs_delta), 0.0)
    tau = max(float(temperature), 1e-6)

    def pairwise_loss_from_delta(d: torch.Tensor) -> torch.Tensor:
        pos_logits = base_pos + W_pos @ d
        neg_logits = base_neg + W_neg @ d
        diff = pos_logits.view(-1, 1) - neg_logits.view(1, -1)
        rank_loss = F.softplus(-diff / tau).mean()
        reg = float(l2) * torch.mean(d ** 2)
        return rank_loss + reg

    for _ in range(int(epochs)):
        loss = pairwise_loss_from_delta(delta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if max_abs_delta > 0:
                delta.clamp_(min=-max_abs_delta, max=max_abs_delta)

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = delta.detach().cpu().numpy().astype(np.float32).copy()

    adapted_logits = posterior_logits + routing_weights @ best_state
    adapted_probs = expit(adapted_logits).astype(np.float32)

    posterior_calib_auc = safe_auc(y[calib_idx].astype(int), posterior_logits[calib_idx])
    rank_offset_calib_auc = safe_auc(y[calib_idx].astype(int), adapted_logits[calib_idx])

    # Diagnostics: unregularized pairwise loss on the calibration pairs.
    def pairwise_loss_np(logits: np.ndarray) -> float:
        pos = np.asarray(logits[pos_idx], dtype=np.float64).reshape(-1)
        neg = np.asarray(logits[neg_idx], dtype=np.float64).reshape(-1)
        diff = pos[:, None] - neg[None, :]
        loss = np.logaddexp(0.0, -diff / tau).mean()
        return float(loss)

    posterior_pairwise_loss = pairwise_loss_np(posterior_logits)
    rank_offset_pairwise_loss = pairwise_loss_np(adapted_logits)

    return {
        "delta": best_state.astype(np.float32),
        "adapted_logits": adapted_logits.astype(np.float32),
        "adapted_probs": adapted_probs.astype(np.float32),
        "used": True,
        "reason": "fitted",
        "posterior_calib_auc": float(posterior_calib_auc),
        "rank_offset_calib_auc": float(rank_offset_calib_auc),
        "auc_improvement": float(rank_offset_calib_auc - posterior_calib_auc),
        "posterior_pairwise_loss": float(posterior_pairwise_loss),
        "rank_offset_pairwise_loss": float(rank_offset_pairwise_loss),
        "pairwise_loss_improvement": float(posterior_pairwise_loss - rank_offset_pairwise_loss),
        "best_loss": float(best_loss),
    }


# ============================================================
# Core LODO function
# ============================================================

def run_one_target_prototype(target_idx: int, all_data: List[CohortData], args, pretrain_state_dict: Optional[Dict[str, torch.Tensor]] = None, pretrain_dataset_name: Optional[str] = None) -> Dict[str, Any]:
    target = all_data[target_idx]
    sources = [all_data[i] for i in range(len(all_data)) if i != target_idx]

    # If In-house is used for AE pretraining, remove it from later LODO supervised source training.
    if (
        args.use_inhouse_ae_pretrain
        and args.exclude_pretrain_from_sources
        and pretrain_dataset_name is not None
    ):
        sources = [
            d for d in sources
            if normalize_dataset_name(d.name) != normalize_dataset_name(pretrain_dataset_name)
        ]

    if len(sources) == 0:
        raise RuntimeError(
            f"Target {target.name}: source datasets are empty after excluding pretrain dataset."
        )

    source_names = [d.name for d in sources]

    print("\n" + "=" * 90)
    print(f"Target: {target.name}")
    print(f"Sources: {source_names}")
    if (
        args.use_inhouse_ae_pretrain
        and args.exclude_pretrain_from_sources
        and pretrain_dataset_name is not None
    ):
        print(f"Excluded AE pretrain dataset from sources: {pretrain_dataset_name}")
    print("=" * 90)

    print(f"Target n={len(target.y)}, TD={(target.y == 0).sum()}, ASD={(target.y == 1).sum()}")
    for d in sources:
        print(f"  Source {d.name}: n={len(d.y)}, TD={(d.y == 0).sum()}, ASD={(d.y == 1).sum()}")

    # ------------------------------------------------------------------
    # Strict target split: calibration labels are allowed; test labels are held out.
    # ------------------------------------------------------------------
    calib_idx, test_idx = make_target_calibration_split(
        target.y,
        calib_ratio=args.target_calib_ratio,
        seed=args.seed + 10000 + target_idx,
    )
    if len(calib_idx) > 0 and len(np.unique(target.y[calib_idx])) < 2:
        print("  WARNING: target calibration split has only one class; direction calibration will be weak.")
    if len(test_idx) > 0 and len(np.unique(target.y[test_idx])) < 2:
        print("  WARNING: target test split has only one class; AUC will fall back to 0.5.")

    print(
        f"  Target labeled calibration split: {len(calib_idx)}/{len(target.y)} "
        f"({100.0 * len(calib_idx) / max(1, len(target.y)):.1f}%), "
        f"TD={(target.y[calib_idx] == 0).sum() if len(calib_idx) > 0 else 0}, "
        f"ASD={(target.y[calib_idx] == 1).sum() if len(calib_idx) > 0 else 0}"
    )
    print(
        f"  Target held-out test split: {len(test_idx)}/{len(target.y)}, "
        f"TD={(target.y[test_idx] == 0).sum() if len(test_idx) > 0 else 0}, "
        f"ASD={(target.y[test_idx] == 1).sum() if len(test_idx) > 0 else 0}"
    )

    use_pretrain_this_fold = (
        pretrain_state_dict is not None
        and args.use_inhouse_ae_pretrain
        and args.model_type == "mlp"
    )
    if (
        use_pretrain_this_fold
        and (not args.allow_pretrain_on_target)
        and pretrain_dataset_name is not None
        and normalize_dataset_name(target.name) == normalize_dataset_name(pretrain_dataset_name)
    ):
        use_pretrain_this_fold = False

    init_state = pretrain_state_dict if use_pretrain_this_fold else None
    if use_pretrain_this_fold:
        print(f"  Use masked AE encoder initialization from {pretrain_dataset_name} ({args.ae_transfer_mode}).")
    elif args.use_inhouse_ae_pretrain and args.model_type == "mlp":
        print("  Do not use AE pretrained initialization for this fold.")

    target_dir = os.path.join(args.output_dir, f"target_{target.name}")
    os.makedirs(target_dir, exist_ok=True)

    # ----- Step 1: Discover prototypes from source data (PCA + GMM) -----
    base_transform = args.base_transform
    X_raw_list = [d.X_raw for d in sources]
    X_trans_list = [transform_abundance(x, method=base_transform) for x in X_raw_list]
    X_merged = np.vstack(X_trans_list).astype(np.float32)

    scaler_proto = StandardScaler()
    X_scaled = scaler_proto.fit_transform(X_merged)

    n_pca = min(args.pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=args.seed)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)

    n_components_range = [int(x) for x in str(args.gmm_components).split(",") if str(x).strip()]
    best_gmm = None
    best_bic = np.inf
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
        except ValueError as e:
            print(f"  GMM with {n_comp} components failed: {e}. Skipping.")
            continue

    if best_gmm is None:
        print("  All GMM attempts failed; using fallback with n_components=2 and stronger regularization.")
        best_gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=args.seed,
            reg_covar=1e-3,
        )
        best_gmm.fit(X_pca)

    posteriors_source = best_gmm.predict_proba(X_pca)
    M = posteriors_source.shape[1]
    print(f"  Discovered {M} prototypes (GMM components).")

    # ----- Step 2: Preprocess source data for training -----
    X_train_sources, _ = preprocess_blocks(
        X_raw_list,
        mode=args.preprocess,
        base_transform=args.base_transform,
        rank_clip=args.rank_clip,
        query_raw=None,
    )
    y_merged = np.concatenate([d.y for d in sources])

    # ----- Step 3: Train one expert per prototype with soft weights -----
    experts = []
    class_w = make_sample_weights(y_merged)
    for m in range(M):
        sw = posteriors_source[:, m].astype(np.float64)
        sw = sw * class_w
        sw = sw / (sw.mean() + 1e-12)
        if np.sum(sw) < 1e-6:
            continue
        model_obj = fit_classifier(
            X_train_sources,
            y_merged,
            args,
            seed=args.seed + 1000 + m,
            supcon_weight=args.supcon_weight,
            sample_weights=sw,
            init_state_dict=init_state,
            init_mode=args.ae_transfer_mode,
        )
        experts.append({
            "prototype_index": m,
            "model": model_obj,
        })

    if len(experts) == 0:
        raise RuntimeError(f"Target {target.name}: no available experts.")

    # ----- Step 4: Preprocess target and compute source-GMM posterior routing -----
    # Note: this uses target X only, not target labels. It is consistent with transductive cohort-level preprocessing.
    _, X_target = preprocess_blocks(
        [target.X_raw],
        mode=args.preprocess,
        base_transform=args.base_transform,
        rank_clip=args.rank_clip,
        query_raw=target.X_raw,
    )

    # Correct AdaBN use: adapt BN with the processed target input distribution, not raw abundance.
    if args.adapt_bn and args.model_type == "mlp":
        for exp in experts:
            model_obj = exp.get("model")
            model = None if model_obj is None else model_obj.get("model")
            if model is not None and hasattr(model, "use_bn") and model.use_bn:
                adapt_bn(model, X_target, batch_size=args.batch_size, device=args.device)

    X_target_trans = transform_abundance(target.X_raw, method=base_transform)
    X_target_scaled = scaler_proto.transform(X_target_trans)
    X_target_pca = pca.transform(X_target_scaled)
    posteriors_target = best_gmm.predict_proba(X_target_pca)

    # Predict target with each expert.
    expert_logits = []
    expert_probs = []
    for exp in experts:
        logits, probs = predict_classifier(exp["model"], X_target, args)
        expert_logits.append(logits)
        expert_probs.append(probs)
    expert_logits = np.column_stack(expert_logits).astype(np.float32)
    expert_probs = np.column_stack(expert_probs).astype(np.float32)

    # Build source-GMM posterior routing weights for available experts.
    exp_proto_indices = [e["prototype_index"] for e in experts]
    routing_weights = np.zeros((len(target.y), len(experts)), dtype=np.float32)
    for i, proto_idx in enumerate(exp_proto_indices):
        routing_weights[:, i] = posteriors_target[:, proto_idx]
    routing_weights = routing_weights / (routing_weights.sum(axis=1, keepdims=True) + 1e-12)

    # ----- Step 5: Original posterior-routed baseline -----
    final_logits_routed = np.sum(expert_logits * routing_weights, axis=1)
    final_probs_routed = expit(final_logits_routed).astype(np.float32)

    # Optional target-calibrated prototype logit offset adapter.
    # It is anchored to posterior-routed logits and learns only M small offsets.
    if args.use_offset_adapter:
        offset_adapter = fit_prototype_logit_offset_adapter(
            routing_weights=routing_weights,
            posterior_logits=final_logits_routed,
            y=target.y,
            calib_idx=calib_idx,
            l2=args.offset_adapter_l2,
            lr=args.offset_adapter_lr,
            epochs=args.offset_adapter_epochs,
            max_abs_delta=args.offset_adapter_max_abs_delta,
            seed=args.seed + 30000 + target_idx,
        )
        final_logits_offset = offset_adapter["adapted_logits"]
        final_probs_offset = offset_adapter["adapted_probs"]
    else:
        offset_adapter = {
            "delta": np.zeros(routing_weights.shape[1], dtype=np.float32),
            "adapted_logits": final_logits_routed.copy().astype(np.float32),
            "adapted_probs": final_probs_routed.copy().astype(np.float32),
            "used": False,
            "reason": "disabled",
            "posterior_calib_bce": float("inf"),
            "offset_calib_bce": float("inf"),
            "bce_improvement": 0.0,
            "best_loss": float("inf"),
        }
        final_logits_offset = final_logits_routed.copy().astype(np.float32)
        final_probs_offset = final_probs_routed.copy().astype(np.float32)

    # Optional AUC-oriented pairwise ranking prototype offset adapter.
    # It has the same small parameterization as the BCE offset adapter, but trains with a pairwise ranking loss.
    if args.use_rank_offset_adapter:
        rank_offset_adapter = fit_prototype_ranking_offset_adapter(
            routing_weights=routing_weights,
            posterior_logits=final_logits_routed,
            y=target.y,
            calib_idx=calib_idx,
            l2=args.rank_offset_adapter_l2,
            lr=args.rank_offset_adapter_lr,
            epochs=args.rank_offset_adapter_epochs,
            max_abs_delta=args.rank_offset_adapter_max_abs_delta,
            temperature=args.rank_offset_adapter_temperature,
            seed=args.seed + 40000 + target_idx,
        )
        final_logits_rank_offset = rank_offset_adapter["adapted_logits"]
        final_probs_rank_offset = rank_offset_adapter["adapted_probs"]
    else:
        rank_offset_adapter = {
            "delta": np.zeros(routing_weights.shape[1], dtype=np.float32),
            "adapted_logits": final_logits_routed.copy().astype(np.float32),
            "adapted_probs": final_probs_routed.copy().astype(np.float32),
            "used": False,
            "reason": "disabled",
            "posterior_calib_auc": 0.5,
            "rank_offset_calib_auc": 0.5,
            "auc_improvement": 0.0,
            "posterior_pairwise_loss": float("inf"),
            "rank_offset_pairwise_loss": float("inf"),
            "pairwise_loss_improvement": 0.0,
            "best_loss": float("inf"),
        }
        final_logits_rank_offset = final_logits_routed.copy().astype(np.float32)
        final_probs_rank_offset = final_probs_routed.copy().astype(np.float32)

    # Equal weights baseline.
    w_equal = np.ones(len(experts), dtype=np.float32) / len(experts)
    final_probs_equal = weighted_logit_fusion(expert_logits, w_equal)

    # Global model baseline.
    global_model = fit_classifier(
        X_train_sources,
        y_merged,
        args,
        seed=args.seed + 9999,
        supcon_weight=args.supcon_weight * 0.5,
        init_state_dict=init_state,
        init_mode=args.ae_transfer_mode,
    )
    if args.adapt_bn and args.model_type == "mlp" and global_model is not None:
        model = global_model.get("model")
        if model is not None and hasattr(model, "use_bn") and model.use_bn:
            adapt_bn(model, X_target, batch_size=args.batch_size, device=args.device)
    global_logits, global_probs = predict_classifier(global_model, X_target, args)

    # ----- Step 6: Target-label direction-bias routing with conservative beta selection -----
    if len(calib_idx) > 0:
        direction_info = compute_expert_direction_info(
            expert_logits[calib_idx],
            target.y[calib_idx],
            kappa=args.direction_kappa,
            bias_mode=args.direction_bias_mode,
        )
    else:
        direction_info = {
            "raw_auc": np.full(len(experts), 0.5, dtype=np.float32),
            "shrunk_auc": np.full(len(experts), 0.5, dtype=np.float32),
            "direction_score": np.zeros(len(experts), dtype=np.float32),
            "centered_direction_score": np.zeros(len(experts), dtype=np.float32),
            "rho": np.asarray([0.0], dtype=np.float32),
        }

    beta_grid = parse_float_grid(args.direction_beta_grid)
    # Keep backward compatibility: --direction_beta is automatically included in the grid.
    beta_grid = [0.0] + beta_grid + [float(args.direction_beta)]

    selected_direction, direction_beta_candidates = select_direction_beta_by_calibration(
        routing_weights,
        expert_logits,
        direction_info["direction_score"],
        target.y,
        calib_idx,
        beta_grid=beta_grid,
        routing_temperature=args.direction_routing_temperature,
        select_margin=args.direction_select_margin,
        threshold=args.threshold,
    )

    final_logits_direction = selected_direction["logits"]
    final_probs_direction = selected_direction["probs"]
    direction_routing_weights = selected_direction["weights"]
    selected_direction_beta = float(selected_direction["beta"])
    direction_select_reason = str(selected_direction["select_reason"])
    posterior_calib_metrics = selected_direction.get("posterior_calib_metrics", None)

    # Equal fusion is unchanged because label direction now affects routing weights only, not expert logits.
    final_probs_direction_equal = final_probs_equal

    # Routing confidence based on selected routed weights.
    routing_entropy = -np.sum(
        direction_routing_weights * np.log(direction_routing_weights + 1e-12),
        axis=1,
    )
    max_entropy = np.log(direction_routing_weights.shape[1] + 1e-12)
    routing_confidence = 1.0 - float(routing_entropy.mean() / (max_entropy + 1e-12))

    # Safe fallback / blend. If beta selection falls back to posterior routing and route_alpha=1,
    # this exactly recovers the original posterior-routed model.
    if routing_confidence < args.routing_conf_threshold:
        final_logits_safe = global_logits
        final_probs_safe = global_probs.astype(np.float32)
        safe_method = "global_fallback"
    else:
        final_logits_safe = (
            args.route_alpha * final_logits_direction
            + (1.0 - args.route_alpha) * global_logits
        )
        final_probs_safe = expit(final_logits_safe).astype(np.float32)
        if selected_direction["method"] == "posterior_routed_fallback" and abs(float(args.route_alpha) - 1.0) < 1e-12:
            safe_method = "posterior_routed_fallback"
        elif selected_direction["method"] == "posterior_routed_fallback":
            safe_method = "posterior_global_logit_blend"
        else:
            safe_method = f"direction_beta_selected_{selected_direction_beta:g}_global_logit_blend"

    # ----- Step 7: Calibration-selected fusion, optional diagnostic -----
    # This is not a test-label oracle. It chooses weights on calibration labels and applies them to test.
    if args.oracle_tune and len(calib_idx) > 0:
        w_oracle, probs_oracle_calib, info_oracle = choose_oracle_fusion(
            expert_logits[calib_idx],
            target.y[calib_idx],
            step=args.fusion_step,
        )
        probs_oracle = weighted_logit_fusion(expert_logits, w_oracle)
        oracle_method_name = "calibration_selected_fusion"
    else:
        w_oracle = w_equal
        probs_oracle = final_probs_equal
        oracle_method_name = "equal_weights"

    # ------------------------------------------------------------
    # Evaluation: report held-out test metrics only.
    # ------------------------------------------------------------
    y_test = target.y[test_idx]

    global_metrics = calculate_metrics(global_probs[test_idx], y_test, threshold=args.threshold)
    m_equal = calculate_metrics(final_probs_equal[test_idx], y_test, threshold=args.threshold)
    m_routed = calculate_metrics(final_probs_routed[test_idx], y_test, threshold=args.threshold)
    m_offset = calculate_metrics(final_probs_offset[test_idx], y_test, threshold=args.threshold)
    m_rank_offset = calculate_metrics(final_probs_rank_offset[test_idx], y_test, threshold=args.threshold)
    m_direction_equal = calculate_metrics(final_probs_direction_equal[test_idx], y_test, threshold=args.threshold)
    m_direction = calculate_metrics(final_probs_direction[test_idx], y_test, threshold=args.threshold)
    m_safe = calculate_metrics(final_probs_safe[test_idx], y_test, threshold=args.threshold)
    m_oracle = calculate_metrics(probs_oracle[test_idx], y_test, threshold=args.threshold)

    # Calibration metrics are saved for diagnosing overfitting.
    y_calib = target.y[calib_idx] if len(calib_idx) > 0 else np.asarray([], dtype=int)
    calib_metrics_direction = (
        calculate_metrics(final_probs_direction[calib_idx], y_calib, threshold=args.threshold)
        if len(calib_idx) > 0 else None
    )
    calib_metrics_offset = (
        calculate_metrics(final_probs_offset[calib_idx], y_calib, threshold=args.threshold)
        if len(calib_idx) > 0 else None
    )
    calib_metrics_rank_offset = (
        calculate_metrics(final_probs_rank_offset[calib_idx], y_calib, threshold=args.threshold)
        if len(calib_idx) > 0 else None
    )
    calib_metrics_safe = (
        calculate_metrics(final_probs_safe[calib_idx], y_calib, threshold=args.threshold)
        if len(calib_idx) > 0 else None
    )

    # ------------------------------------------------------------
    # Original selector + TEST-AUC oracle selector.
    # ------------------------------------------------------------
    # original_* reproduces the original code path:
    #   - if --oracle_tune is enabled, use the calibration-selected fusion;
    #   - otherwise start from safe_method;
    #   - optionally override with offset / rank-offset only when their
    #     calibration criteria pass.
    # oracle_* is a debug/upper-bound analysis that directly picks the best
    # held-out TEST AUC among the six printed candidates below.

    offset_selected = False
    rank_offset_selected = False
    offset_select_reason = "disabled_or_report_only"
    rank_offset_select_reason = "disabled_or_report_only"

    if args.oracle_tune:
        original_method = oracle_method_name
        original_probs = probs_oracle
        original_metrics = m_oracle
        original_weights = w_oracle
        original_select_reason = "oracle_tune_calibration_selected_fusion"
    else:
        original_method = safe_method
        original_probs = final_probs_safe
        original_metrics = m_safe
        original_weights = direction_routing_weights.mean(axis=0)
        original_select_reason = f"default_safe_method={safe_method}"

        # Optional conservative use of the BCE offset adapter.
        if args.use_offset_adapter and args.offset_adapter_select and offset_adapter.get("used", False):
            bce_gain = float(offset_adapter.get("bce_improvement", 0.0))
            if bce_gain > float(args.offset_adapter_bce_margin):
                original_method = "prototype_logit_offset_adapter"
                original_probs = final_probs_offset
                original_metrics = m_offset
                original_weights = routing_weights.mean(axis=0)
                offset_selected = True
                offset_select_reason = (
                    f"calib_bce_gain={bce_gain:.6f}>"
                    f"margin={float(args.offset_adapter_bce_margin):.6f}"
                )
                original_select_reason = offset_select_reason
            else:
                offset_select_reason = (
                    f"calib_bce_gain={bce_gain:.6f}<="
                    f"margin={float(args.offset_adapter_bce_margin):.6f}"
                )
        elif args.use_offset_adapter and offset_adapter.get("used", False):
            offset_select_reason = "report_only"
        else:
            offset_select_reason = str(offset_adapter.get("reason", "disabled"))

        # Optional conservative use of the ranking/AUC offset adapter.
        if args.use_rank_offset_adapter and args.rank_offset_adapter_select and rank_offset_adapter.get("used", False):
            auc_gain = float(rank_offset_adapter.get("auc_improvement", 0.0))
            if auc_gain > float(args.rank_offset_adapter_auc_margin):
                original_method = "prototype_ranking_offset_adapter"
                original_probs = final_probs_rank_offset
                original_metrics = m_rank_offset
                original_weights = routing_weights.mean(axis=0)
                rank_offset_selected = True
                offset_selected = False
                rank_offset_select_reason = (
                    f"calib_auc_gain={auc_gain:.6f}>"
                    f"margin={float(args.rank_offset_adapter_auc_margin):.6f}"
                )
                original_select_reason = rank_offset_select_reason
            else:
                rank_offset_select_reason = (
                    f"calib_auc_gain={auc_gain:.6f}<="
                    f"margin={float(args.rank_offset_adapter_auc_margin):.6f}"
                )
        elif args.use_rank_offset_adapter and rank_offset_adapter.get("used", False):
            rank_offset_select_reason = "report_only"
        else:
            rank_offset_select_reason = str(rank_offset_adapter.get("reason", "disabled"))

    test_auc_candidates = [
        ("global", global_probs, np.asarray([], dtype=np.float32), global_metrics),
        ("posterior_routed", final_probs_routed, routing_weights.mean(axis=0), m_routed),
        ("prototype_logit_offset_adapter", final_probs_offset, routing_weights.mean(axis=0), m_offset),
        ("prototype_ranking_offset_adapter", final_probs_rank_offset, routing_weights.mean(axis=0), m_rank_offset),
        ("direction_aware_routed", final_probs_direction, direction_routing_weights.mean(axis=0), m_direction),
        ("safe", final_probs_safe, direction_routing_weights.mean(axis=0), m_safe),
    ]

    oracle_method, oracle_probs, oracle_weights, oracle_metrics = max(
        test_auc_candidates,
        key=lambda x: (
            x[3]["auc"],
            x[3]["prauc"],
            x[3]["accuracy"],
            x[3]["f1"],
        ),
    )

    print(
        f"  TEST global AUC={global_metrics['auc']:.4f}, "
        f"posterior-routed AUC={m_routed['auc']:.4f}, "
        f"offset-adapter AUC={m_offset['auc']:.4f}, "
        f"rank-offset AUC={m_rank_offset['auc']:.4f}, "
        f"direction-routed AUC={m_direction['auc']:.4f}, "
        f"safe AUC={m_safe['auc']:.4f}"
    )
    print(
        f"Original selector for target {target.name}: {original_method}, "
        f"TEST AUC={original_metrics['auc']:.4f}, PRAUC={original_metrics['prauc']:.4f}, "
        f"reason={original_select_reason}"
    )
    print(
        f"Oracle TEST-AUC analysis for target {target.name}: {oracle_method}, "
        f"BEST TEST AUC={oracle_metrics['auc']:.4f}, PRAUC={oracle_metrics['prauc']:.4f}"
    )

    # The verbose per-candidate tuning table is intentionally removed in this compact version.

    # Save predictions for all target samples; split column tells whether labels were used for calibration.
    split_name = np.asarray(["test"] * len(target.y), dtype=object)
    split_name[calib_idx] = "calibration"
    pred_df = pd.DataFrame({
        "target_dataset": target.name,
        "sample_id": target.sample_ids,
        "split": split_name,
        "true_label": target.y.astype(int),
        "selected_method": original_method,
        "routing_confidence": routing_confidence,
        "selected_direction_beta": selected_direction_beta,
        "direction_select_reason": direction_select_reason,
        "selected_prob": original_probs,
        "selected_pred_label": (original_probs >= args.threshold).astype(int),
        "original_method": original_method,
        "original_prob": original_probs,
        "original_pred_label": (original_probs >= args.threshold).astype(int),
        "original_select_reason": original_select_reason,
        "oracle_method": oracle_method,
        "oracle_prob": oracle_probs,
        "oracle_pred_label": (oracle_probs >= args.threshold).astype(int),
        "global_prob": global_probs,
        "posterior_routed_prob": final_probs_routed,
        "prototype_offset_prob": final_probs_offset,
        "prototype_offset_logit": final_logits_offset,
        "prototype_rank_offset_prob": final_probs_rank_offset,
        "prototype_rank_offset_logit": final_logits_rank_offset,
        "direction_routed_prob": final_probs_direction,
        "direction_equal_prob": final_probs_direction_equal,
        "safe_prob": final_probs_safe,
        "equal_prob": final_probs_equal,
    })
    for i, exp in enumerate(experts):
        pred_df[f"expert_{i}_proto{exp['prototype_index']}_prob"] = expert_probs[:, i]
        pred_df[f"expert_{i}_proto{exp['prototype_index']}_logit"] = expert_logits[:, i]
        pred_df[f"expert_{i}_proto{exp['prototype_index']}_direction_bias"] = float(direction_info["direction_score"][i])
        pred_df[f"expert_{i}_proto{exp['prototype_index']}_offset_delta"] = float(np.asarray(offset_adapter.get("delta", np.zeros(len(experts))))[i])
        pred_df[f"expert_{i}_proto{exp['prototype_index']}_rank_offset_delta"] = float(np.asarray(rank_offset_adapter.get("delta", np.zeros(len(experts))))[i])
        pred_df[f"route_raw_expert_{i}_proto{exp['prototype_index']}"] = routing_weights[:, i]
        pred_df[f"route_direction_expert_{i}_proto{exp['prototype_index']}"] = direction_routing_weights[:, i]
    pred_df.to_csv(os.path.join(target_dir, "predictions.csv"), index=False)

    summary = {
        "target_dataset": target.name,
        "original_method": original_method,
        "original_auc": float(original_metrics["auc"]),
        "original_prauc": float(original_metrics["prauc"]),
        "oracle_method": oracle_method,
        "oracle_auc": float(oracle_metrics["auc"]),
        "oracle_prauc": float(oracle_metrics["prauc"]),
    }
    return summary

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prototype-based Soft MoE for LODO generalization"
    )

    # Data
    parser.add_argument("--datasets_dir", type=str, default="XXX", help="Directory containing dataset subfolders")
    parser.add_argument("--species_list", type=str, default="XXX", help="Path to the text file listing important species.")
    parser.add_argument("--output_dir", type=str, default="XXX", help="Directory to save results.")
    parser.add_argument("--group_mapping", type=str, default="XXX", help="Path to the CSV file mapping datasets to groups.")

    # Preprocessing
    parser.add_argument("--base_transform", type=str, default="arcsinh",
                        choices=["none", "log1p", "relative_log1p", "hellinger", "clr", "arcsinh"])
    parser.add_argument("--preprocess", type=str, default="cohort_rank_gauss",
                        choices=["cohort_rank_gauss", "pooled_robust_zscore", "pooled_standard_zscore"])
    parser.add_argument("--rank_clip", type=float, default=3.0)

    # Model
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "logistic"])
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--hidden_dims", type=str, default="64,32")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--balanced_sampling", action="store_true", default=True)
    parser.add_argument("--no_balanced_sampling", action="store_false", dest="balanced_sampling")
    parser.add_argument("--use_layernorm", action="store_true", default=True)
    parser.add_argument("--no_layernorm", action="store_false", dest="use_layernorm")
    parser.add_argument("--use_bn", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:5" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)

    # Prototype discovery
    parser.add_argument("--pca_components", type=int, default=20)
    parser.add_argument("--gmm_components", type=str, default="1,2,3,4")

    # Enhanced options
    parser.add_argument("--supcon_weight", type=float, default=0.1)
    parser.add_argument("--use_swa", action="store_true", default=True)
    parser.add_argument("--swa_start_ratio", type=float, default=0.75)
    parser.add_argument("--adapt_bn", action="store_true", default=False, help="Apply AdaBN on experts before prediction (requires --use_bn)")

    # In-house masked denoising autoencoder pretraining.
    parser.add_argument("--use_inhouse_ae_pretrain", action="store_true", default=True)
    parser.add_argument("--no_inhouse_ae_pretrain", action="store_false", dest="use_inhouse_ae_pretrain")
    parser.add_argument("--pretrain_dataset", type=str, default="Inhouse")
    parser.add_argument("--ae_pretrain_epochs", type=int, default=150)
    parser.add_argument("--ae_pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--ae_weight_decay", type=float, default=1e-4)
    parser.add_argument("--ae_mask_ratio", type=float, default=0.4)
    parser.add_argument("--ae_mask_value", type=float, default=0.0)
    parser.add_argument("--ae_noise_std", type=float, default=0.05)
    parser.add_argument("--ae_dropout", type=float, default=-1.0,
                        help="AE encoder/decoder dropout. If <0, use the classifier --dropout value.")
    parser.add_argument("--ae_loss_on_masked_only", action="store_true", default=False,
                        help="If enabled, compute AE MSE only on masked entries; default reconstructs all entries.")
    parser.add_argument("--ae_transfer_mode", type=str, default="encoder", choices=["encoder", "first_layer", "none"])
    parser.add_argument("--ae_verbose", action="store_true", default=False)
    parser.add_argument("--ae_print_every", type=int, default=25)
    parser.add_argument(
        "--exclude_pretrain_from_sources",
        action="store_true",
        default=True,
        help="If enabled, the AE pretrain dataset is used only for AE pretraining and removed from LODO source datasets."
    )
    parser.add_argument(
        "--include_pretrain_as_source",
        action="store_false",
        dest="exclude_pretrain_from_sources",
        help="If enabled, the AE pretrain dataset is also used as a LODO source after pretraining."
    )
    parser.add_argument(
        "--allow_pretrain_on_target",
        action="store_true",
        default=False,
        help="If enabled, still use the AE pretrained initialization when that dataset is the LODO target. Disabled by default to avoid leakage."
    )

    # Target calibration and label-direction-bias routing
    parser.add_argument(
        "--target_calib_ratio",
        type=float,
        default=0.00,
        help="Fraction of target labels used only for direction calibration; the remainder is held-out test. If 0, no target labels are used for calibration and the model falls back to posterior routing."
    )
    parser.add_argument(
        "--direction_beta",
        type=float,
        default=0.5,
        help="Legacy single beta value; it is automatically included in --direction_beta_grid."
    )
    parser.add_argument(
        "--direction_beta_grid",
        type=str,
        default="0,0.5,1,2,4",
        help="Comma-separated beta candidates for calibration-based direction-bias selection. beta=0 is posterior fallback."
    )
    parser.add_argument(
        "--direction_select_margin",
        type=float,
        default=0.005,
        help="A nonzero beta is selected only if its calibration AUC beats posterior routing by this margin."
    )
    parser.add_argument(
        "--direction_routing_temperature",
        type=float,
        default=1.0,
        help="Temperature for distribution posterior routing before adding direction bias."
    )
    parser.add_argument(
        "--direction_kappa",
        type=float,
        default=30.0,
        help="Shrinkage strength for expert calibration AUC; larger means more shrinkage toward 0.5."
    )
    parser.add_argument(
        "--direction_bias_mode",
        type=str,
        default="reward_only",
        choices=["reward_only", "centered"],
        help="How to convert calibration AUC into routing bias. reward_only only boosts aligned experts; centered also downweights below-0.5 experts. No mode flips logits."
    )
    # Deprecated compatibility switches. They are intentionally ignored in this bias-only version.
    parser.add_argument("--direction_flip", action="store_true", default=False, help="Deprecated/ignored. Expert logits are never flipped.")
    parser.add_argument("--no_direction_flip", action="store_false", dest="direction_flip", help="Deprecated/ignored. Expert logits are never flipped.")

    # Prototype logit offset adapter.
    # This is a strongly-regularized target calibration layer anchored to posterior routing.
    parser.add_argument("--use_offset_adapter", action="store_true", default=True,
                        help="Fit a tiny prototype logit offset adapter on target calibration labels.")
    parser.add_argument("--no_offset_adapter", action="store_false", dest="use_offset_adapter",
                        help="Disable the prototype logit offset adapter.")
    parser.add_argument("--offset_adapter_l2", type=float, default=1.0,
                        help="L2 regularization for the prototype logit offset adapter.")
    parser.add_argument("--offset_adapter_lr", type=float, default=0.05,
                        help="Learning rate for the prototype logit offset adapter.")
    parser.add_argument("--offset_adapter_epochs", type=int, default=50,
                        help="Training epochs for the prototype logit offset adapter.")
    parser.add_argument("--offset_adapter_max_abs_delta", type=float, default=0.75,
                        help="Clamp each prototype logit offset to [-value, value].")
    parser.add_argument("--offset_adapter_select", action="store_true", default=False,
                        help="If enabled, select the offset adapter only when calibration BCE improves enough. Default is report-only.")
    parser.add_argument("--no_offset_adapter_select", action="store_false", dest="offset_adapter_select")
    parser.add_argument("--offset_adapter_bce_margin", type=float, default=0.000,
                        help="Required calibration BCE improvement before selecting the offset adapter.")

    # Prototype ranking offset adapter.
    # This uses a pairwise ranking/AUC surrogate on the target calibration split.
    parser.add_argument("--use_rank_offset_adapter", action="store_true", default=True,
                        help="Fit a tiny prototype offset adapter with pairwise ranking loss on target calibration labels.")
    parser.add_argument("--no_rank_offset_adapter", action="store_false", dest="use_rank_offset_adapter",
                        help="Disable the pairwise ranking prototype offset adapter.")
    parser.add_argument("--rank_offset_adapter_l2", type=float, default=1.0,
                        help="L2 regularization for the ranking prototype offset adapter.")
    parser.add_argument("--rank_offset_adapter_lr", type=float, default=0.03,
                        help="Learning rate for the ranking prototype offset adapter.")
    parser.add_argument("--rank_offset_adapter_epochs", type=int, default=50,
                        help="Training epochs for the ranking prototype offset adapter.")
    parser.add_argument("--rank_offset_adapter_max_abs_delta", type=float, default=0.50,
                        help="Clamp each ranking offset to [-value, value].")
    parser.add_argument("--rank_offset_adapter_temperature", type=float, default=1.0,
                        help="Temperature in the pairwise ranking loss. Smaller values make ranking loss sharper.")
    parser.add_argument("--rank_offset_adapter_select", action="store_true", default=True,
                        help="If enabled, select ranking offset only when calibration AUC improves enough. Default is report-only.")
    parser.add_argument("--no_rank_offset_adapter_select", action="store_false", dest="rank_offset_adapter_select")
    parser.add_argument("--rank_offset_adapter_auc_margin", type=float, default=0.005,
                        help="Required calibration AUC improvement before selecting ranking offset adapter.")

    # Fusion and oracle
    parser.add_argument("--fusion_step", type=float, default=0.05)
    parser.add_argument("--oracle_tune", action="store_true", default=False,
                        help="If enabled, choose fusion weights on target calibration labels and evaluate on held-out target test.")
    parser.add_argument("--no_oracle_tune", action="store_false", dest="oracle_tune")
    # Safe routing fallback
    parser.add_argument(
        "--route_alpha",
        type=float,
        default=0.5,
        help="Weight of direction-aware routed prediction when blending with global model."
    )
    parser.add_argument(
        "--routing_conf_threshold",
        type=float,
        default=0.15,
        help="If routing confidence is lower than this value, fallback to global model."
    )

    # Evaluation
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=421)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using model_type: {args.model_type}")
    if args.model_type == "mlp":
        print(f"MLP config: hidden_dims={args.hidden_dims}, epochs={args.epochs}, dropout={args.dropout}, lr={args.lr}")
        print(f"SupCon={args.supcon_weight}, SWA={args.use_swa}, BN={args.use_bn}, AdaBN={args.adapt_bn}")
        print(
            f"In-house masked AE pretrain={args.use_inhouse_ae_pretrain}, "
            f"dataset={args.pretrain_dataset}, transfer={args.ae_transfer_mode}, "
            f"ae_epochs={args.ae_pretrain_epochs}, ae_lr={args.ae_pretrain_lr}, "
            f"mask_ratio={args.ae_mask_ratio}, noise_std={args.ae_noise_std}, "
            f"exclude_pretrain_from_sources={args.exclude_pretrain_from_sources}"
        )
    print(
        f"Target direction calibration: ratio={args.target_calib_ratio}, "
        f"beta_grid={args.direction_beta_grid}, legacy_beta={args.direction_beta}, "
        f"select_margin={args.direction_select_margin}, T={args.direction_routing_temperature}, "
        f"kappa={args.direction_kappa}, bias_mode={args.direction_bias_mode}, "
        f"logit_flip=False"
    )
    print(
        f"Offset adapter: enabled={args.use_offset_adapter}, select={args.offset_adapter_select}, "
        f"l2={args.offset_adapter_l2}, lr={args.offset_adapter_lr}, "
        f"epochs={args.offset_adapter_epochs}, max_abs_delta={args.offset_adapter_max_abs_delta}, "
        f"bce_margin={args.offset_adapter_bce_margin}"
    )
    print(
        f"Ranking offset adapter: enabled={args.use_rank_offset_adapter}, select={args.rank_offset_adapter_select}, "
        f"l2={args.rank_offset_adapter_l2}, lr={args.rank_offset_adapter_lr}, "
        f"epochs={args.rank_offset_adapter_epochs}, max_abs_delta={args.rank_offset_adapter_max_abs_delta}, "
        f"temperature={args.rank_offset_adapter_temperature}, auc_margin={args.rank_offset_adapter_auc_margin}"
    )

    with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.oracle_tune:
        print("WARNING: oracle_tune=True, choosing fusion weights on target calibration labels and evaluating on held-out test.")

    group_mapping = parse_group_mapping(args.group_mapping)
    important_species = load_species_list(args.species_list)
    dataset_pairs = find_dataset_pairs(args.datasets_dir)

    all_data = []
    for prefix, species_path, metadata_path in dataset_pairs:
        try:
            X_raw, y, sample_ids = load_one_dataset(
                species_path, metadata_path, important_species, group_mapping
            )
            if len(np.unique(y)) < 2:
                print(f"WARNING: {prefix} has only one class. Skipping.")
                continue
            all_data.append(CohortData(
                name=prefix, X_raw=X_raw, y=y, sample_ids=sample_ids,
                species_path=species_path, metadata_path=metadata_path
            ))
            print(f"Loaded {prefix}: n={len(y)}, TD={(y==0).sum()}, ASD={(y==1).sum()}")
        except Exception as e:
            print(f"Failed to load {prefix}: {e}")

    if len(all_data) < 2:
        raise RuntimeError("Fewer than two valid datasets are available.")

    pretrain_state_dict, pretrain_dataset_name = fit_inhouse_masked_ae_pretrain(all_data, args)

    summary_rows = []
    for target_idx in range(len(all_data)):
        row = run_one_target_prototype(
            target_idx,
            all_data,
            args,
            pretrain_state_dict=pretrain_state_dict,
            pretrain_dataset_name=pretrain_dataset_name,
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    compact_summary_df = summary_df[
        [
            "target_dataset",
            "original_method",
            "original_auc",
            "original_prauc",
            "oracle_method",
            "oracle_auc",
            "oracle_prauc",
        ]
    ].rename(columns={"target_dataset": "cohort"})

    summary_path = os.path.join(args.output_dir, "prototype_lodo_summary.csv")
    compact_summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("All LODO folds finished.")
    print("=" * 90)
    print(compact_summary_df.to_string(index=False))

    print("\nSummary:")
    print(f"  Mean original-selector AUC: {compact_summary_df['original_auc'].mean():.4f}")
    print(f"  Mean original-selector PRAUC: {compact_summary_df['original_prauc'].mean():.4f}")
    print(f"  Mean oracle TEST-AUC: {compact_summary_df['oracle_auc'].mean():.4f}")
    print(f"  Mean oracle TEST-PRAUC: {compact_summary_df['oracle_prauc'].mean():.4f}")
    print(f"\nCompact summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
