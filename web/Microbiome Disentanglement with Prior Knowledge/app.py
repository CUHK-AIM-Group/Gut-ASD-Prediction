#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microbiome Disentanglement App (Final Version)
- Reads model checkpoint to obtain training species list
- Aligns input abundance to that list (subset + reorder)
- Loads prior CSV and aligns to same species
- Handles mismatched metadata columns gracefully
"""

import os
os.environ["GRADIO_UI_LANG"] = "en"
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== Model Definition (exactly as training) ========================

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return GradientReversal.apply(x, self.lambd)

def mlp(in_dim, hidden_dims, out_dim, dropout=0.0, act=nn.LeakyReLU, bn=True, last_activation=False):
    layers = []
    dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(dim, h))
        if bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(act(0.1))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    if last_activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class IntrinsicEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128,
                 hidden_dims: List[int] = [512, 256], dropout: float = 0.3):
        super().__init__()
        self.encoder = mlp(input_dim, hidden_dims, latent_dim, dropout=dropout, last_activation=True)
    def forward(self, x):
        return self.encoder(x)

class IntrinsicDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: List[int] = [256, 512], dropout: float = 0.3):
        super().__init__()
        self.decoder = mlp(latent_dim, hidden_dims, output_dim, dropout=dropout, last_activation=False)
    def forward(self, z):
        return self.decoder(z)

class PriorAttentionLayer(nn.Module):
    def __init__(self, num_features: int, prior_weight: torch.Tensor = None,
                 beta: float = 0.7, hidden: int = 64):
        super().__init__()
        self.num_features = num_features
        self.beta = beta
        device = prior_weight.device if prior_weight is not None else torch.device('cpu')
        self.register_buffer('prior_weight', prior_weight if prior_weight is not None else torch.zeros(num_features, device=device))
        self.gate_net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_features),
            nn.Sigmoid()
        )
        self.register_parameter('gate_bias', nn.Parameter(torch.zeros(num_features)))
    def forward(self, delta_raw: torch.Tensor, x_base: torch.Tensor = None):
        with torch.no_grad():
            delta_mag = delta_raw.abs().mean(dim=0)
        gate_in = delta_mag + (0.8 * self.prior_weight)
        gate = self.gate_net(gate_in.unsqueeze(0)).squeeze(0)
        gate = torch.clamp(gate + torch.sigmoid(self.gate_bias), 0.0, 1.0)
        gate = gate.unsqueeze(0).expand(delta_raw.size(0), -1)
        delta = delta_raw * (1.0 + self.beta * gate)
        return delta, gate

class MultiHeadPriorAttention(nn.Module):
    def __init__(self, num_features: int, num_heads: int = 3, prior_info: dict = None, beta: float = 0.7):
        super().__init__()
        self.num_heads = num_heads
        self.num_features = num_features
        prior_weight = prior_info.get('prior_weight') if prior_info else None
        self.heads = nn.ModuleList([
            PriorAttentionLayer(num_features, prior_weight, beta=beta) for _ in range(num_heads)
        ])
        self.head_logits = nn.Parameter(torch.zeros(num_heads))
    def forward(self, delta_raw, x_base=None):
        head_outs, head_gates = [], []
        for h in self.heads:
            d, g = h(delta_raw, x_base)
            head_outs.append(d)
            head_gates.append(g)
        w = torch.softmax(self.head_logits, dim=0)
        stacked = torch.stack(head_outs, dim=0)
        merged = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        gates = torch.stack(head_gates, dim=0)
        avg_gate = (gates * w.view(-1, 1, 1)).sum(dim=0)
        return merged, avg_gate

class DiseaseShiftEncoder(nn.Module):
    def __init__(self, disease_dim: int, output_dim: int, latent_dim: int = 32,
                 hidden_dims: List[int] = [64,32], dropout: float = 0.3,
                 prior_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = mlp(disease_dim, hidden_dims, latent_dim, dropout=dropout, last_activation=True)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, output_dim),
        )
        device = prior_weight.device if prior_weight is not None else torch.device('cpu')
        self.register_buffer('prior_weight', prior_weight if prior_weight is not None else torch.zeros(output_dim, device=device))
        self.alpha_prior = nn.Parameter(torch.full((output_dim,), 0.05))
    def forward(self, c_d):
        h = self.encoder(c_d)
        delta_raw = self.head(h)
        delta_raw = delta_raw + (self.alpha_prior * self.prior_weight)
        return delta_raw

class OtherMetadataShiftEncoder(nn.Module):
    def __init__(self, others_dim: int, output_dim: int,
                 latent_dim: int = 64, hidden_dims: List[int] = [256,128], dropout: float = 0.4):
        super().__init__()
        self.output_dim = output_dim
        if others_dim > 0:
            self.encoder = mlp(others_dim, hidden_dims, latent_dim, dropout=dropout, last_activation=True)
            self.head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout * 0.3),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, output_dim)
            )
        else:
            self.encoder = None
            self.head = None
    def forward(self, c_o):
        if self.encoder is None or c_o.shape[1] == 0:
            return torch.zeros(c_o.shape[0], self.output_dim, device=c_o.device)
        return self.head(self.encoder(c_o))

class UnknownFactorEncoder(nn.Module):
    def __init__(self, x_dim: int, u_dim: int = 64,
                 hidden_dims: List[int] = [256,128], dropout: float = 0.4):
        super().__init__()
        self.u_dim = u_dim
        self.encoder = mlp(x_dim, hidden_dims, u_dim, dropout=dropout, last_activation=True)
        self.delta_predictor = nn.Sequential(
            nn.Linear(u_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, x_dim)
        )
        nn.init.normal_(self.delta_predictor[-1].weight, mean=0, std=0.01)
    def forward(self, residual):
        u = self.encoder(residual)
        delta_u = self.delta_predictor(u)
        return u, delta_u

class ImprovedCompositeModelWithAttention(nn.Module):
    def __init__(self, x_dim, c_dim, disease_dim, others_dim, z_dim=128, u_dim=32,
                 enc_hidden_dims=[1024,512], dec_hidden_dims=[512,1024], dropout=0.3,
                 shift_dropout=0.4, grl_lambda=1.0, adversary=None,
                 prior_info=None, use_attention=True, attention_heads=5, attn_beta=0.7):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.use_attention = use_attention

        self.intrinsic_encoder = IntrinsicEncoder(x_dim, z_dim, enc_hidden_dims, dropout)
        self.intrinsic_decoder = IntrinsicDecoder(z_dim, x_dim, dec_hidden_dims, dropout)

        prior_weight = prior_info.get('prior_weight') if prior_info else None
        self.disease_shift_encoder = DiseaseShiftEncoder(
            disease_dim, x_dim, latent_dim=16, hidden_dims=[64,32], dropout=shift_dropout*0.8,
            prior_weight=prior_weight
        )
        self.other_shift_encoder = OtherMetadataShiftEncoder(others_dim, x_dim, dropout=shift_dropout)
        self.unknown_factor_encoder = UnknownFactorEncoder(x_dim, u_dim, [256,128], dropout)

        if use_attention and prior_info is not None:
            self.prior_attention = MultiHeadPriorAttention(
                x_dim, num_heads=attention_heads, prior_info={'prior_weight': prior_weight}, beta=attn_beta
            )
        else:
            self.prior_attention = None

        self.grl = GRL(grl_lambda)
        self.adversary = adversary
        self.register_buffer('prior_weight', prior_weight if prior_weight is not None else torch.zeros(x_dim))
        # Prior masks (not strictly needed for inference)
        self.register_buffer('prior_mask_any', torch.zeros(x_dim))

    def forward(self, x, c, disease_start, disease_dim, training_stage="full", apply_attention=True):
        c_d = c[:, disease_start:disease_start+disease_dim]
        c_o = torch.cat([c[:, :disease_start], c[:, disease_start+disease_dim:]], dim=1) if c.shape[1] > disease_dim else c.new_zeros(c.size(0), 0)

        z_base = self.intrinsic_encoder(x)
        x_base = self.intrinsic_decoder(z_base)
        delta_d_raw = self.disease_shift_encoder(c_d)

        attn_gate = None
        if self.use_attention and self.prior_attention is not None and apply_attention:
            delta_d, attn_gate = self.prior_attention(delta_d_raw, x_base)
        else:
            delta_d = delta_d_raw

        delta_o = self.other_shift_encoder(c_o)
        x_base = torch.abs(x_base)

        if training_stage == "known_only":
            x_hat = x_base + delta_d + delta_o
            return {"x_hat": x_hat, "x_base": x_base, "delta_d": delta_d, "delta_o": delta_o,
                    "z_base": z_base, "delta_u": None, "attention_weights": attn_gate}
        elif training_stage == "add_unknown":
            residual = x - (x_base + delta_d + delta_o)
            u, delta_u = self.unknown_factor_encoder(residual)
            x_hat = x_base + delta_d + delta_o + delta_u
            return {"x_hat": x_hat, "x_base": x_base, "delta_d": delta_d, "delta_o": delta_o,
                    "z_base": z_base, "delta_u": delta_u, "u": u, "attention_weights": attn_gate}
        else:
            residual = x - (x_base + delta_d + delta_o)
            u, delta_u = self.unknown_factor_encoder(residual)
            x_hat = x_base + delta_d + delta_o + delta_u
            return {"x_hat": x_hat, "x_base": x_base, "delta_d": delta_d, "delta_o": delta_o,
                    "z_base": z_base, "delta_u": delta_u, "u": u, "attention_weights": attn_gate}


def clr_transform(x: np.ndarray) -> np.ndarray:
    return np.arcsinh(x).astype(np.float32)


def build_c_matrix_from_metadata(
    meta: pd.DataFrame,
    disease_col_name: str,
    disease_onehot: np.ndarray,
    c_names: List[str],
    cont_vars: List[str],
    cont_means: Dict[str, float],
    cont_stds: Dict[str, float],
    disc_numeric_vars: List[str],
    disc_numeric_means: Dict[str, float],
    disc_numeric_stds: Dict[str, float],
    categorical_vars: List[str],
    categorical_cats: Dict[str, List[str]],
    missing_suffix: str = "_missing",
) -> np.ndarray:

    n_samples = len(meta)
    blocks = []

    blocks.append(disease_onehot[:, 0].reshape(-1, 1).astype(np.float32))
    blocks.append(disease_onehot[:, 1].reshape(-1, 1).astype(np.float32))

    for col_name in c_names[2:]:
        if col_name.endswith(missing_suffix):
            orig_col = col_name[:-len(missing_suffix)]
            if orig_col not in cont_vars:
                raise ValueError(f"Missing indicator {col_name} corresponds to unknown continuous var {orig_col}")
            missing = meta[orig_col].isna().astype(np.float32).values
            blocks.append(missing.reshape(-1, 1))

        elif col_name in cont_vars:
            values = pd.to_numeric(meta[col_name], errors='coerce').values.astype(np.float32)
            mean = cont_means[col_name]
            std = cont_stds[col_name]
            values = np.nan_to_num(values, nan=mean)
            normalized = (values - mean) / std
            blocks.append(normalized.reshape(-1, 1))

        elif col_name in disc_numeric_vars:
            values = pd.to_numeric(meta[col_name], errors='coerce').values.astype(np.float32)
            mean = disc_numeric_means[col_name]
            std = disc_numeric_stds[col_name]
            values = np.nan_to_num(values, nan=mean)
            normalized = (values - mean) / std
            blocks.append(normalized.reshape(-1, 1))

        else:
            found = False
            for cat_var in categorical_vars:
                prefix = cat_var + "="
                if col_name.startswith(prefix):
                    category_value = col_name[len(prefix):]
                    allowed_cats = categorical_cats.get(cat_var, [])
                    if category_value not in allowed_cats:
                        raise ValueError(f"Category {category_value} not in pre-defined cats for {cat_var}")
                    series = meta[cat_var].astype(str)
                    indicator = (series == category_value).astype(np.float32).values
                    blocks.append(indicator.reshape(-1, 1))
                    found = True
                    break
            if not found:
                raise ValueError(f"Column {col_name} cannot be matched to any known covariate type.")

    C = np.concatenate(blocks, axis=1)
    if C.shape[1] != len(c_names):
        raise ValueError(f"Constructed C has {C.shape[1]} columns, expected {len(c_names)}")
    return C


def build_inference_dataloader(
    abundance_csv: str,
    meta_csv: str,
    disease_col_name: str,
    checkpoint_info: Dict,
    target_species_order: List[str],
    batch_size: int = 32,
) -> Tuple[torch.utils.data.DataLoader, int, int, int, int, List[str], List[str]]:

    abund_raw = pd.read_csv(abundance_csv, index_col=0)
    abund = abund_raw.T  # samples x species

    meta = pd.read_csv(meta_csv, index_col='ID')
    meta.index = meta.index.astype(str).str.strip()
    abund.index = abund.index.astype(str).str.strip()

    common_ids = abund.index.intersection(meta.index)
    if len(common_ids) == 0:
        abund.index = abund.index.astype(str)
        meta.index = meta.index.astype(str)
        common_ids = abund.index.intersection(meta.index)
    if len(common_ids) == 0:
        raise ValueError(f"No common sample IDs. Abundance: {abund.index[:5].tolist()}, Metadata: {meta.index[:5].tolist()}")
    abund = abund.loc[common_ids]
    meta = meta.loc[common_ids]


    if disease_col_name not in meta.columns:
        raise ValueError(f"Disease column '{disease_col_name}' not found in metadata")
    disease_series = meta[disease_col_name]
    missing_mask = disease_series.isna()
    if missing_mask.any():
        keep = ~missing_mask
        abund = abund.loc[keep]
        meta = meta.loc[keep]
        disease_series = disease_series.loc[keep]
    uniq = disease_series.unique()
    if set(uniq) == {1, 2}:
        disease_series_encoded = disease_series.replace({2: 0})
    else:
        disease_series_encoded = disease_series
    disease_onehot = np.zeros((len(disease_series_encoded), 2), dtype=np.float32)
    for i, v in enumerate(disease_series_encoded.values):
        if v == 0:
            disease_onehot[i, 0] = 1.0
        elif v == 1:
            disease_onehot[i, 1] = 1.0
    c_names = checkpoint_info.get('c_names')
    if c_names is None:
        raise ValueError("Checkpoint missing 'c_names' which is required for exact C reconstruction.")
    cont_vars = checkpoint_info.get('cont_vars', [])
    cont_means = checkpoint_info.get('cont_means', {})
    cont_stds = checkpoint_info.get('cont_stds', {})
    disc_numeric_vars = checkpoint_info.get('disc_numeric_vars', [])
    disc_numeric_means = checkpoint_info.get('disc_numeric_means', {})
    disc_numeric_stds = checkpoint_info.get('disc_numeric_stds', {})
    categorical_vars = checkpoint_info.get('categorical_vars', [])
    categorical_cats = checkpoint_info.get('categorical_cats', {})
    missing_suffix = checkpoint_info.get('missing_suffix', '_missing')

    C_full = build_c_matrix_from_metadata(
        meta=meta,
        disease_col_name=disease_col_name,
        disease_onehot=disease_onehot,
        c_names=c_names,
        cont_vars=cont_vars,
        cont_means=cont_means,
        cont_stds=cont_stds,
        disc_numeric_vars=disc_numeric_vars,
        disc_numeric_means=disc_numeric_means,
        disc_numeric_stds=disc_numeric_stds,
        categorical_vars=categorical_vars,
        categorical_cats=categorical_cats,
        missing_suffix=missing_suffix,
    )

    if target_species_order is None:
        target_species_order = checkpoint_info.get('selected_species')
        if target_species_order is None:
            raise ValueError("No target species order provided and checkpoint missing 'selected_species'")
    missing_species = [sp for sp in target_species_order if sp not in abund.columns]
    if missing_species:
        raise ValueError(f"Input abundance missing required species: {missing_species[:10]}... (total {len(missing_species)})")
    abund = abund[target_species_order]

    X_clr = clr_transform(abund.values.astype(np.float32))
    sample_ids = abund.index.tolist()
    species_names = abund.columns.tolist()

    X_tensor = torch.from_numpy(X_clr).float()
    C_tensor = torch.from_numpy(C_full).float()
    dataset = torch.utils.data.TensorDataset(X_tensor, C_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    disease_start = 0
    disease_dim = 2
    x_dim = X_clr.shape[1]
    c_dim = C_full.shape[1]
    return loader, x_dim, c_dim, disease_start, disease_dim, sample_ids, species_names


def load_prior_csv(csv_path: str, species_list: List[str]) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    col_map = {c.lower(): c for c in df.columns}
    sp_col = col_map.get("species") or col_map.get("taxon") or "species"
    dir_col = col_map.get("direction_score") or "direction_score"
    conf_col = col_map.get("aggregated_confidence") or "aggregated_confidence"
    if sp_col not in df.columns or dir_col not in df.columns or conf_col not in df.columns:
        raise ValueError(f"Prior CSV must contain columns: species, direction_score, aggregated_confidence. Found: {df.columns.tolist()}")
    X = len(species_list)
    prior_weight = np.zeros(X, dtype=np.float32)
    pos_mask = np.zeros(X, dtype=np.float32)
    neg_mask = np.zeros(X, dtype=np.float32)
    neu_mask = np.zeros(X, dtype=np.float32)
    sp2idx = {sp: i for i, sp in enumerate(species_list)}
    found = 0
    for _, row in df.iterrows():
        sp = str(row[sp_col]).strip()
        if sp in sp2idx:
            idx = sp2idx[sp]
            direction = float(row[dir_col])
            conf = float(row[conf_col])
            conf = max(0.0, min(1.0, conf))
            prior_weight[idx] = conf
            if direction > 0:
                pos_mask[idx] = conf
            elif direction < 0:
                neg_mask[idx] = conf
            else:
                neu_mask[idx] = conf
            found += 1
    print(f"Prior matched {found}/{X} species.")
    return {
        "prior_weight": prior_weight,
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
        "neu_mask": neu_mask
    }


def load_model_and_weights(
    checkpoint_path: Path,
    target_species_list: List[str],
    c_dim: int,
    disease_dim: int,
    use_attention: bool,
    device: torch.device,
    prior_dict: Dict,
) -> nn.Module:
    x_dim = len(target_species_list)
    others_dim = c_dim - disease_dim
    prior_info = {
        'prior_weight': torch.from_numpy(prior_dict['prior_weight']).float(),
        'pos_mask': prior_dict['pos_mask'],
        'neg_mask': prior_dict['neg_mask'],
        'neu_mask': prior_dict['neu_mask'],
    }
    model = ImprovedCompositeModelWithAttention(
        x_dim=x_dim, c_dim=c_dim, disease_dim=disease_dim, others_dim=others_dim,
        z_dim=512, u_dim=32, enc_hidden_dims=[1024,512], dec_hidden_dims=[512,1024],
        dropout=0.3, shift_dropout=0.4, grl_lambda=0.1, adversary=None,
        prior_info=prior_info, use_attention=use_attention, attention_heads=5, attn_beta=0.7
    )
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state']
    keys_to_remove = [k for k in state_dict if 'prior_weight' in k or 'prior_mask' in k or 'pos_mask' in k or 'neg_mask' in k or 'neu_mask' in k]
    for k in keys_to_remove:
        del state_dict[k]
    model_state = model.state_dict()
    for k in list(state_dict.keys()):
        if k in model_state:
            if state_dict[k].shape != model_state[k].shape:
                print(f"Removing mismatched key {k}: checkpoint {state_dict[k].shape} vs model {model_state[k].shape}")
                del state_dict[k]
        else:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model


MODELS_DIR = Path("./models")
model_files = [f.name for f in MODELS_DIR.glob("*.pt")] if MODELS_DIR.exists() else []
default_model = model_files[0] if model_files else None

METADATA_TXT_PATH = MODELS_DIR / "metadata_select.txt"
if METADATA_TXT_PATH.exists():
    with open(METADATA_TXT_PATH, "r") as f:
        WANTED_COLS = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    WANTED_COLS = []  


def run_inference(abundance_file, metadata_file, prior_csv_file, disease_col_name, model_filename, batch_size, use_attention):
    if any(x is None for x in [abundance_file, metadata_file, prior_csv_file]):
        return "❌ Please upload all required files.", None
    if not model_filename:
        return "❌ No model selected.", None
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        return f"❌ Model file not found: {model_path}", None

    if not METADATA_TXT_PATH.exists():
        return f"❌ Required file '{METADATA_TXT_PATH}' not found. Please ensure the file exists in the models directory.", None

    abund_path = abundance_file
    meta_path = metadata_file
    prior_path = prior_csv_file

    checkpoint = torch.load(model_path, map_location='cpu')
    if 'info' not in checkpoint:
        return "❌ Checkpoint missing 'info' dictionary. Cannot reconstruct preprocessing.", None
    info = checkpoint['info']
    required_keys = ['selected_species', 'c_names']
    for k in required_keys:
        if k not in info:
            return f"❌ Checkpoint info missing '{k}'. Please retrain with updated training script.", None

    target_species = info['selected_species']
    print(f"Model expects {len(target_species)} species. Aligning input data...")

    try:
        loader, x_dim, c_dim, disease_start, disease_dim, sample_ids, species_names = build_inference_dataloader(
            abund_path, meta_path, disease_col_name, info, target_species, batch_size
        )
    except Exception as e:
        return f"❌ Data loading with exact C reconstruction failed: {str(e)}\n{traceback.format_exc()}", None

    try:
        prior_dict = load_prior_csv(prior_path, target_species)
    except Exception as e:
        return f"❌ Prior CSV loading failed: {str(e)}", None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_model_and_weights(
            model_path, target_species, c_dim, disease_dim, use_attention, device, prior_dict
        )
        model.eval()
    except Exception as e:
        return f"❌ Model loading failed: {str(e)}\n{traceback.format_exc()}", None

    all_delta_d, all_delta_o, all_x_base = [], [], []
    with torch.no_grad():
        for x, c in loader:
            x = x.to(device).float()
            c = c.to(device).float()
            out = model(x, c, disease_start, disease_dim, training_stage="full", apply_attention=use_attention)
            all_delta_d.append(out["delta_d"].cpu().numpy())
            all_delta_o.append(out["delta_o"].cpu().numpy())
            all_x_base.append(out["x_base"].cpu().numpy())
    delta_d = np.concatenate(all_delta_d, axis=0)
    delta_o = np.concatenate(all_delta_o, axis=0)
    x_base = np.concatenate(all_x_base, axis=0)


    def save_with_name(df, filename):
        tmp_dir = tempfile.mkdtemp(prefix="disentangle_")
        file_path = os.path.join(tmp_dir, filename)
        df.to_csv(file_path, index=True)
        return file_path

    df_delta_d = pd.DataFrame(delta_d, index=sample_ids, columns=target_species)
    df_delta_o = pd.DataFrame(delta_o, index=sample_ids, columns=target_species)
    df_x_base = pd.DataFrame(x_base, index=sample_ids, columns=target_species)

    delta_d_path = save_with_name(df_delta_d, "ΔH.csv")
    delta_o_path = save_with_name(df_delta_o, "ΔC.csv")
    x_base_path = save_with_name(df_x_base, "base.csv")

    summary = f"✅ Inference completed!\nSamples: {len(sample_ids)}\nSpecies: {len(target_species)}\nOutput files: ΔH.csv, ΔC.csv, base.csv"
    return summary, [delta_d_path, delta_o_path, x_base_path]


# ======================== Gradio UI ========================

CUSTOM_CSS = """
:root {
    --bg: #070707;
    --panel: #111111;
    --panel-soft: #171717;
    --input: #1d1d1d;
    --input-hover: #242424;
    --border: #333333;
    --border-soft: #262626;
    --text: #f6f6f6;
    --muted: #b9b9b9;
    --muted2: #8f8f8f;
    --accent: #6d5ef5;
    --accent2: #8b5cf6;
    --success: #22c55e;
}

html, body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    padding: 30px !important;
}

/* Make all default Gradio text readable */
.gradio-container,
.gradio-container * {
    color: var(--text) !important;
}

/* Markdown */
.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .markdown,
.gradio-container .markdown *,
.gradio-container [data-testid="markdown"],
.gradio-container [data-testid="markdown"] * {
    color: var(--text) !important;
}

.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose span {
    color: #d7d7d7 !important;
}

.gradio-container .prose strong,
.gradio-container .prose b {
    color: #ffffff !important;
}

/* Inline code */
code,
.prose code,
.markdown code,
#summary-output code,
.info-card code {
    background: #292929 !important;
    color: #ffffff !important;
    border: 1px solid #434343 !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
}

pre,
.prose pre {
    background: #171717 !important;
    color: #f5f5f5 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
}

pre code,
.prose pre code {
    background: transparent !important;
    border: none !important;
}

/* Header */
#title-box {
    background: linear-gradient(135deg, #111111 0%, #171717 55%, #1b1b1b 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 34px 30px;
    margin-bottom: 26px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.38);
}

#title-box h1 {
    color: #ffffff !important;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 12px 0;
}

#title-box p {
    color: var(--muted) !important;
    font-size: 16px;
    line-height: 1.65;
    margin: 0;
}

/* Cards */
.input-card,
.output-card,
.info-card {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 22px !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.35);
}

.input-card h3,
.output-card h3,
.info-card h3 {
    color: #ffffff !important;
    font-size: 18px !important;
    margin: 0 0 14px 0 !important;
}

/* Remove white wrappers */
.gr-box,
.gr-form,
.gr-group,
.block,
.form,
.panel,
fieldset {
    background: transparent !important;
    border-color: var(--border-soft) !important;
    box-shadow: none !important;
}

/* Labels */
label,
.block-label,
.gradio-container label span,
.gradio-container .block-label span {
    color: #f2f2f2 !important;
    font-weight: 650 !important;
    background: transparent !important;
}

/* Descriptions */
.info,
.description,
.gradio-container small,
.gradio-container [class*="description"],
.gradio-container [class*="info"] {
    color: var(--muted) !important;
}

/* Inputs */
input,
textarea,
select,
.gradio-container [role="combobox"] {
    background: var(--input) !important;
    color: #ffffff !important;
    border: 1px solid #3b3b3b !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

input:focus,
textarea:focus,
select:focus,
.gradio-container [role="combobox"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

input::placeholder,
textarea::placeholder {
    color: #8a8a8a !important;
}

/* Dropdown */
.gradio-container option,
.gradio-container [role="listbox"],
.gradio-container [role="option"],
.gradio-container [class*="dropdown"] {
    background: var(--input) !important;
    color: #ffffff !important;
    border-color: #444444 !important;
}

.gradio-container [role="listbox"] *,
.gradio-container [role="option"] *,
.gradio-container [class*="dropdown"] * {
    color: #ffffff !important;
}

/* Compact file selectors based on UploadButton */
.file-row {
    background: var(--panel-soft) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    margin-bottom: 14px !important;
}

.file-row h4 {
    margin: 0 0 6px 0 !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

.file-row p {
    margin: 0 0 12px 0 !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
}

.upload-btn,
.upload-btn button {
    width: 100% !important;
}

.upload-btn button,
.gradio-container button.secondary {
    background: #202020 !important;
    color: #ffffff !important;
    border: 1px solid #3e3e3e !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

.upload-btn button:hover,
.gradio-container button.secondary:hover {
    background: #272727 !important;
    border-color: #5a5a5a !important;
}

/* Slider */
.gradio-container input[type="range"] {
    accent-color: var(--accent) !important;
}

.gradio-container .noUi-target {
    background: #2a2a2a !important;
    border-color: #444444 !important;
}

.gradio-container .noUi-connect {
    background: var(--accent) !important;
}

.gradio-container .noUi-handle {
    background: #ffffff !important;
    border: 1px solid #ffffff !important;
}

.gradio-container .noUi-value,
.gradio-container .noUi-tooltip,
.gradio-container [class*="slider"] span,
.gradio-container [class*="slider"] label {
    color: #ffffff !important;
}

/* Checkbox */
.gradio-container input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

.gradio-container input[type="checkbox"] + span,
.gradio-container [class*="checkbox"] span,
.gradio-container [class*="checkbox"] label {
    color: #ffffff !important;
}

/* Primary button */
.gradio-container button.primary,
.gradio-container .gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    padding: 14px 22px !important;
    box-shadow: 0 10px 24px rgba(109, 94, 245, 0.28) !important;
}

.gradio-container button.primary:hover,
.gradio-container .gr-button-primary:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px);
}

/* Output */
#summary-output textarea,
#summary-output input {
    background: #171717 !important;
    color: #ffffff !important;
    border: 1px solid #3b3b3b !important;
}

#download-output,
#download-output * {
    background: #171717 !important;
    color: #ffffff !important;
    border-color: #3b3b3b !important;
}

/* Tables and links */
table {
    background: #111111 !important;
    color: #ffffff !important;
}

th {
    background: #222222 !important;
    color: #ffffff !important;
    border: 1px solid #444444 !important;
}

td {
    background: #141414 !important;
    color: #eeeeee !important;
    border: 1px solid #333333 !important;
}

a {
    color: #8ab4ff !important;
}

hr {
    border-color: #303030 !important;
}

footer {
    display: none !important;
}
"""


with gr.Blocks(title="Microbiome Disentanglement") as demo:
    gr.Markdown("""
    <div id="title-box">
        <h1>Microbiome Disentanglement with Prior Knowledge</h1>
        <p>
        Upload microbiome abundance, metadata, and prior knowledge files, then run a pre-trained
        prior-guided disentanglement model to decompose microbial profiles into disease-associated,
        covariate-associated, and intrinsic components.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="input-card"):
            gr.Markdown("### Input Files")

            gr.Markdown("""
            <div class="file-row">
                <h4>Abundance CSV</h4>
                <p>Rows should be microbial species and columns should be samples.</p>
            </div>
            """)
            abundance_file = gr.UploadButton(
                "Choose Abundance CSV",
                type="filepath",
                file_count="single",
                file_types=[".csv"],
                variant="secondary",
                elem_classes="upload-btn"
            )

            gr.Markdown("""
            <div class="file-row">
                <h4>Metadata CSV</h4>
                <p>Must contain an ID column and the selected disease column.</p>
            </div>
            """)
            metadata_file = gr.UploadButton(
                "Choose Metadata CSV",
                type="filepath",
                file_count="single",
                file_types=[".csv"],
                variant="secondary",
                elem_classes="upload-btn"
            )

            gr.Markdown("""
            <div class="file-row">
                <h4>Prior Knowledge CSV</h4>
                <p>Required columns: species, direction_score, aggregated_confidence.</p>
            </div>
            """)
            prior_csv_file = gr.UploadButton(
                "Choose Prior CSV",
                type="filepath",
                file_count="single",
                file_types=[".csv"],
                variant="secondary",
                elem_classes="upload-btn"
            )

        with gr.Column(scale=1, elem_classes="input-card"):
            gr.Markdown("### Model Settings")

            disease_col = gr.Textbox(
                label="Disease column name",
                value="group_x",
                placeholder="e.g., group_x"
            )

            model_selector = gr.Dropdown(
                choices=model_files,
                label="Model checkpoint (.pt)",
                value=default_model,
                interactive=True
            )

            batch_size = gr.Slider(
                label="Batch size",
                minimum=1,
                maximum=256,
                value=32,
                step=1
            )

            use_attention = gr.Checkbox(
                label="Use prior attention",
                value=True
            )

            run_btn = gr.Button(
                "Run Disentanglement",
                variant="primary"
            )

    with gr.Row():
        with gr.Column(scale=1, elem_classes="output-card"):
            gr.Markdown("### Summary")
            output_text = gr.Textbox(
                label="Inference status",
                lines=12,
                elem_id="summary-output"
            )

        with gr.Column(scale=1, elem_classes="output-card"):
            gr.Markdown("### Download Results")
            output_files = gr.File(
                label="Generated CSV files",
                file_count="multiple",
                elem_id="download-output"
            )

    gr.Markdown("""
    <div class="info-card">
        <h3>Instructions</h3>
        <ol>
            <li><b>Abundance CSV:</b> species as rows and samples as columns.</li>
            <li><b>Metadata CSV:</b> must contain an <code>ID</code> column and the selected disease column.</li>
            <li><b>Prior CSV:</b> must contain <code>species</code>, <code>direction_score</code>, and <code>aggregated_confidence</code>.</li>
            <li><b>Model checkpoint:</b> select one of the available <code>.pt</code> files in the models directory.</li>
            <li><b>Outputs:</b> the app generates disease-associated shift <code>ΔH.csv</code>, covariate-associated shift <code>ΔC.csv</code>, and intrinsic baseline <code>base.csv</code>.</li>
        </ol>
    </div>
    """)

    run_btn.click(
        fn=run_inference,
        inputs=[
            abundance_file,
            metadata_file,
            prior_csv_file,
            disease_col,
            model_selector,
            batch_size,
            use_attention
        ],
        outputs=[
            output_text,
            output_files
        ]
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome(), css=CUSTOM_CSS)
