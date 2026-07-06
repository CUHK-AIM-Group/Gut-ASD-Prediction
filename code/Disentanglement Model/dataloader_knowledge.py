import os
import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

CONTINUOUS_COLS_CANON = [
    "ASC_total","ASC_uncertainty","Alc","BMI","CBCL/6-18_SSS_AB","CBCL/6-18_SSS_DB","CBCL_ADHD symptoms",
    "CBCL_AP_T","Caff","Chol","Enrollment Age","F_age","F_edu_years","Height (cm)","Iodine","M_CEBQ_DD",
    "M_CEBQ_EF","M_CEBQ_EOE","M_CEBQ_EUE","M_CEBQ_FF","M_CEBQ_FR","M_CEBQ_SE","M_CEBQ_SR","M_SEQ_hyper",
    "M_SEQ_hypo","M_SEQ_seeking","M_SEQ_social","M_age","M_edu_years","PBC1","PBC2","PBC3","PBC4","PBC5",
    "PBC6","Retinol","T_SRS_AWR","T_SRS_COG","T_SRS_COMM","T_SRS_MOT","TransFat","Vit_B12","Vit_D_mcg",
    "Vit_K","Weight (kg)","beverage","corn","dairy","darkgreen","egg","ferm","firedfood","fish","juice",
    "mushroom","nut","procmeat","raw_SRS_RRB","redmeat","starch","sugarjam","wholegrain"
]
DISCRETE_COLS_CANON = [
    "ASC_Classification","Bristol_stool_chart","CBCL_Externalizing_Classification","CloseRelative_DevPro",
    "Drugs","FAPD","FDD","FNVD","F_occupation","GI","Gender","Health_problem","MR","M_occupation",
    "Mental_disorder","SEQ_total_Classification","SRS_Clinical Category","Sibling_Total","GI ","group_x"
]

def arcsinh_transform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.arcsinh(x).astype(np.float32)

def clr_transform(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_safe = np.maximum(x, 0.0) + eps
    log_x = np.log(x_safe)
    return (log_x - log_x.mean(axis=1, keepdims=True)).astype(np.float32)

def transform_abundance(x: np.ndarray, method: str = "arcsinh", eps: float = 1e-6) -> np.ndarray:
    method = (method or "arcsinh").lower().strip()
    aliases = {
        "asinh": "arcsinh",
        "arcsinh": "arcsinh",
        "clr": "clr",
        "none": "none",
        "raw": "none",
    }
    method = aliases.get(method, method)
    if method == "arcsinh":
        return arcsinh_transform(x)
    if method == "clr":
        return clr_transform(x, eps=eps)
    if method == "none":
        return np.asarray(x, dtype=np.float32)
    raise ValueError("abundance_transform must be one of: 'arcsinh', 'asinh', 'clr', 'none', or 'raw'")

def one_hot_from_series(series: pd.Series, name_override: str = None) -> Tuple[np.ndarray, List[str]]:
    cats = sorted([c for c in series.dropna().unique().tolist()])
    mapping = {c: i for i, c in enumerate(cats)}
    dim = len(cats)
    mat = np.zeros((len(series), dim), dtype=np.float32)
    for idx, v in enumerate(series.values):
        if pd.isna(v):
            continue
        j = mapping[v]
        mat[idx, j] = 1.0
    base_name = name_override if name_override is not None else series.name
    names = [f"{base_name}={c}" for c in cats]
    return mat, names

def read_metadata_list(metadata_txt: str) -> List[str]:
    names = []
    with open(metadata_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s != "" and not s.startswith("#"):
                names.append(s)
    return names

def safe_col_lookup(meta: pd.DataFrame, wanted: List[str]) -> List[str]:
    found = []
    missing = []
    for w in wanted:
        if w in meta.columns:
            found.append(w)
        else:
            matches = [c for c in meta.columns if c.strip() == w.strip()]
            if matches:
                found.append(matches[0])
            else:
                missing.append(w)
    if missing:
        print(f"Warning: the following metadata columns were not found in meta_csv and will be ignored: {missing}")
    return found

def load_species_prior(prior_csv: Optional[str], selected_species: List[str]) -> Dict[str, np.ndarray]:
    """
    Load prior CSV with columns [species, direction_score, aggregated_confidence].
    Returns:
    - prior_weight: Vector aligned to selected_species in [0, 1].
    - pos_mask, neg_mask, neu_mask: Non-negative masks for positive, negative, and neutral directions.
    - prior_idx_pos, prior_idx_neg, prior_idx_neu: Matched index arrays.
    """
    X = len(selected_species)
    prior_weight = np.zeros(X, dtype=np.float32)
    pos_mask = np.zeros(X, dtype=np.float32)
    neg_mask = np.zeros(X, dtype=np.float32)
    neu_mask = np.zeros(X, dtype=np.float32)
    
    prior_idx_pos = []
    prior_idx_neg = []
    prior_idx_neu = []

    if prior_csv is None or not os.path.exists(prior_csv):
        print(f"Prior CSV does not exist or was not specified: {prior_csv}")
        print("Prior guidance will be disabled")
        return {
            "prior_weight": prior_weight, 
            "pos_mask": pos_mask, 
            "neg_mask": neg_mask,
            "neu_mask": neu_mask,
            "prior_idx_pos": np.array(prior_idx_pos, dtype=np.int64),
            "prior_idx_neg": np.array(prior_idx_neg, dtype=np.int64),
            "prior_idx_neu": np.array(prior_idx_neu, dtype=np.int64)
        }

    df = pd.read_csv(prior_csv)
    col_map = {c.lower(): c for c in df.columns}
    sp_col = col_map.get("species") or col_map.get("taxon") or "species"
    dir_col = col_map.get("direction_score") or "direction_score"
    conf_col = col_map.get("aggregated_confidence") or "aggregated_confidence"
    
    if sp_col not in df.columns or dir_col not in df.columns or conf_col not in df.columns:
        print(f"Prior CSV columns do not match. Expected species, direction_score, aggregated_confidence. Current columns: {df.columns.tolist()}")
        print("Ignoring priors.")
        return {
            "prior_weight": prior_weight, 
            "pos_mask": pos_mask, 
            "neg_mask": neg_mask,
            "neu_mask": neu_mask,
            "prior_idx_pos": np.array(prior_idx_pos, dtype=np.int64),
            "prior_idx_neg": np.array(prior_idx_neg, dtype=np.int64),
            "prior_idx_neu": np.array(prior_idx_neu, dtype=np.int64)
        }

    sp2idx = {sp: i for i, sp in enumerate(selected_species)}
    hit = 0
    
    print(f"\nPrior matching diagnostics:")
    print(f"First five selected_species: {selected_species[:5]}")
    print(f"First five species in prior CSV: {df[sp_col].head(5).tolist()}")

    for _, row in df.iterrows():
        sp = str(row[sp_col]).strip()
        if sp in sp2idx:
            i = sp2idx[sp]
            direction = float(row[dir_col])
            conf = float(row[conf_col])
            conf = max(0.0, min(1.0, conf))
            
            prior_weight[i] = conf
            
            if direction > 0:  # positive
                pos_mask[i] = conf
                prior_idx_pos.append(i)
            elif direction < 0:  # negative
                neg_mask[i] = conf
                prior_idx_neg.append(i)
            else:  # direction == 0, neutral
                neu_mask[i] = conf
                prior_idx_neu.append(i)
                
            hit += 1
            if hit <= 5:
                print(f"  Matched: {sp} -> direction={direction}, conf={conf:.3f}")

    if hit == 0:
        print("No overlap between prior CSV and selected_species. Prior guidance will be disabled.")
        csv_species = df[sp_col].head(10).tolist()
        selected_species_short = selected_species[:10]
        print(f"First ten species in prior CSV: {csv_species}")
        print(f"First ten selected_species: {selected_species_short}")
    else:
        print(f"Prior guidance aligned to {hit} species out of{len(selected_species)}")
        print(f"Positive prior species: {(pos_mask > 0).sum()}")
        print(f"Negative prior species: {(neg_mask > 0).sum()}")
        print(f"Neutral prior species: {(neu_mask > 0).sum()}")
        print(f"Prior confidence range: [{prior_weight.min():.3f}, {prior_weight.max():.3f}]")

    return {
        "prior_weight": prior_weight.astype(np.float32),
        "pos_mask": pos_mask.astype(np.float32),
        "neg_mask": neg_mask.astype(np.float32),
        "neu_mask": neu_mask.astype(np.float32),
        "prior_idx_pos": np.array(prior_idx_pos, dtype=np.int64),
        "prior_idx_neg": np.array(prior_idx_neg, dtype=np.int64),
        "prior_idx_neu": np.array(prior_idx_neu, dtype=np.int64),
    }

class MicrobiomeDataset(Dataset):
    def __init__(
        self,
        abundance_csv: str,
        meta_csv: str,
        metadata_txt: str,
        species_prior_csv: str,
        disease_col_name: str = "group_x",
        abundance_transform: str = "arcsinh",
        transform_clr: Optional[bool] = None,
        clr_eps: float = 1e-6,
        subject_intersection_only: bool = True,
        prevalence_threshold: float = 0.1,
        min_nonzero_species: int = 20,
        standardize_continuous: bool = True,
        include_cont_missing_indicator: bool = True,
        numeric_discrete_mode: str = "minmax",
        device: Optional[torch.device] = None,
        save_species_csv: Optional[str] = None,
        remove_missing_disease: bool = True,
    ):
        self.device = device
        self.min_nonzero_species = min_nonzero_species
        if transform_clr is not None:
            abundance_transform = "clr" if transform_clr else "none"
        abundance_transform = (abundance_transform or "arcsinh").lower().strip()

        abund = pd.read_csv(abundance_csv, index_col=0).T
        meta = pd.read_csv(meta_csv, index_col='ID')

        print(f"Loading data summary:")
        print(f"  Abundance data: {abund.shape[0]} samples, {abund.shape[1]} features")
        print(f"  Metadata: {meta.shape[0]} samples, {meta.shape[1]} features")

        abund = abund.fillna(0.0)

        if disease_col_name not in meta.columns:
            matches = [c for c in meta.columns if c.strip() == disease_col_name.strip()]
            if matches:
                disease_col_name = matches[0]
            else:
                raise ValueError(f"Disease column '{disease_col_name}' was not found in metadata")
        
        if subject_intersection_only:
            common_ids = abund.index.intersection(meta.index)
            abund = abund.loc[common_ids]
            meta = meta.loc[common_ids]
            print(f"\nSample intersection: {len(common_ids)}")
        else:
            meta = meta.reindex(abund.index)
            print(f"\nAligned samples: {len(abund)}")

        if remove_missing_disease:
            disease_series = meta[disease_col_name]
            missing_mask = disease_series.isna()
            if missing_mask.any():
                n_missing = missing_mask.sum()
                print(f"\nRemoved samples with missing disease column '{disease_col_name}': {n_missing}")
                keep_mask = ~missing_mask
                meta = meta.loc[keep_mask]
                abund = abund.loc[keep_mask]
                print(f"  Samples remaining after filtering: {len(abund)}")
            else:
                print(f"\nDisease column '{disease_col_name}' has no missing values")
        else:
            print(f"\nNote: samples with missing disease labels were not removed")

        disease_series = meta[disease_col_name]
        counts_before = disease_series.value_counts()
        print("\nOriginal disease label distribution:")
        for value, count in counts_before.items():
            print(f"  label{value}: {count}samples")
        
        mask_keep = disease_series != 3
        n_remove = (~mask_keep).sum()
        if n_remove > 0:
            print(f"\nRemoved samples with label 3: {n_remove}")
            meta = meta[mask_keep]; abund = abund[mask_keep]
            disease_series = meta[disease_col_name]
            print(f"Samples remaining after filtering: {len(abund)}")
        else:
            print(f"\nNo samples with label 3 were found")
        
        disease_series_encoded = disease_series.copy().replace({2: 0})
        counts_after = disease_series_encoded.value_counts()
        print("\nFinal disease label distribution after recoding:")
        for value, count in counts_after.items():
            label_name = "disease" if value == 1 else "control"
            print(f"  {label_name}({value}): {count}samples")

        print(f"\nFiltering species by prevalence threshold={prevalence_threshold:.1%})...")
        n_samples = abund.shape[0]
        prevalence = (abund > 0).sum() / max(1, n_samples)
        selected_species = prevalence[prevalence >= prevalence_threshold].index.tolist()
        if len(selected_species) == 0:
            raise ValueError(f"No species passed the prevalence threshold {prevalence_threshold:.1%}")
        if len(selected_species) < abund.shape[1]:
            print(f"  Kept {len(selected_species)} / {abund.shape[1]} species")

        if save_species_csv:
            pd.DataFrame({'species': selected_species, 'prevalence': prevalence[selected_species].values}).to_csv(save_species_csv, index=False)
            print(f"  Species list saved to: {save_species_csv}")
        abund = abund[selected_species]

        nonzero_counts = (abund > 0).sum(axis=1)
        low_div = nonzero_counts[nonzero_counts < min_nonzero_species]
        if len(low_div) > 0:
            print(f"  Removed samples with nonzero species count < {min_nonzero_species}: {len(low_div)}")
        abund = abund.loc[nonzero_counts >= min_nonzero_species]
        meta = meta.reindex(abund.index)
        disease_series = meta[disease_col_name]
        disease_series_encoded = disease_series.replace({2: 0})
        
        print(f"\nFinal data dimensions:")
        print(f"  Number of samples: {len(abund)}")
        print(f"  Number of species: {len(selected_species)}")
        print(f"  Disease label distribution:")
        for value, count in disease_series_encoded.value_counts().items():
            label_name = "disease" if value == 1 else "control"
            print(f"    {label_name}({value}): {count}samples")

        self.subject_ids = abund.index.astype(str).tolist()
        X_raw = abund.values.astype(np.float32)
        X_clr = transform_abundance(X_raw, method=abundance_transform, eps=clr_eps)
        print(f"\nAbundance transform: {abundance_transform}")
        self.X_raw = X_raw
        self.selected_species = selected_species

        wanted_cols = read_metadata_list(metadata_txt)
        if not wanted_cols:
            raise ValueError("metadata.txt is empty or no field names could be read")
        found_cols = safe_col_lookup(meta, wanted_cols)
        meta_used = meta[found_cols].copy()

        disease_col_cand = [c for c in found_cols if c.strip() == disease_col_name]
        if len(disease_col_cand) == 0:
            raise ValueError(f"metadata_txt does not include disease column {disease_col_name}, Please add it to {metadata_txt} ")
        disease_col = disease_col_cand[0]

        disease_series = meta_used[disease_col]
        disease_series_encoded = disease_series.copy().replace({2: 0})
        
        disease_onehot = np.zeros((len(disease_series_encoded), 2), dtype=np.float32)
        for i, v in enumerate(disease_series_encoded.values):
            if v == 0:
                disease_onehot[i, 0] = 1.0
            elif v == 1:
                disease_onehot[i, 1] = 1.0
        disease_names = [f"{disease_col}=0(control)", f"{disease_col}=1(disease)"]
        
        meta_others = meta_used.drop(columns=[disease_col])

        cont_in_use = [c for c in meta_others.columns if c in CONTINUOUS_COLS_CANON]
        disc_in_use = [c for c in meta_others.columns if c in DISCRETE_COLS_CANON and c != disease_col]
        others = [c for c in meta_others.columns if (c not in cont_in_use and c not in disc_in_use)]

        for c in others:
            s = meta_others[c]
            is_numeric = pd.to_numeric(s, errors="coerce").notna().mean() > 0.95
            if is_numeric:
                cont_in_use.append(c)
            else:
                disc_in_use.append(c)

        cont_blocks, cont_names = [], []
        cont_missing_blocks = []
        cont_scaler = None
        cont_means = {}
        cont_stds = {}

        if len(cont_in_use) > 0:
            cont_scaler = StandardScaler() if standardize_continuous else None
            cont_df = meta_others[cont_in_use].apply(pd.to_numeric, errors="coerce")
            cont_mask = ~cont_df.isna()
            fill_vals = cont_df.mean(axis=0)
            cont_filled = cont_df.fillna(fill_vals)
            if cont_scaler is not None:
                cont_arr = cont_scaler.fit_transform(cont_filled.values).astype(np.float32)
                for i, col in enumerate(cont_in_use):
                    cont_means[col] = cont_scaler.mean_[i]
                    cont_stds[col] = cont_scaler.scale_[i]
            else:
                cont_arr = cont_filled.values.astype(np.float32)
            cont_blocks.append(cont_arr)
            cont_names.extend(cont_in_use)
            if include_cont_missing_indicator:
                miss_ind = (~cont_mask).astype(np.float32).values
                cont_missing_blocks.append(miss_ind.astype(np.float32))

        disc_blocks, disc_names = [], []
        numeric_disc_blocks, numeric_disc_names = [], []
        numeric_disc_scalers = {}
        disc_numeric_means = {}
        disc_numeric_stds = {}
        categorical_vars_list = []
        categorical_cats_dict = {}

        for c in disc_in_use:
            s = meta_others[c]
            s_num = pd.to_numeric(s, errors="coerce")
            frac_num = s_num.notna().mean()
            if c == "Sibling_Total":
                if frac_num > 0.95:
                    arr = s_num.values.reshape(-1, 1).astype(np.float32)
                    if numeric_discrete_mode == "zscore":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    arr = scaler.fit_transform(arr).astype(np.float32)
                    numeric_disc_blocks.append(arr)
                    numeric_disc_names.append(c)
                    numeric_disc_scalers[c] = scaler
                    if hasattr(scaler, 'mean_'):
                        disc_numeric_means[c] = scaler.mean_[0]
                        disc_numeric_stds[c] = scaler.scale_[0]
                    else:
                        disc_numeric_means[c] = scaler.min_[0]
                        disc_numeric_stds[c] = scaler.scale_[0]
                else:
                    mat, names = one_hot_from_series(s.astype(str))
                    disc_blocks.append(mat)
                    disc_names.extend(names)
                    categorical_vars_list.append(c)
                    cats = [name.split('=')[1] for name in names]
                    categorical_cats_dict[c] = cats
            else:
                if frac_num > 0.95:
                    arr = s_num.values.reshape(-1, 1).astype(np.float32)
                    if numeric_discrete_mode == "zscore":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    arr = scaler.fit_transform(arr).astype(np.float32)
                    numeric_disc_blocks.append(arr)
                    numeric_disc_names.append(c)
                    numeric_disc_scalers[c] = scaler
                    if hasattr(scaler, 'mean_'):
                        disc_numeric_means[c] = scaler.mean_[0]
                        disc_numeric_stds[c] = scaler.scale_[0]
                    else:
                        disc_numeric_means[c] = scaler.min_[0]
                        disc_numeric_stds[c] = scaler.scale_[0]
                else:
                    mat, names = one_hot_from_series(s.astype(str))
                    disc_blocks.append(mat)
                    disc_names.extend(names)
                    categorical_vars_list.append(c)
                    cats = [name.split('=')[1] for name in names]
                    categorical_cats_dict[c] = cats

        blocks = []
        c_names = []

        disease_start = 0
        disease_dim = disease_onehot.shape[1]
        blocks.append(disease_onehot.astype(np.float32))
        c_names.extend(disease_names)
        disease_end = disease_start + disease_dim

        cont_dim = 0
        cont_miss_dim = 0
        num_disc_dim = 0
        onehot_disc_dim = 0

        if cont_blocks:
            cont_all = np.concatenate(cont_blocks, axis=1)
            blocks.append(cont_all)
            c_names.extend(cont_names)
            cont_dim = cont_all.shape[1]
            if cont_missing_blocks:
                miss_all = np.concatenate(cont_missing_blocks, axis=1)
                blocks.append(miss_all)
                c_names.extend([f"{n}_missing" for n in cont_names])
                cont_miss_dim = miss_all.shape[1]

        if numeric_disc_blocks:
            numd_all = np.concatenate(numeric_disc_blocks, axis=1)
            blocks.append(numd_all)
            c_names.extend(numeric_disc_names)
            num_disc_dim = numd_all.shape[1]

        if disc_blocks:
            disc_all = np.concatenate(disc_blocks, axis=1)
            blocks.append(disc_all)
            c_names.extend(disc_names)
            onehot_disc_dim = disc_all.shape[1]

        C_full = np.concatenate(blocks, axis=1).astype(np.float32)
        others_start = disease_end
        others_end = C_full.shape[1]

        self.X_clr = torch.from_numpy(X_clr)
        self.C = torch.from_numpy(C_full)

        if torch.isnan(self.X_clr).any():
            print("Warning: X contains NaN values. Filling with 0.")
            self.X_clr = torch.nan_to_num(self.X_clr, nan=0.0)
        if torch.isnan(self.C).any():
            print("Warning: C contains NaN values. Filling with 0.")
            self.C = torch.nan_to_num(self.C, nan=0.0)

        self.X_dim = self.X_clr.shape[1]
        self.C_dim = self.C.shape[1]
        self.c_names = c_names

        prior = load_species_prior(species_prior_csv, selected_species)
        prior_weight = prior["prior_weight"]
        pos_mask = prior["pos_mask"]
        neg_mask = prior["neg_mask"]
        neu_mask = prior["neu_mask"]
        prior_idx_pos = prior["prior_idx_pos"]
        prior_idx_neg = prior["prior_idx_neg"]
        prior_idx_neu = prior["prior_idx_neu"]

        self.info = {
            "X_dim": self.X_dim,
            "C_dim": self.C_dim,
            "c_names": self.c_names,
            "subject_ids": self.subject_ids,
            "selected_species": selected_species,
            "abundance_transform": abundance_transform,
            "clr_eps": clr_eps,
            "min_nonzero_species": self.min_nonzero_species,
            "cont_dim": cont_dim,
            "cont_miss_dim": cont_miss_dim,
            "num_disc_dim": num_disc_dim,
            "onehot_disc_dim": onehot_disc_dim,
            "disease_col_name": disease_col,
            "disease_start": disease_start,
            "disease_dim": disease_dim,
            "others_start": others_start,
            "others_dim": others_end - others_start,
            "prior_weight": prior_weight,
            "prior_pos_mask": pos_mask,
            "prior_neg_mask": neg_mask,
            "prior_neu_mask": neu_mask,
            "prior_idx_pos": prior_idx_pos,
            "prior_idx_neg": prior_idx_neg,
            "prior_idx_neu": prior_idx_neu,
            "cont_vars": cont_in_use,
            "cont_means": cont_means,
            "cont_stds": cont_stds,
            "disc_numeric_vars": numeric_disc_names,
            "disc_numeric_means": disc_numeric_means,
            "disc_numeric_stds": disc_numeric_stds,
            "categorical_vars": categorical_vars_list,
            "categorical_cats": categorical_cats_dict,
            "missing_suffix": "_missing",
        }

        print(f"\nFinal dimensions: X_dim={self.X_dim}, C_dim={self.C_dim}")
        print(f"Disease one-hot dim: {disease_dim}, Others dim: {self.info['others_dim']}")
        if np.abs(prior_weight).sum() > 0:
            print(f"Prior guidance enabled: positive={len(prior_idx_pos)}, negative={len(prior_idx_neg)}, neutral={len(prior_idx_neu)}")
            print(f"Total prior confidence: {float(prior_weight.sum()):.3f}")

    def __len__(self):
        return self.X_clr.shape[0]

    def __getitem__(self, idx):
        x_clr = self.X_clr[idx]
        c = self.C[idx]
        return {
            "x_clr": x_clr if self.device is None else x_clr.to(self.device),
            "c": c if self.device is None else c.to(self.device),
            "idx": idx,
        }


def build_dataloaders(
    abundance_csv: str,
    meta_csv: str,
    metadata_txt: str,
    species_prior_csv: str,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    val_split: float = 0.1,
    seed: int = 42,
    save_species_csv: Optional[str] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, Dict]:
    torch.manual_seed(seed)
    ds = MicrobiomeDataset(
        abundance_csv, meta_csv, metadata_txt, species_prior_csv, save_species_csv=save_species_csv, **dataset_kwargs
    )
    n = len(ds)
    n_val = int(n * val_split)
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    info = dict(ds.info)
    return train_loader, val_loader, info
