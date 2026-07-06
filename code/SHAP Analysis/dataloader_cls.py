import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader


CONTINUOUS_COLS_CANON = [
    "ASC_total", "ASC_uncertainty", "Alc", "BMI", "CBCL/6-18_SSS_AB", "CBCL/6-18_SSS_DB",
    "CBCL_ADHD symptoms", "CBCL_AP_T", "Caff", "Chol", "Enrollment Age", "F_age", "F_edu_years",
    "Height (cm)", "Iodine", "M_CEBQ_DD", "M_CEBQ_EF", "M_CEBQ_EOE", "M_CEBQ_EUE",
    "M_CEBQ_FF", "M_CEBQ_FR", "M_CEBQ_SE", "M_CEBQ_SR", "M_SEQ_hyper", "M_SEQ_hypo",
    "M_SEQ_seeking", "M_SEQ_social", "M_age", "M_edu_years", "PBC1", "PBC2", "PBC3", "PBC4",
    "PBC5", "PBC6", "Retinol", "T_SRS_AWR", "T_SRS_COG", "T_SRS_COMM", "T_SRS_MOT",
    "TransFat", "Vit_B12", "Vit_D_mcg", "Vit_K", "beverage", "corn", "dairy", "darkgreen", "egg",
    "ferm", "firedfood", "fish", "juice", "mushroom", "nut", "procmeat", "raw_SRS_RRB",
    "redmeat", "starch", "sugarjam", "wholegrain",
]

DISCRETE_COLS_CANON = [
    "ASC_Classification", "Bristol_stool_chart", "CBCL_Externalizing_Classification",
    "CloseRelative_DevPro", "Drugs", "FAPD", "FDD", "FNVD", "F_occupation", "GI", "Gender",
    "Health_problem", "MR", "M_occupation", "Mental_disorder", "SEQ_total_Classification",
    "SRS_Clinical Category", "Sibling_Total", "GI ",
]


def arcsinh_transformer(x: np.ndarray, scale: float = 1e-6) -> np.ndarray:
    """Apply an arcsinh transformation to abundance features."""
    x = np.asarray(x, dtype=np.float32)
    return np.arcsinh(x / scale).astype(np.float32)


def read_metadata_list(metadata_txt: str) -> List[str]:
    """Read selected metadata column names from a text file."""
    names = []
    with open(metadata_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(s)
    return names


def safe_col_lookup(meta: pd.DataFrame, wanted: List[str]) -> List[str]:
    """Find metadata columns while allowing minor whitespace differences."""
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
        print(f"Warning: the following metadata columns were not found and will be ignored: {missing}")

    return found


def one_hot_from_series(series: pd.Series, name_override: str = None) -> Tuple[np.ndarray, List[str]]:
    """Convert a categorical variable to one-hot encoding."""
    series_clean = series.fillna("missing")
    cats = sorted(series_clean.unique().tolist())
    mapping = {c: i for i, c in enumerate(cats)}
    mat = np.zeros((len(series_clean), len(cats)), dtype=np.float32)

    for idx, v in enumerate(series_clean.values):
        mat[idx, mapping[v]] = 1.0

    base_name = name_override if name_override is not None else series.name
    names = [f"{base_name}={c}" for c in cats]
    return mat, names


class ASCClassificationDataset(Dataset):
    """Dataset for ASC classification using selected gut species and optional metadata."""

    def __init__(
        self,
        abundance_csv: str,
        meta_csv: str,
        important_species_txt: str,
        metadata_select_txt: Optional[str] = None,
        disease_col_name: str = "group_x",
        disease_mapping: Dict[str, int] = None,
        use_arcsinh_transform: bool = True,
        arcsinh_scale: float = 1e-6,
        remove_missing_disease: bool = True,
        remove_disease_3: bool = True,
        standardize_continuous: bool = True,
        include_cont_missing_indicator: bool = False,
        numeric_discrete_mode: str = "minmax",
        use_metadata: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device
        self.remove_disease_3 = remove_disease_3
        self.use_metadata = use_metadata

        self.disease_mapping = {"2": 0, "1": 1} if disease_mapping is None else disease_mapping

        with open(important_species_txt, "r", encoding="utf-8") as f:
            important_species = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(important_species)} selected species.")

        abund = pd.read_csv(abundance_csv, index_col=0).T
        available_species = [s for s in important_species if s in abund.columns]

        if len(available_species) != len(important_species):
            missing = sorted(set(important_species) - set(available_species))
            print(f"Warning: {len(missing)} selected species were not found in the abundance table: {missing[:5]}")

        abund = abund[available_species].fillna(0.0)
        meta = pd.read_csv(meta_csv, index_col="ID")

        common_ids = abund.index.intersection(meta.index)
        abund = abund.loc[common_ids]
        meta = meta.loc[common_ids]

        print(f"Sample intersection size: {len(common_ids)}")

        disease_col_cand = [c for c in meta.columns if c.strip() == disease_col_name]
        if len(disease_col_cand) == 0:
            raise ValueError(f"Disease column '{disease_col_name}' was not found in the metadata table.")
        disease_col = disease_col_cand[0]

        disease_series = meta[disease_col]

        if remove_missing_disease:
            missing_mask = disease_series.isna()
            if missing_mask.any():
                n_missing = int(missing_mask.sum())
                print(f"Removing {n_missing} samples with missing disease labels in '{disease_col}'.")
                keep_mask = ~missing_mask
                abund = abund.loc[keep_mask]
                meta = meta.loc[keep_mask]
                disease_series = disease_series.loc[keep_mask]

        if remove_disease_3:
            disease_str = disease_series.astype(str).str.strip()
            disease_3_mask = disease_str == "3"
            if disease_3_mask.any():
                n_disease_3 = int(disease_3_mask.sum())
                print(f"Removing {n_disease_3} samples with disease label 3.")
                keep_mask = ~disease_3_mask
                abund = abund.loc[keep_mask]
                meta = meta.loc[keep_mask]
                disease_series = disease_series.loc[keep_mask]
                print(f"Remaining samples after filtering: {len(abund)}")

        self.subject_ids = abund.index.astype(str).tolist()

        X_raw = abund.values.astype(np.float32)
        if use_arcsinh_transform:
            X_gut = arcsinh_transformer(X_raw, scale=arcsinh_scale)
        else:
            X_gut = X_raw

        cont_dim = 0
        num_disc_dim = 0
        onehot_disc_dim = 0

        if self.use_metadata and metadata_select_txt is not None:
            wanted_cols = read_metadata_list(metadata_select_txt)
            if not wanted_cols:
                print(f"Warning: {metadata_select_txt} is empty or contains no valid metadata fields. Metadata will be disabled.")
                metadata_matrix = np.zeros((len(abund), 0), dtype=np.float32)
                metadata_names = []
            else:
                if disease_col not in wanted_cols:
                    wanted_cols.append(disease_col)

                found_cols = safe_col_lookup(meta, wanted_cols)
                meta_used = meta[found_cols].copy()
                meta_others = meta_used.drop(columns=[disease_col] if disease_col in meta_used.columns else [])

                cont_in_use = [c for c in meta_others.columns if c in CONTINUOUS_COLS_CANON]
                disc_in_use = [c for c in meta_others.columns if c in DISCRETE_COLS_CANON and c != disease_col]
                others = [c for c in meta_others.columns if c not in cont_in_use and c not in disc_in_use]

                for c in others:
                    s = meta_others[c]
                    is_numeric = pd.to_numeric(s, errors="coerce").notna().mean() > 0.95
                    if is_numeric:
                        cont_in_use.append(c)
                    else:
                        disc_in_use.append(c)

                cont_blocks = []
                cont_names = []
                if len(cont_in_use) > 0:
                    scaler = StandardScaler() if standardize_continuous else None
                    cont_df = meta_others[cont_in_use].apply(pd.to_numeric, errors="coerce")
                    fill_vals = cont_df.mean(axis=0)
                    cont_filled = cont_df.fillna(fill_vals)
                    cont_arr = scaler.fit_transform(cont_filled.values) if scaler is not None else cont_filled.values
                    cont_blocks.append(cont_arr.astype(np.float32))
                    cont_names.extend(cont_in_use)

                disc_blocks = []
                disc_names = []
                numeric_disc_blocks = []
                numeric_disc_names = []

                for c in disc_in_use:
                    s = meta_others[c]
                    s_num = pd.to_numeric(s, errors="coerce")
                    frac_num = s_num.notna().mean()

                    if frac_num > 0.95:
                        arr = s_num.values.reshape(-1, 1).astype(np.float32)
                        scaler = StandardScaler() if numeric_discrete_mode == "zscore" else MinMaxScaler()
                        arr = scaler.fit_transform(arr).astype(np.float32)
                        numeric_disc_blocks.append(arr)
                        numeric_disc_names.append(c)
                    else:
                        mat, names = one_hot_from_series(s.astype(str))
                        disc_blocks.append(mat)
                        disc_names.extend(names)

                blocks = []
                metadata_names = []

                if cont_blocks:
                    cont_all = np.concatenate(cont_blocks, axis=1)
                    blocks.append(cont_all)
                    metadata_names.extend(cont_names)
                    cont_dim = cont_all.shape[1]

                if numeric_disc_blocks:
                    numd_all = np.concatenate(numeric_disc_blocks, axis=1)
                    blocks.append(numd_all)
                    metadata_names.extend(numeric_disc_names)
                    num_disc_dim = numd_all.shape[1]

                if disc_blocks:
                    disc_all = np.concatenate(disc_blocks, axis=1)
                    blocks.append(disc_all)
                    metadata_names.extend(disc_names)
                    onehot_disc_dim = disc_all.shape[1]

                if blocks:
                    metadata_matrix = np.concatenate(blocks, axis=1).astype(np.float32)
                else:
                    metadata_matrix = np.zeros((len(abund), 0), dtype=np.float32)
        else:
            print("Metadata features are disabled.")
            metadata_matrix = np.zeros((len(abund), 0), dtype=np.float32)
            metadata_names = []

        disease_labels_raw = disease_series.values
        disease_labels = np.zeros(len(disease_labels_raw), dtype=np.float32)

        for i, label in enumerate(disease_labels_raw):
            label_str = str(label).strip()
            if label_str in self.disease_mapping:
                disease_labels[i] = self.disease_mapping[label_str]
            else:
                label_lower = label_str.lower()
                if any(x in label_lower for x in ["normal", "control", "0"]):
                    disease_labels[i] = 0
                elif any(x in label_lower for x in ["elevated", "case", "1"]):
                    disease_labels[i] = 1
                else:
                    print(f"Warning: unknown disease label '{label}'. It will be mapped to 0.")
                    disease_labels[i] = 0

        unique_labels, counts = np.unique(disease_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))

        self.X_gut = torch.from_numpy(X_gut.astype(np.float32))
        self.X_metadata = torch.from_numpy(metadata_matrix.astype(np.float32))
        self.y = torch.from_numpy(disease_labels.astype(np.float32))

        self.info = {
            "gut_dim": self.X_gut.shape[1],
            "metadata_dim": self.X_metadata.shape[1],
            "n_samples": len(self),
            "species_names": available_species,
            "metadata_names": metadata_names,
            "subject_ids": self.subject_ids,
            "disease_mapping": self.disease_mapping,
            "class_distribution": label_distribution,
            "cont_dim": cont_dim if self.use_metadata else 0,
            "num_disc_dim": num_disc_dim if self.use_metadata else 0,
            "onehot_disc_dim": onehot_disc_dim if self.use_metadata else 0,
            "removed_disease_3": remove_disease_3,
            "use_metadata": self.use_metadata,
            "abundance_transform": "arcsinh_transformer" if use_arcsinh_transform else "none",
        }

        print("\nDataset construction completed:")
        print(f"  Gut feature dimension: {self.info['gut_dim']}")
        print(f"  Metadata dimension: {self.info['metadata_dim']}")
        print(f"  Number of samples: {self.info['n_samples']}")
        print(f"  Class distribution: 0={label_distribution.get(0, 0)}, 1={label_distribution.get(1, 0)}")
        if self.use_metadata:
            print(f"  Continuous features: {cont_dim}, numeric discrete features: {num_disc_dim}, one-hot discrete features: {onehot_disc_dim}")

        if torch.isnan(self.X_gut).any():
            print("Warning: X_gut contains NaN values. Replacing them with 0.")
            self.X_gut = torch.nan_to_num(self.X_gut, nan=0.0)
        if torch.isnan(self.X_metadata).any():
            print("Warning: X_metadata contains NaN values. Replacing them with 0.")
            self.X_metadata = torch.nan_to_num(self.X_metadata, nan=0.0)
        if torch.isnan(self.y).any():
            print("Warning: y contains NaN values. Replacing them with 0.")
            self.y = torch.nan_to_num(self.y, nan=0.0)

    def __len__(self) -> int:
        return self.X_gut.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_gut = self.X_gut[idx]
        x_metadata = self.X_metadata[idx]
        y = self.y[idx]

        if self.device is not None:
            x_gut = x_gut.to(self.device)
            x_metadata = x_metadata.to(self.device)
            y = y.to(self.device)

        return {
            "x_gut": x_gut,
            "x_metadata": x_metadata,
            "y": y,
            "idx": idx,
        }


def build_classification_dataloaders(
    abundance_csv: str,
    meta_csv: str,
    important_species_txt: str,
    metadata_select_txt: Optional[str] = None,
    disease_mapping: Dict[str, int] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = True,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    remove_disease_3: bool = True,
    use_metadata: bool = True,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Build train, validation, and test DataLoaders for ASC classification."""
    torch.manual_seed(seed)

    ds = ASCClassificationDataset(
        abundance_csv=abundance_csv,
        meta_csv=meta_csv,
        important_species_txt=important_species_txt,
        metadata_select_txt=metadata_select_txt,
        disease_mapping=disease_mapping,
        remove_disease_3=remove_disease_3,
        use_metadata=use_metadata,
        **dataset_kwargs,
    )

    n = len(ds)
    n_test = int(n * test_split)
    n_val = int((n - n_test) * val_split)

    indices = torch.randperm(n)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())
    test_subset = torch.utils.data.Subset(ds, test_idx.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    info = dict(ds.info)
    info.update({
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test_size": len(test_subset),
    })

    print(f"\nData split: train={len(train_subset)}, validation={len(val_subset)}, test={len(test_subset)}")

    return train_loader, val_loader, test_loader, info


def build_classification_kfold_loaders(
    abundance_csv: str,
    meta_csv: str,
    important_species_txt: str,
    metadata_select_txt: Optional[str] = None,
    disease_mapping: Dict[str, int] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
    n_splits: int = 5,
    seed: int = 42,
    remove_disease_3: bool = True,
    use_metadata: bool = True,
    drop_last_train: bool = True,
    **dataset_kwargs,
):
    """Build stratified K-fold DataLoaders for ASC classification."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = ASCClassificationDataset(
        abundance_csv=abundance_csv,
        meta_csv=meta_csv,
        important_species_txt=important_species_txt,
        metadata_select_txt=metadata_select_txt,
        disease_mapping=disease_mapping,
        remove_disease_3=remove_disease_3,
        use_metadata=use_metadata,
        **dataset_kwargs,
    )

    y_np = ds.y.numpy().astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_np)), y_np), start=1):
        train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
        val_subset = torch.utils.data.Subset(ds, val_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=drop_last_train,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        info = dict(ds.info)
        info.update({
            "train_size": len(train_subset),
            "val_size": len(val_subset),
            "test_size": 0,
            "fold": fold_id,
            "n_splits": n_splits,
        })

        print(f"\nK-fold {fold_id}/{n_splits}: train={len(train_subset)}, validation={len(val_subset)}")
        folds.append((train_loader, val_loader, info))

    return folds
