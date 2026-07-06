import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


CONTINUOUS_COLS_CANON = [
    "ASC_total", "ASC_uncertainty", "Alc", "BMI", "CBCL/6-18_SSS_AB", "CBCL/6-18_SSS_DB",
    "CBCL_ADHD symptoms", "CBCL_AP_T", "Caff", "Chol", "Enrollment Age", "F_age",
    "F_edu_years", "Height (cm)", "Iodine", "M_CEBQ_DD", "M_CEBQ_EF", "M_CEBQ_EOE",
    "M_CEBQ_EUE", "M_CEBQ_FF", "M_CEBQ_FR", "M_CEBQ_SE", "M_CEBQ_SR", "M_SEQ_hyper",
    "M_SEQ_hypo", "M_SEQ_seeking", "M_SEQ_social", "M_age", "M_edu_years", "PBC1",
    "PBC2", "PBC3", "PBC4", "PBC5", "PBC6", "Retinol", "T_SRS_AWR", "T_SRS_COG",
    "T_SRS_COMM", "T_SRS_MOT", "TransFat", "Vit_B12", "Vit_D_mcg", "Vit_K",
    "beverage", "corn", "dairy", "darkgreen", "egg", "ferm", "firedfood", "fish",
    "juice", "mushroom", "nut", "procmeat", "raw_SRS_RRB", "redmeat", "starch",
    "sugarjam", "wholegrain"
]

DISCRETE_COLS_CANON = [
    "ASC_Classification", "Bristol_stool_chart", "CBCL_Externalizing_Classification",
    "CloseRelative_DevPro", "Drugs", "FAPD", "FDD", "FNVD", "F_occupation", "GI",
    "Gender", "Health_problem", "MR", "M_occupation", "Mental_disorder",
    "SEQ_total_Classification", "SRS_Clinical Category", "Sibling_Total", "GI "
]


def arcsinh_transformer(x: np.ndarray, scale: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.arcsinh(x / scale).astype(np.float32)


def read_metadata_list(metadata_txt: str) -> List[str]:
    names = []
    with open(metadata_txt, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                names.append(item)
    return names


def safe_col_lookup(meta: pd.DataFrame, wanted: List[str]) -> List[str]:
    found = []
    missing = []
    for col in wanted:
        if col in meta.columns:
            found.append(col)
        else:
            matches = [c for c in meta.columns if c.strip() == col.strip()]
            if matches:
                found.append(matches[0])
            else:
                missing.append(col)

    if missing:
        print(f"Warning: the following metadata columns were not found and will be ignored: {missing}")
    return found


def one_hot_from_series(series: pd.Series, name_override: str = None) -> Tuple[np.ndarray, List[str]]:
    series_clean = series.fillna("missing")
    categories = sorted(series_clean.unique().tolist())
    mapping = {cat: i for i, cat in enumerate(categories)}
    matrix = np.zeros((len(series_clean), len(categories)), dtype=np.float32)

    for idx, value in enumerate(series_clean.values):
        matrix[idx, mapping[value]] = 1.0

    base_name = name_override if name_override is not None else series.name
    names = [f"{base_name}={cat}" for cat in categories]
    return matrix, names


class ASCClassificationDataset(Dataset):
    def __init__(
        self,
        abundance_csv: str,
        meta_csv: str,
        important_species_txt: str,
        metadata_select_txt: Optional[str] = None,
        disease_col_name: str = "group_x",
        disease_mapping: Optional[Dict[str, int]] = None,
        arcsinh_transformer_enabled: bool = True,
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

        abundance = pd.read_csv(abundance_csv, index_col=0).T
        available_species = [sp for sp in important_species if sp in abundance.columns]

        if len(available_species) != len(important_species):
            missing = sorted(set(important_species) - set(available_species))
            print(f"Warning: {len(missing)} species were not found in the abundance table: {missing[:5]}")

        abundance = abundance[available_species].fillna(0.0)
        meta = pd.read_csv(meta_csv, index_col="ID")

        common_ids = abundance.index.intersection(meta.index)
        abundance = abundance.loc[common_ids]
        meta = meta.loc[common_ids]
        print(f"Matched samples: {len(common_ids)}")

        disease_col_candidates = [c for c in meta.columns if c.strip() == disease_col_name]
        if not disease_col_candidates:
            raise ValueError(f"Disease column '{disease_col_name}' was not found in the metadata table.")
        disease_col = disease_col_candidates[0]
        disease_series = meta[disease_col]

        if remove_missing_disease:
            missing_mask = disease_series.isna()
            if missing_mask.any():
                n_missing = int(missing_mask.sum())
                print(f"Removing {n_missing} samples with missing disease labels in '{disease_col}'.")
                keep_mask = ~missing_mask
                abundance = abundance.loc[keep_mask]
                meta = meta.loc[keep_mask]
                disease_series = disease_series.loc[keep_mask]

        if remove_disease_3:
            disease_str = disease_series.astype(str).str.strip()
            disease_3_mask = disease_str == "3"
            if disease_3_mask.any():
                n_removed = int(disease_3_mask.sum())
                print(f"Removing {n_removed} samples with disease label 3.")
                keep_mask = ~disease_3_mask
                abundance = abundance.loc[keep_mask]
                meta = meta.loc[keep_mask]
                disease_series = disease_series.loc[keep_mask]
                print(f"Remaining samples: {len(abundance)}")

        self.subject_ids = abundance.index.astype(str).tolist()

        x_raw = abundance.values.astype(np.float32)
        if arcsinh_transformer_enabled:
            x_gut = arcsinh_transformer(x_raw, scale=arcsinh_scale)
        else:
            x_gut = x_raw

        metadata_matrix, metadata_names, cont_dim, num_disc_dim, onehot_disc_dim = self._build_metadata_matrix(
            meta=meta,
            metadata_select_txt=metadata_select_txt,
            disease_col=disease_col,
            standardize_continuous=standardize_continuous,
            include_cont_missing_indicator=include_cont_missing_indicator,
            numeric_discrete_mode=numeric_discrete_mode,
            n_samples=len(abundance),
        )

        disease_labels = self._encode_disease_labels(disease_series)
        unique_labels, counts = np.unique(disease_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))

        self.X_gut = torch.from_numpy(x_gut.astype(np.float32))
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
        }

        print("\nDataset summary:")
        print(f"  Gut feature dimension: {self.info['gut_dim']}")
        print(f"  Metadata dimension: {self.info['metadata_dim']}")
        print(f"  Number of samples: {self.info['n_samples']}")
        print(f"  Class distribution: 0={label_distribution.get(0.0, 0)}, 1={label_distribution.get(1.0, 0)}")
        if self.use_metadata:
            print(f"  Continuous variables: {cont_dim}, numeric discrete variables: {num_disc_dim}, one-hot variables: {onehot_disc_dim}")

        self.X_gut = torch.nan_to_num(self.X_gut, nan=0.0)
        self.X_metadata = torch.nan_to_num(self.X_metadata, nan=0.0)
        self.y = torch.nan_to_num(self.y, nan=0.0)

    def _build_metadata_matrix(
        self,
        meta: pd.DataFrame,
        metadata_select_txt: Optional[str],
        disease_col: str,
        standardize_continuous: bool,
        include_cont_missing_indicator: bool,
        numeric_discrete_mode: str,
        n_samples: int,
    ):
        if not self.use_metadata or metadata_select_txt is None:
            print("Metadata features are disabled.")
            return np.zeros((n_samples, 0), dtype=np.float32), [], 0, 0, 0

        wanted_cols = read_metadata_list(metadata_select_txt)
        if not wanted_cols:
            print(f"Warning: {metadata_select_txt} is empty. Metadata features will be disabled.")
            return np.zeros((n_samples, 0), dtype=np.float32), [], 0, 0, 0

        if disease_col not in wanted_cols:
            wanted_cols.append(disease_col)

        found_cols = safe_col_lookup(meta, wanted_cols)
        meta_used = meta[found_cols].copy()
        meta_others = meta_used.drop(columns=[disease_col] if disease_col in meta_used.columns else [])

        cont_in_use = [c for c in meta_others.columns if c in CONTINUOUS_COLS_CANON]
        disc_in_use = [c for c in meta_others.columns if c in DISCRETE_COLS_CANON and c != disease_col]
        unknown_cols = [c for c in meta_others.columns if c not in cont_in_use and c not in disc_in_use]

        for col in unknown_cols:
            series = meta_others[col]
            is_numeric = pd.to_numeric(series, errors="coerce").notna().mean() > 0.95
            if is_numeric:
                cont_in_use.append(col)
            else:
                disc_in_use.append(col)

        blocks = []
        metadata_names = []
        cont_dim = 0
        num_disc_dim = 0
        onehot_disc_dim = 0

        if cont_in_use:
            cont_df = meta_others[cont_in_use].apply(pd.to_numeric, errors="coerce")
            cont_mask = ~cont_df.isna()
            cont_filled = cont_df.fillna(cont_df.mean(axis=0))

            if standardize_continuous:
                scaler = StandardScaler()
                cont_arr = scaler.fit_transform(cont_filled.values).astype(np.float32)
            else:
                cont_arr = cont_filled.values.astype(np.float32)

            blocks.append(cont_arr)
            metadata_names.extend(cont_in_use)
            cont_dim = cont_arr.shape[1]

            if include_cont_missing_indicator:
                missing_arr = (~cont_mask).astype(np.float32).values
                blocks.append(missing_arr)
                metadata_names.extend([f"{c}_missing" for c in cont_in_use])
                cont_dim += missing_arr.shape[1]

        disc_blocks = []
        disc_names = []
        numeric_disc_blocks = []
        numeric_disc_names = []

        for col in disc_in_use:
            series = meta_others[col]
            series_num = pd.to_numeric(series, errors="coerce")
            frac_num = series_num.notna().mean()

            if frac_num > 0.95:
                arr = series_num.values.reshape(-1, 1).astype(np.float32)
                scaler = StandardScaler() if numeric_discrete_mode == "zscore" else MinMaxScaler()
                arr = scaler.fit_transform(arr).astype(np.float32)
                numeric_disc_blocks.append(arr)
                numeric_disc_names.append(col)
            else:
                mat, names = one_hot_from_series(series.astype(str))
                disc_blocks.append(mat)
                disc_names.extend(names)

        if numeric_disc_blocks:
            num_disc_all = np.concatenate(numeric_disc_blocks, axis=1)
            blocks.append(num_disc_all)
            metadata_names.extend(numeric_disc_names)
            num_disc_dim = num_disc_all.shape[1]

        if disc_blocks:
            onehot_all = np.concatenate(disc_blocks, axis=1)
            blocks.append(onehot_all)
            metadata_names.extend(disc_names)
            onehot_disc_dim = onehot_all.shape[1]

        if blocks:
            metadata_matrix = np.concatenate(blocks, axis=1).astype(np.float32)
        else:
            metadata_matrix = np.zeros((n_samples, 0), dtype=np.float32)

        return metadata_matrix, metadata_names, cont_dim, num_disc_dim, onehot_disc_dim

    def _encode_disease_labels(self, disease_series: pd.Series) -> np.ndarray:
        disease_labels = np.zeros(len(disease_series), dtype=np.float32)

        for i, label in enumerate(disease_series.values):
            label_str = str(label).strip()
            if label_str in self.disease_mapping:
                disease_labels[i] = self.disease_mapping[label_str]
                continue

            label_lower = label_str.lower()
            if any(token in label_lower for token in ["normal", "control", "0"]):
                disease_labels[i] = 0
            elif any(token in label_lower for token in ["elevated", "case", "1"]):
                disease_labels[i] = 1
            else:
                print(f"Warning: unknown disease label '{label}'. It will be mapped to 0.")
                disease_labels[i] = 0

        return disease_labels

    def __len__(self):
        return self.X_gut.shape[0]

    def __getitem__(self, idx):
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
    disease_mapping: Optional[Dict[str, int]] = None,
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
    torch.manual_seed(seed)

    dataset = ASCClassificationDataset(
        abundance_csv=abundance_csv,
        meta_csv=meta_csv,
        important_species_txt=important_species_txt,
        metadata_select_txt=metadata_select_txt,
        disease_mapping=disease_mapping,
        remove_disease_3=remove_disease_3,
        use_metadata=use_metadata,
        **dataset_kwargs,
    )

    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int((n - n_test) * val_split)

    indices = torch.randperm(n)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_subset = torch.utils.data.Subset(dataset, test_idx.tolist())

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

    info = dict(dataset.info)
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
    disease_mapping: Optional[Dict[str, int]] = None,
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ASCClassificationDataset(
        abundance_csv=abundance_csv,
        meta_csv=meta_csv,
        important_species_txt=important_species_txt,
        metadata_select_txt=metadata_select_txt,
        disease_mapping=disease_mapping,
        remove_disease_3=remove_disease_3,
        use_metadata=use_metadata,
        **dataset_kwargs,
    )

    y_np = dataset.y.numpy().astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_np)), y_np), start=1):
        train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())

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

        info = dict(dataset.info)
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
