"""
Staged species-subset selection for ASD classification.

This script reads two candidate species groups from a CSV file:
- Beneficial
- harmful

It then performs a staged search:
1. Search the best subset among the top Beneficial species.
2. Search the best subset among the top harmful species.
3. Merge the selected Beneficial and harmful subsets, then perform a final
   exhaustive mixed search over the merged candidate space.

"""

import os
import argparse
import tempfile
import itertools
import math
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

from dataloader_cls import build_classification_kfold_loaders


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_from_loader(loader):
    X_gut_list, X_meta_list, y_list = [], [], []

    for batch in loader:
        X_gut_list.append(batch["x_gut"].numpy())
        X_meta_list.append(batch["x_metadata"].numpy())
        y_list.append(batch["y"].numpy())

    if not X_gut_list:
        return None, None, None

    X_gut = np.concatenate(X_gut_list, axis=0)
    X_meta = np.concatenate(X_meta_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X_gut, X_meta, y


def train_eval_one_fold_svm(
    train_loader,
    val_loader,
    use_metadata: bool,
    svm_params: Dict[str, Any],
    return_model: bool = False,
):
    X_gut_train, X_meta_train, y_train = extract_from_loader(train_loader)
    X_gut_val, X_meta_val, y_val = extract_from_loader(val_loader)

    if X_gut_train is None or X_gut_val is None:
        raise ValueError("Empty training or validation loader.")

    if use_metadata:
        X_train = np.concatenate([X_gut_train, X_meta_train], axis=1)
        X_val = np.concatenate([X_gut_val, X_meta_val], axis=1)
    else:
        X_train = X_gut_train
        X_val = X_gut_val

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    gamma = svm_params["gamma"]
    try:
        if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
            gamma = float(gamma)
    except ValueError:
        raise ValueError("svm_gamma must be 'scale', 'auto', or a numeric value.")

    svm = SVC(
        C=svm_params["C"],
        kernel=svm_params["kernel"],
        gamma=gamma,
        probability=True,
        random_state=svm_params["seed"],
        class_weight="balanced" if svm_params["balanced"] else None,
        max_iter=svm_params["max_iter"],
    )

    svm.fit(X_train_scaled, y_train)

    if len(np.unique(y_val)) > 1:
        y_val_proba = svm.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)
    else:
        val_auc = 0.5

    if return_model:
        return val_auc, svm, scaler

    return val_auc


def train_and_evaluate_kfold(species_list: List[str], args, n_splits: int = 5):
    """
    Build K-fold loaders for a species subset, train one SVM per fold, and
    return the mean validation AUC together with the best fold model and scaler.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for sp in species_list:
            f.write(sp + "\n")
        temp_species_file = f.name

    try:
        folds = build_classification_kfold_loaders(
            abundance_csv=args.abundance_csv,
            meta_csv=args.meta_csv,
            important_species_txt=temp_species_file,
            metadata_select_txt=args.metadata_select_txt if args.use_metadata else None,
            disease_mapping=args.disease_mapping_dict,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle_train=True,
            n_splits=n_splits,
            seed=args.seed,
            remove_disease_3=True,
            use_metadata=args.use_metadata,
            drop_last_train=True,
            disease_col_name=args.disease_col_name,
            device=None,
        )

        svm_params = {
            "C": args.svm_C,
            "kernel": args.svm_kernel,
            "gamma": args.svm_gamma,
            "balanced": args.svm_balanced,
            "max_iter": args.svm_max_iter,
            "seed": args.seed,
        }

        aucs = []
        best_fold_auc = -1.0
        best_fold_model = None
        best_fold_scaler = None

        for fold_id, (train_loader, val_loader, info) in enumerate(folds, start=1):
            val_auc, svm_model, scaler = train_eval_one_fold_svm(
                train_loader,
                val_loader,
                args.use_metadata,
                svm_params,
                return_model=True,
            )

            aucs.append(val_auc)

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_fold_model = svm_model
                best_fold_scaler = scaler

            print(f"[Fold {fold_id}/{n_splits}] Validation AUC = {val_auc:.4f}")

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        return mean_auc, (best_fold_model, best_fold_scaler)

    finally:
        try:
            os.unlink(temp_species_file)
        except OSError:
            pass


def normalize_species_list(species_list: List[str]) -> List[str]:
    return sorted(set([s.strip() for s in species_list if isinstance(s, str) and len(s.strip()) > 0]))


def combo_key(species_list: List[str]) -> str:
    return "|".join(normalize_species_list(species_list))


def bounded_mixed_combinations(
    beneficial: List[str],
    harmful: List[str],
    b_max: int,
    h_max: int,
    min_size: int,
    allow_only_beneficial: bool = True,
    allow_only_harmful: bool = True,
):
    """
    Generate bounded candidate subsets.

    The generator includes:
    - Mixed subsets containing Beneficial and harmful species.
    - Beneficial-only subsets when enabled.
    - harmful-only subsets when enabled.
    """
    B = normalize_species_list(beneficial)
    H = normalize_species_list(harmful)

    def subsets(arr: List[str], max_k: int):
        n = len(arr)
        upper = n if max_k <= 0 or max_k > n else max_k
        for r in range(1, upper + 1):
            for comb in itertools.combinations(arr, r):
                yield list(comb)

    b_subsets = list(subsets(B, b_max)) if b_max != 0 else []
    h_subsets = list(subsets(H, h_max)) if h_max != 0 else []

    for bset in b_subsets:
        for hset in h_subsets:
            total = len(bset) + len(hset)
            if total >= min_size:
                yield normalize_species_list(bset + hset)

    if allow_only_beneficial:
        for bset in b_subsets:
            if len(bset) >= min_size:
                yield normalize_species_list(bset)

    if allow_only_harmful:
        for hset in h_subsets:
            if len(hset) >= min_size:
                yield normalize_species_list(hset)


def count_subsets(n: int, max_k: int) -> int:
    upper = n if max_k <= 0 or max_k > n else max_k
    return sum(math.comb(n, k) for k in range(1, upper + 1))


def append_records(records: List[Dict[str, Any]], results_csv_path: str, force: bool = False) -> List[Dict[str, Any]]:
    if not records:
        return records

    if not force and len(records) < 50:
        return records

    df = pd.DataFrame(records)
    mode = "a" if os.path.exists(results_csv_path) else "w"
    header = not os.path.exists(results_csv_path)
    df.to_csv(results_csv_path, index=False, mode=mode, header=header)

    return []


def parse_disease_mapping(mapping_text: str) -> Dict[str, int]:
    mapping = {}
    if mapping_text:
        for pair in mapping_text.split(","):
            label, value = pair.split(":")
            mapping[label.strip()] = int(value.strip())
    return mapping


def load_previous_results(results_csv_path: str) -> Dict[str, Dict[str, Any]]:
    done_map: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(results_csv_path):
        return done_map

    try:
        prev_df = pd.read_csv(results_csv_path)
        for _, row in prev_df.iterrows():
            key = str(row.get("combo_key", ""))
            if key:
                done_map[key] = row.to_dict()
        print(f"Loaded {len(done_map)} previous result records. Completed subsets will be skipped.")
    except Exception as e:
        print(f"Failed to read previous result records. The search will start without cache. Error: {e}")

    return done_map


def parse_args():
    parser = argparse.ArgumentParser(description="Exhaustive SVM species-subset search for ASD classification.")

    parser.add_argument("--top20_csv", type=str, default="XXX", help="CSV file with columns: Beneficial and harmful.")
    parser.add_argument("--abundance_csv", type=str, default='XXX')
    parser.add_argument("--meta_csv", type=str, default='XXX')
    parser.add_argument("--metadata_select_txt", type=str, default='XXX', help="Optional text file listing metadata columns to include.")
    parser.add_argument("--disease_col_name", type=str, default="XXX", help="Column name in meta_csv that contains disease labels.")
    parser.add_argument("--disease_mapping", type=str, default="XXX", help="Mapping of disease labels to integers, e.g., 'Control:0,ASD:1'.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--use_metadata", action="store_true", default=True)

    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--svm_C", type=float, default=1.0, help="SVM regularization parameter C.")
    parser.add_argument(
        "--svm_kernel",
        type=str,
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="SVM kernel type.",
    )
    parser.add_argument("--svm_gamma", type=str, default="scale", help="Kernel coefficient: 'scale', 'auto', or a numeric value.")
    parser.add_argument("--svm_balanced", action="store_true", default=True, help="Use balanced class weights.")
    parser.add_argument("--svm_max_iter", type=int, default=-1, help="Maximum number of SVM iterations. Use -1 for no limit.")

    parser.add_argument(
        "--beneficial_max",
        type=int,
        default=0,
        help="Maximum number of Beneficial species in each subset.",
    )
    parser.add_argument(
        "--harmful_max",
        type=int,
        default=10,
        help="Maximum number of harmful species in each subset.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=0,
        help="Minimum total number of species in each subset.",
    )

    parser.add_argument("--save_dir", type=str, default="./svm_exhaustive_results")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.disease_mapping_dict = parse_disease_mapping(args.disease_mapping)

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    results_csv_path = os.path.join(args.save_dir, "exhaustive_selection_results.csv")
    global_best_model_path = os.path.join(args.save_dir, "global_best_svm_model.joblib")
    global_best_species_path = os.path.join(args.save_dir, "global_best_species.txt")

    top20_df = pd.read_csv(args.top20_csv)
    cols_lower = {c.lower(): c for c in top20_df.columns}

    if "beneficial" not in cols_lower or "harmful" not in cols_lower:
        raise ValueError("top20_csv must contain two columns: Beneficial and harmful.")

    beneficial_col = cols_lower["beneficial"]
    harmful_col = cols_lower["harmful"]

    beneficial_list = [str(x).strip() for x in top20_df[beneficial_col].dropna().tolist() if str(x).strip()]
    harmful_list = [str(x).strip() for x in top20_df[harmful_col].dropna().tolist() if str(x).strip()]

    beneficial_list = list(dict.fromkeys(beneficial_list))
    harmful_list = list(dict.fromkeys(harmful_list))

    if not beneficial_list or not harmful_list:
        raise ValueError("The Beneficial and harmful columns must both contain at least one valid species name.")

    print(f"Loaded {len(beneficial_list)} Beneficial species and {len(harmful_list)} harmful species.")

    b_count = count_subsets(len(beneficial_list), args.beneficial_max)
    h_count = count_subsets(len(harmful_list), args.harmful_max)
    total_est = b_count * h_count

    print(f"Estimated number of bounded mixed combinations: {total_est:,}")

    done_map = load_previous_results(results_csv_path)

    best_auc = -1.0
    best_combo: List[str] = []
    best_model_scaler: Tuple[Optional[SVC], Optional[StandardScaler]] = (None, None)

    if done_map:
        try:
            vals = [(float(v.get("val_auc", -1.0)), v) for v in done_map.values()]
            vals.sort(key=lambda x: x[0], reverse=True)
            if vals and vals[0][0] >= 0:
                best_auc = vals[0][0]
                best_combo = str(vals[0][1].get("species_list", "")).split()
                print(f"Previous best AUC: {best_auc:.4f}; subset size = {len(best_combo)}")
        except Exception:
            pass

    progress = 0
    records: List[Dict[str, Any]] = []

    combo_iter = bounded_mixed_combinations(
        beneficial_list,
        harmful_list,
        args.beneficial_max,
        args.harmful_max,
        args.min_size,
    )

    for species_combo in combo_iter:
        progress += 1
        key = combo_key(species_combo)

        if key in done_map:
            continue

        print(f"\n=== Combination {progress} / ~{total_est or '?'} ===")
        print(f"Subset size: {len(species_combo)} | {species_combo}")

        try:
            val_auc, (model, scaler) = train_and_evaluate_kfold(
                species_combo,
                args,
                n_splits=args.n_splits,
            )
        except Exception as e:
            print(f"Combination evaluation failed: {e}")
            records.append(
                {
                    "combo_key": key,
                    "species_list": " ".join(species_combo),
                    "size": len(species_combo),
                    "val_auc": -1.0,
                    "status": "error",
                    "error": str(e),
                }
            )
            records = append_records(records, results_csv_path)
            continue

        kept_as_best = False

        if val_auc > best_auc:
            best_auc = val_auc
            best_combo = species_combo
            best_model_scaler = (model, scaler)
            kept_as_best = True

            if model is not None:
                joblib.dump({"model": model, "scaler": scaler}, global_best_model_path)

            with open(global_best_species_path, "w", encoding="utf-8") as f:
                for sp in best_combo:
                    f.write(sp + "\n")

            print(f"New global best AUC: {best_auc:.4f} | subset size = {len(best_combo)}")
            print("Best model and species subset have been saved.")

        records.append(
            {
                "combo_key": key,
                "species_list": " ".join(species_combo),
                "size": len(species_combo),
                "val_auc": val_auc,
                "is_global_best": kept_as_best,
                "status": "ok",
            }
        )
        records = append_records(records, results_csv_path)

    records = append_records(records, results_csv_path, force=True)

    best_species_txt = os.path.join(args.save_dir, "best_species.txt")
    with open(best_species_txt, "w", encoding="utf-8") as f:
        for sp in best_combo:
            f.write(sp + "\n")

    if best_model_scaler[0] is not None:
        joblib.dump({"model": best_model_scaler[0], "scaler": best_model_scaler[1]}, global_best_model_path)

    print("\n" + "=" * 50)
    print("Exhaustive SVM search completed.")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Best species subset size: {len(best_combo)}")

    for sp in best_combo:
        print(f"  - {sp}")

    print(f"Best species list saved to: {best_species_txt}")
    print(f"Detailed search records saved to: {results_csv_path}")

    if best_model_scaler[0] is not None:
        print(f"Best model and scaler saved to: {global_best_model_path}")


if __name__ == "__main__":
    main()
