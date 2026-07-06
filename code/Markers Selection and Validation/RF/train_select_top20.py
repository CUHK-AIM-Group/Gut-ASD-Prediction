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
import random
import math
from typing import List, Dict, Any, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

from dataloader_cls import build_classification_kfold_loaders


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_from_loader(loader):
    x_gut_list = []
    x_meta_list = []
    y_list = []

    for batch in loader:
        x_gut_list.append(batch["x_gut"].cpu().numpy())
        x_meta_list.append(batch["x_metadata"].cpu().numpy())
        y_list.append(batch["y"].cpu().numpy())

    if not x_gut_list:
        return None, None, None

    x_gut = np.concatenate(x_gut_list, axis=0)
    x_meta = np.concatenate(x_meta_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return x_gut, x_meta, y


def train_eval_one_fold(
    train_loader,
    val_loader,
    use_metadata: bool,
    rf_params: Dict[str, Any],
    return_model: bool = False,
):
    x_gut_train, x_meta_train, y_train = extract_from_loader(train_loader)
    x_gut_val, x_meta_val, y_val = extract_from_loader(val_loader)

    if x_gut_train is None or x_gut_val is None:
        raise ValueError("Empty train or validation loader.")

    if use_metadata:
        x_train = np.concatenate([x_gut_train, x_meta_train], axis=1)
        x_val = np.concatenate([x_gut_val, x_meta_val], axis=1)
    else:
        x_train = x_gut_train
        x_val = x_gut_val

    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_split=rf_params["min_samples_split"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=rf_params["seed"],
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    y_val_proba = rf.predict_proba(x_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)

    if return_model:
        return val_auc, rf
    return val_auc


def train_and_evaluate_kfold(species_list: List[str], args, n_splits: int = 5):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for species in species_list:
            f.write(species + "\n")
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

        rf_params = {
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
            "min_samples_split": args.rf_min_samples_split,
            "min_samples_leaf": args.rf_min_samples_leaf,
            "seed": args.seed,
        }

        aucs = []
        best_fold_auc = -1.0
        best_fold_model = None

        for train_loader, val_loader, _ in folds:
            val_auc, rf_model = train_eval_one_fold(
                train_loader,
                val_loader,
                args.use_metadata,
                rf_params,
                return_model=True,
            )
            aucs.append(val_auc)

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_fold_model = rf_model

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        return mean_auc, best_fold_model

    finally:
        try:
            os.unlink(temp_species_file)
        except OSError:
            pass


def normalize_species_list(species_list: List[str]) -> List[str]:
    return sorted(set([s.strip() for s in species_list if isinstance(s, str) and s.strip()]))


def combo_key(species_list: List[str]) -> str:
    return "|".join(normalize_species_list(species_list))


def subsets(arr: List[str], max_k: int) -> Iterable[List[str]]:
    n = len(arr)
    upper = n if max_k <= 0 or max_k > n else max_k
    for r in range(1, upper + 1):
        for comb in itertools.combinations(arr, r):
            yield list(comb)


def bounded_mixed_combinations(
    beneficial: List[str],
    harmful: List[str],
    beneficial_max: int,
    harmful_max: int,
    min_size: int,
    allow_only_beneficial: bool = True,
    allow_only_harmful: bool = True,
):
    beneficial = normalize_species_list(beneficial)
    harmful = normalize_species_list(harmful)

    beneficial_subsets = list(subsets(beneficial, beneficial_max)) if beneficial_max != 0 else []
    harmful_subsets = list(subsets(harmful, harmful_max)) if harmful_max != 0 else []

    for beneficial_subset in beneficial_subsets:
        for harmful_subset in harmful_subsets:
            combo = normalize_species_list(beneficial_subset + harmful_subset)
            if len(combo) >= min_size:
                yield combo

    if allow_only_beneficial:
        for beneficial_subset in beneficial_subsets:
            if len(beneficial_subset) >= min_size:
                yield normalize_species_list(beneficial_subset)

    if allow_only_harmful:
        for harmful_subset in harmful_subsets:
            if len(harmful_subset) >= min_size:
                yield normalize_species_list(harmful_subset)


def count_subsets(n: int, max_k: int) -> int:
    upper = n if max_k <= 0 or max_k > n else max_k
    return sum(math.comb(n, k) for k in range(1, upper + 1))


def parse_disease_mapping(mapping_text: str) -> Dict[str, int]:
    disease_mapping = {}
    if not mapping_text:
        return disease_mapping

    for pair in mapping_text.split(","):
        label, value = pair.split(":")
        disease_mapping[label.strip()] = int(value.strip())

    return disease_mapping


def load_previous_results(results_csv_path: str) -> Dict[str, Dict[str, Any]]:
    done_map: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(results_csv_path):
        return done_map

    try:
        previous_df = pd.read_csv(results_csv_path)
        for _, row in previous_df.iterrows():
            key = str(row.get("combo_key", ""))
            if key:
                done_map[key] = row.to_dict()
        print(f"Loaded {len(done_map)} previous result records. Completed combinations will be skipped.")
    except Exception as exc:
        print(f"Failed to read previous results. The search will restart without cache. Error: {exc}")

    return done_map


def append_record(records: List[Dict[str, Any]], results_csv_path: str, record: Dict[str, Any], force: bool = False):
    records.append(record)

    if not force and len(records) < 50:
        return records

    df = pd.DataFrame(records)
    mode = "a" if os.path.exists(results_csv_path) else "w"
    header = not os.path.exists(results_csv_path)
    df.to_csv(results_csv_path, index=False, mode=mode, header=header)
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="Random-forest exhaustive species-subset search.")

    parser.add_argument("--top20_csv", type=str, default="XXX", help="CSV file with columns: Beneficial and harmful.")
    parser.add_argument("--abundance_csv", type=str, default="XXX")
    parser.add_argument("--meta_csv", type=str, default="XXX")
    parser.add_argument("--metadata_select_txt", type=str, default="XXX")
    parser.add_argument("--disease_col_name", type=str, default="XXX", help="Column name in meta_csv that contains disease labels.")
    parser.add_argument("--disease_mapping", type=str, default="XXX", help="Mapping of disease labels to integer values, e.g., 'Control:0,ASD:1'.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--use_metadata", action="store_true", default=True)

    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--rf_n_estimators", type=int, default=100)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_split", type=int, default=2)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=1)

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
        help="Minimum total number of species in each candidate subset.",
    )

    parser.add_argument("--save_dir", type=str, default="./rf_exhaustive_results")
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.disease_mapping_dict = parse_disease_mapping(args.disease_mapping)

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    results_csv_path = os.path.join(args.save_dir, "exhaustive_selection_results.csv")
    global_best_model_path = os.path.join(args.save_dir, "global_best_rf_model.joblib")
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

    beneficial_count = count_subsets(len(beneficial_list), args.beneficial_max)
    harmful_count = count_subsets(len(harmful_list), args.harmful_max)
    total_estimate = beneficial_count * harmful_count

    print(f"Estimated mixed search size after subset limits: {total_estimate:,}")

    done_map = load_previous_results(results_csv_path)

    best_auc = -1.0
    best_combo = []
    best_model = None

    if done_map:
        try:
            previous_values = [(float(v.get("val_auc", -1.0)), v) for v in done_map.values()]
            previous_values.sort(key=lambda item: item[0], reverse=True)
            if previous_values and previous_values[0][0] >= 0:
                best_auc = previous_values[0][0]
                best_combo = str(previous_values[0][1].get("species_list", "")).split()
                print(f"Previous best AUC: {best_auc:.4f}; subset size={len(best_combo)}")
        except Exception:
            pass

    records = []
    progress = 0

    combo_iter = bounded_mixed_combinations(
        beneficial=beneficial_list,
        harmful=harmful_list,
        beneficial_max=args.beneficial_max,
        harmful_max=args.harmful_max,
        min_size=args.min_size,
    )

    for species_combo in combo_iter:
        progress += 1
        key = combo_key(species_combo)

        if key in done_map:
            continue

        print(f"\n=== Combination {progress} / ~{total_estimate or '?'} ===")
        print(f"Subset size: {len(species_combo)} | {species_combo}")

        try:
            val_auc, model = train_and_evaluate_kfold(species_combo, args, n_splits=args.n_splits)
        except Exception as exc:
            print(f"Combination evaluation failed: {exc}")
            record = {
                "combo_key": key,
                "species_list": " ".join(species_combo),
                "size": len(species_combo),
                "val_auc": -1.0,
                "status": "error",
                "error": str(exc),
            }
            records = append_record(records, results_csv_path, record)
            continue

        kept_as_best = False
        if val_auc > best_auc:
            best_auc = val_auc
            best_combo = species_combo
            best_model = model
            kept_as_best = True

            if best_model is not None:
                joblib.dump(best_model, global_best_model_path)

            with open(global_best_species_path, "w", encoding="utf-8") as f:
                for species in best_combo:
                    f.write(species + "\n")

            print(f"New global best AUC: {best_auc:.4f} | subset size={len(best_combo)}")

        record = {
            "combo_key": key,
            "species_list": " ".join(species_combo),
            "size": len(species_combo),
            "val_auc": val_auc,
            "is_global_best": kept_as_best,
            "status": "ok",
        }
        records = append_record(records, results_csv_path, record)

    if records:
        df = pd.DataFrame(records)
        mode = "a" if os.path.exists(results_csv_path) else "w"
        header = not os.path.exists(results_csv_path)
        df.to_csv(results_csv_path, index=False, mode=mode, header=header)
        records = []

    best_species_txt = os.path.join(args.save_dir, "best_species.txt")
    with open(best_species_txt, "w", encoding="utf-8") as f:
        for species in best_combo:
            f.write(species + "\n")

    print("\n" + "=" * 50)
    print("Exhaustive search completed.")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Best species subset size: {len(best_combo)}")
    for species in best_combo:
        print(f"  - {species}")
    print(f"Best species list saved to: {best_species_txt}")
    print(f"Detailed search records saved to: {results_csv_path}")
    if best_model is not None:
        print(f"Best model saved to: {global_best_model_path}")


if __name__ == "__main__":
    main()
