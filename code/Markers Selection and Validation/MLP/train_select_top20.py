#!/usr/bin/env python3
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

If the Beneficial and harmful groups each contain 10 species, a direct full
mixed exhaustive search would require:

    (2^10 - 1) * (2^10 - 1) = 1,046,529 combinations

The staged strategy reduces the search space while still allowing the final
model to use both candidate groups.

"""

import os
import argparse
import tempfile
import random
import itertools
import math
from typing import List, Dict, Any, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from species_select.code.MLP.cls import ASCClassifier, ASCClassificationLoss
from species_select.code.MLP.dataloader_cls import build_classification_kfold_loaders


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def calculate_metrics(outputs: Dict[str, torch.Tensor], targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    with torch.no_grad():
        probs = outputs["probabilities"].squeeze()
        preds = (probs >= threshold).float()
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        tp = ((preds_np == 1) & (targets_np == 1)).sum()
        tn = ((preds_np == 0) & (targets_np == 0)).sum()
        fp = ((preds_np == 1) & (targets_np == 0)).sum()
        fn = ((preds_np == 0) & (targets_np == 1)).sum()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if len(np.unique(targets_np)) > 1:
            auc = roc_auc_score(targets_np, probs.cpu().numpy())
        else:
            auc = 0.5

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }


def train_epoch(model, train_loader, criterion, optimizer, device: torch.device, grad_clip: float = 10.0):
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        x_gut = batch["x_gut"].to(device)
        x_metadata = batch["x_metadata"].to(device)
        y = batch["y"].to(device)

        outputs = model(x_gut, x_metadata)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x_gut.size(0)
        all_outputs.append(outputs)
        all_targets.append(y)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(1, len(train_loader.dataset))
    all_outputs_dict = {
        "logits": torch.cat([o["logits"] for o in all_outputs]),
        "probabilities": torch.cat([o["probabilities"] for o in all_outputs]),
    }
    all_targets_tensor = torch.cat(all_targets)
    metrics = calculate_metrics(all_outputs_dict, all_targets_tensor)
    return avg_loss, metrics


def validate(model, val_loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x_gut = batch["x_gut"].to(device)
            x_metadata = batch["x_metadata"].to(device)
            y = batch["y"].to(device)

            outputs = model(x_gut, x_metadata)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x_gut.size(0)
            all_outputs.append(outputs)
            all_targets.append(y)

    avg_loss = total_loss / max(1, len(val_loader.dataset))
    all_outputs_dict = {
        "logits": torch.cat([o["logits"] for o in all_outputs]),
        "probabilities": torch.cat([o["probabilities"] for o in all_outputs]),
    }
    all_targets_tensor = torch.cat(all_targets)
    metrics = calculate_metrics(all_outputs_dict, all_targets_tensor)
    return avg_loss, metrics


def train_and_evaluate_kfold(species_list: List[str], args, device: torch.device, n_splits: int = 5):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for sp in species_list:
            f.write(sp + "\n")
        temp_species_file = f.name

    fold_aucs = []
    best_overall_auc = -1.0
    best_model_state = None

    try:
        folds = build_classification_kfold_loaders(
            abundance_csv=args.abundance_csv,
            meta_csv=args.meta_csv,
            important_species_txt=temp_species_file,
            disease_mapping=args.disease_mapping_dict,
            metadata_select_txt=args.metadata_select_txt if args.use_metadata else None,
            batch_size=args.batch_size,
            n_splits=n_splits,
            seed=args.seed,
            use_metadata=args.use_metadata,
            device=None,
        )

        for fold_id, (train_loader, val_loader, info) in enumerate(folds, start=1):
            model = ASCClassifier(
                gut_dim=info["gut_dim"],
                metadata_dim=info["metadata_dim"],
                gut_latent_dim=args.gut_latent_dim,
                metadata_latent_dim=args.metadata_latent_dim,
                classifier_hidden_dims=args.classifier_hidden_dims,
                dropout=args.dropout,
                use_batch_norm=args.use_batch_norm,
                use_metadata=args.use_metadata,
            ).to(device)

            pos_weight = torch.tensor([args.class_weight_1 / args.class_weight_0], device=device)
            criterion = ASCClassificationLoss(pos_weight=pos_weight)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=10,
            )

            best_val_auc = 0.0
            best_state = None
            patience_counter = 0

            for _ in range(1, args.epochs + 1):
                train_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
                _, val_metrics = validate(model, val_loader, criterion, device)
                scheduler.step(val_metrics["auc"])

                if val_metrics["auc"] > best_val_auc:
                    best_val_auc = val_metrics["auc"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        break

            print(f"[Fold {fold_id}/{n_splits}] Best validation AUC = {best_val_auc:.4f}")
            fold_aucs.append(best_val_auc)

            if best_val_auc > best_overall_auc:
                best_overall_auc = best_val_auc
                best_model_state = best_state

    finally:
        try:
            os.unlink(temp_species_file)
        except Exception:
            pass

    mean_val_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    return mean_val_auc, 0.0, best_model_state


def normalize_species_list(species_list: List[str]) -> List[str]:
    return sorted(set([s.strip() for s in species_list if isinstance(s, str) and len(s.strip()) > 0]))


def combo_key(species_list: List[str]) -> str:
    return "|".join(normalize_species_list(species_list))


def all_nonempty_subsets(arr: List[str], max_k: Optional[int] = None) -> Iterable[List[str]]:
    n = len(arr)
    upper = n if max_k is None or max_k <= 0 or max_k > n else max_k
    for r in range(1, upper + 1):
        for comb in itertools.combinations(arr, r):
            yield list(comb)


def mixed_subsets(beneficial: List[str], harmful: List[str], min_size: int = 2) -> Iterable[List[str]]:
    B = normalize_species_list(beneficial)
    H = normalize_species_list(harmful)
    for bset in all_nonempty_subsets(B):
        for hset in all_nonempty_subsets(H):
            combo = normalize_species_list(bset + hset)
            if len(combo) >= min_size:
                yield combo


def estimate_subset_count(n: int, max_k: int) -> int:
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


def evaluate_combo(
    species_combo: List[str],
    stage: str,
    args,
    device: torch.device,
    done_map: Dict[str, Dict[str, Any]],
    results_csv_path: str,
    records: List[Dict[str, Any]],
):
    key = f"{stage}::{combo_key(species_combo)}"
    if key in done_map:
        try:
            cached_auc = float(done_map[key].get("val_auc", -1.0))
        except Exception:
            cached_auc = -1.0
        return cached_auc, None, records, True

    print(f"\n[{stage}] Subset size: {len(species_combo)}")
    print(f"Species: {species_combo}")

    try:
        val_auc, test_auc, model_state = train_and_evaluate_kfold(
            species_combo,
            args,
            device,
            n_splits=args.n_splits,
        )
        rec = {
            "stage": stage,
            "combo_key": key,
            "species_list": " ".join(species_combo),
            "size": len(species_combo),
            "val_auc": val_auc,
            "test_auc": test_auc,
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        print(f"Subset evaluation failed: {e}")
        val_auc = -1.0
        model_state = None
        rec = {
            "stage": stage,
            "combo_key": key,
            "species_list": " ".join(species_combo),
            "size": len(species_combo),
            "val_auc": -1.0,
            "test_auc": -1.0,
            "status": "error",
            "error": str(e),
        }

    records.append(rec)
    records = append_records(records, results_csv_path)
    return val_auc, model_state, records, False


def run_single_group_stage(
    group_name: str,
    species_list: List[str],
    max_k: int,
    args,
    device: torch.device,
    done_map: Dict[str, Dict[str, Any]],
    results_csv_path: str,
    records: List[Dict[str, Any]],
) -> Tuple[List[str], float, Optional[Dict[str, torch.Tensor]], List[Dict[str, Any]]]:
    best_auc = -1.0
    best_subset = []
    best_model_state = None
    candidates = list(all_nonempty_subsets(normalize_species_list(species_list), max_k=max_k))

    print("\n" + "=" * 60)
    print(f"Stage: {group_name}")
    print(f"Number of candidate subsets: {len(candidates):,}")
    print("=" * 60)

    for index, subset in enumerate(candidates, start=1):
        print(f"\nProgress: {index:,}/{len(candidates):,}")
        val_auc, model_state, records, _ = evaluate_combo(
            subset,
            stage=group_name,
            args=args,
            device=device,
            done_map=done_map,
            results_csv_path=results_csv_path,
            records=records,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_subset = subset
            best_model_state = model_state
            print(f"New best {group_name} AUC: {best_auc:.4f}")

    return best_subset, best_auc, best_model_state, records


def run_final_mixed_stage(
    beneficial_subset: List[str],
    harmful_subset: List[str],
    args,
    device: torch.device,
    done_map: Dict[str, Dict[str, Any]],
    results_csv_path: str,
    records: List[Dict[str, Any]],
):
    best_auc = -1.0
    best_combo = []
    best_model_state = None
    candidates = list(mixed_subsets(beneficial_subset, harmful_subset, min_size=args.min_size))

    print("\n" + "=" * 60)
    print("Stage: final_mixed")
    print(f"Number of candidate subsets: {len(candidates):,}")
    print("=" * 60)

    for index, subset in enumerate(candidates, start=1):
        print(f"\nProgress: {index:,}/{len(candidates):,}")
        val_auc, model_state, records, _ = evaluate_combo(
            subset,
            stage="final_mixed",
            args=args,
            device=device,
            done_map=done_map,
            results_csv_path=results_csv_path,
            records=records,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_combo = subset
            best_model_state = model_state
            print(f"New best final mixed AUC: {best_auc:.4f}")

            if best_model_state is not None:
                torch.save(best_model_state, os.path.join(args.save_dir, "global_best_model.pt"))
            with open(os.path.join(args.save_dir, "global_best_species.txt"), "w", encoding="utf-8") as f:
                for sp in best_combo:
                    f.write(sp + "\n")

    return best_combo, best_auc, best_model_state, records


def load_done_map(results_csv_path: str) -> Dict[str, Dict[str, Any]]:
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
    parser = argparse.ArgumentParser(description="Staged species-subset search for ASD classification.")

    parser.add_argument("--top20_csv", type=str, default="XXX", help="CSV file with columns: Beneficial and harmful.")
    parser.add_argument("--abundance_csv", type=str, default='XXX')
    parser.add_argument("--meta_csv", type=str, default='XXX')
    parser.add_argument("--metadata_select_txt", type=str, default='XXX', help="Optional text file listing metadata columns to use, one per line. If not provided, all metadata columns will be used.")
    parser.add_argument("--disease_col_name", type=str, default="XXX", help="Column name in meta_csv that contains disease labels.")
    parser.add_argument("--disease_mapping", type=str, default="XXX", help="Mapping of disease labels to binary values, e.g., 'ASD:1,Control:0'.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--use_metadata", action="store_true", default=True)

    parser.add_argument("--gut_latent_dim", type=int, default=128)
    parser.add_argument("--metadata_latent_dim", type=int, default=64)
    parser.add_argument("--classifier_hidden_dims", type=str, default="128,64,32")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_batch_norm", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--class_weight_0", type=float, default=1.0)
    parser.add_argument("--class_weight_1", type=float, default=1.0)

    parser.add_argument("--save_dir", type=str, default="./selection_results")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--beneficial_max",
        type=int,
        default=10,
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
        default=2,
        help="Minimum total number of species in the final mixed subset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    disease_mapping_dict = {}
    if args.disease_mapping:
        for pair in args.disease_mapping.split(","):
            label, value = pair.split(":")
            disease_mapping_dict[label.strip()] = int(value.strip())
    args.disease_mapping_dict = disease_mapping_dict

    args.classifier_hidden_dims = [int(x) for x in args.classifier_hidden_dims.split(",")]

    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    results_csv_path = os.path.join(args.save_dir, "staged_selection_results.csv")
    best_species_txt = os.path.join(args.save_dir, "best_species.txt")

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
    print(f"Using device: {device}")

    beneficial_count = estimate_subset_count(len(beneficial_list), args.beneficial_max)
    harmful_count = estimate_subset_count(len(harmful_list), args.harmful_max)
    full_mixed_count = (2 ** len(beneficial_list) - 1) * (2 ** len(harmful_list) - 1)

    print(f"Full direct mixed search size: {full_mixed_count:,}")
    print(f"Stage 1 Beneficial subset count: {beneficial_count:,}")
    print(f"Stage 2 harmful subset count: {harmful_count:,}")

    done_map = load_done_map(results_csv_path)
    records: List[Dict[str, Any]] = []

    best_beneficial_subset, best_beneficial_auc, _, records = run_single_group_stage(
        group_name="beneficial_only",
        species_list=beneficial_list,
        max_k=args.beneficial_max,
        args=args,
        device=device,
        done_map=done_map,
        results_csv_path=results_csv_path,
        records=records,
    )

    best_harmful_subset, best_harmful_auc, _, records = run_single_group_stage(
        group_name="harmful_only",
        species_list=harmful_list,
        max_k=args.harmful_max,
        args=args,
        device=device,
        done_map=done_map,
        results_csv_path=results_csv_path,
        records=records,
    )

    print("\nBest Beneficial subset:")
    print(f"AUC: {best_beneficial_auc:.4f}")
    for sp in best_beneficial_subset:
        print(f"  - {sp}")

    print("\nBest harmful subset:")
    print(f"AUC: {best_harmful_auc:.4f}")
    for sp in best_harmful_subset:
        print(f"  - {sp}")

    best_combo, best_auc, best_model_state, records = run_final_mixed_stage(
        beneficial_subset=best_beneficial_subset,
        harmful_subset=best_harmful_subset,
        args=args,
        device=device,
        done_map=done_map,
        results_csv_path=results_csv_path,
        records=records,
    )

    records = append_records(records, results_csv_path, force=True)

    with open(best_species_txt, "w", encoding="utf-8") as f:
        for sp in best_combo:
            f.write(sp + "\n")

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(args.save_dir, "final_best_model.pt"))

    print("\n" + "=" * 60)
    print("Staged search completed.")
    print(f"Best final mixed validation AUC: {best_auc:.4f}")
    print(f"Best final species subset size: {len(best_combo)}")
    for sp in best_combo:
        print(f"  - {sp}")
    print(f"Best species list saved to: {best_species_txt}")
    print(f"Detailed search records saved to: {results_csv_path}")


if __name__ == "__main__":
    main()