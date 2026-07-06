import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Optional Weights & Biases import
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

from cls import ASCClassifier, ASCClassificationLoss
from dataloader_cls import ASCClassificationDataset

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(outputs, targets, threshold=0.5):
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

def train_epoch(model, train_loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_probs = []
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
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item() * x_gut.size(0)
        all_logits.append(outputs["logits"].detach())
        all_probs.append(outputs["probabilities"].detach())
        all_targets.append(y.detach())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader.dataset)
    all_outputs_dict = {
        "logits": torch.cat(all_logits),
        "probabilities": torch.cat(all_probs),
    }
    all_targets_tensor = torch.cat(all_targets)
    metrics = calculate_metrics(all_outputs_dict, all_targets_tensor)

    return avg_loss, metrics

def normalize_ids_to_str(sample_ids):
    if isinstance(sample_ids, torch.Tensor):
        sample_ids = sample_ids.cpu().numpy().tolist()
    elif isinstance(sample_ids, np.ndarray):
        sample_ids = sample_ids.tolist()
    sample_ids = [str(s) for s in sample_ids]
    return sample_ids

def extract_ids_from_batch(batch, fallback_start_idx):
    if "subject_id" in batch:
        sample_ids = batch["subject_id"]
    elif "id" in batch:
        sample_ids = batch["id"]
    elif "ids" in batch:
        sample_ids = batch["ids"]
    elif "sample_id" in batch:
        sample_ids = batch["sample_id"]
    elif "idx" in batch:
        sample_ids = batch["idx"]
    else:
        sample_ids = [f"sample_{fallback_start_idx+i}" for i in range(len(batch["y"]))]
    return normalize_ids_to_str(sample_ids)

@torch.no_grad()
def validate_with_ids(model, val_loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_probs = []
    all_targets = []
    all_ids = []

    for batch in val_loader:
        x_gut = batch["x_gut"].to(device)
        x_metadata = batch["x_metadata"].to(device)
        y = batch["y"].to(device)

        sample_ids = extract_ids_from_batch(batch, len(all_ids))

        outputs = model(x_gut, x_metadata)
        loss = criterion(outputs, y)

        total_loss += loss.item() * x_gut.size(0)
        all_logits.append(outputs["logits"].detach())
        all_probs.append(outputs["probabilities"].detach())
        all_targets.append(y.detach())
        all_ids.extend(sample_ids)

    avg_loss = total_loss / len(val_loader.dataset)
    all_outputs = {
        "logits": torch.cat(all_logits),
        "probabilities": torch.cat(all_probs),
    }
    all_targets_tensor = torch.cat(all_targets)
    metrics = calculate_metrics(all_outputs, all_targets_tensor)

    pred_probs = all_outputs["probabilities"].cpu().numpy().astype(np.float64).flatten()
    true_labels = all_targets_tensor.cpu().numpy().astype(np.float64).flatten()

    return avg_loss, metrics, all_ids, true_labels, pred_probs

def save_predictions_csv(ids, true_labels, pred_probs, save_path, verbose=True):
    df = pd.DataFrame({
        'ID': ids,
        'True_Label': true_labels,
        'Predicted_Score': pred_probs
    })
    df.to_csv(save_path, index=False)
    if verbose:
        print(f"Predictions saved to: {save_path} (n={len(df)})")
    return df

# ================= SHAP utilities =================
def compute_shap_for_validation(model, train_loader, val_loader, full_dataset, device, args, fold):
    """Compute and save SHAP values for the validation split of one fold."""
    if not SHAP_AVAILABLE:
        print("WARNING: shap is not installed. SHAP computation is skipped. Please run: pip install shap")
        return

    os.makedirs(args.shap_output_dir, exist_ok=True)

    # Sample background data from the training split.
    train_indices = np.array(train_loader.dataset.indices, dtype=np.int64)
    X_gut_train = full_dataset.X_gut[train_indices].numpy()
    n_bg = min(args.shap_background_samples, len(X_gut_train))
    np.random.seed(args.seed + fold)
    idx_bg = np.random.choice(len(X_gut_train), n_bg, replace=False)
    background = X_gut_train[idx_bg]

    # Wrap the model to explain x_gut while fixing metadata to zero.
    # To explain metadata as well, concatenate gut and metadata inputs before calling the model.
    metadata_dim = full_dataset.X_metadata.shape[1]

    class WrappedModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            meta = torch.zeros(x.shape[0], metadata_dim, device=x.device)
            return self.base_model(x, meta)["probabilities"].squeeze(-1)

    wrapped_model = WrappedModel(model).to(device)
    wrapped_model.eval()

    def predict_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).to(device)
            probs = wrapped_model(x_t).cpu().numpy()
        return probs

    # Initialize KernelExplainer.
    explainer = shap.KernelExplainer(predict_fn, background)

    # Collect validation samples.
    val_indices = np.array(val_loader.dataset.indices, dtype=np.int64)
    val_ids = [str(full_dataset.subject_ids[i]) for i in val_indices]
    X_gut_val = full_dataset.X_gut[val_indices].numpy()
    y_val = full_dataset.y[val_indices].numpy().astype(float).flatten()
    species_names = list(full_dataset.info['species_names'])
    n_samples = len(X_gut_val)

    # Compute validation prediction scores for sample-level tracking.
    with torch.no_grad():
        x_val_t = torch.tensor(X_gut_val, dtype=torch.float32).to(device)
        meta_zero = torch.zeros(x_val_t.shape[0], metadata_dim, device=device)
        pred_scores = model(x_val_t, meta_zero)["probabilities"].detach().cpu().numpy().astype(float).flatten()

    # Compute SHAP values sample by sample.
    shap_values_list = []
    print(f"  Start computing SHAP values for {n_samples} validation samples (nsamples={args.shap_nsamples})...")

    for i, sample in enumerate(X_gut_val):
        target = sample.reshape(1, -1)
        shap_vals = explainer.shap_values(target, nsamples=args.shap_nsamples)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = np.asarray(shap_vals).squeeze()
        shap_values_list.append(shap_vals)

        if (i + 1) % 10 == 0 or (i + 1) == n_samples:
            print(f"    Processed {i + 1}/{n_samples} samples")

    shap_matrix = np.asarray(shap_values_list)  # shape: (n_samples, gut_dim)

    # Save mean SHAP values for each feature in this fold.
    mean_shap = np.mean(shap_matrix, axis=0)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    df_mean = pd.DataFrame({
        'fold': fold,
        'feature': species_names,
        'mean_shap': mean_shap,
        'mean_abs_shap': mean_abs_shap,
    })
    df_mean = df_mean.sort_values('mean_abs_shap', ascending=False)

    mean_path = os.path.join(args.shap_output_dir, f"fold_{fold}_shap_mean.csv")
    df_mean.to_csv(mean_path, index=False)
    print(f"  Mean SHAP values saved to: {mean_path}")

    # Save per-sample, per-feature SHAP values for beeswarm plots.
    rows = []
    for i, sid in enumerate(val_ids):
        for j, fname in enumerate(species_names):
            rows.append({
                'fold': fold,
                'sample_id': sid,
                'true_label': y_val[i],
                'predicted_score': pred_scores[i],
                'feature': fname,
                'feature_value': X_gut_val[i, j],
                'shap_value': shap_matrix[i, j],
            })

    df_long = pd.DataFrame(rows)
    long_path = os.path.join(args.shap_output_dir, f"fold_{fold}_shap_long.csv")
    df_long.to_csv(long_path, index=False)

    df_wide = pd.DataFrame(shap_matrix, columns=species_names)
    df_wide.insert(0, 'predicted_score', pred_scores)
    df_wide.insert(0, 'true_label', y_val)
    df_wide.insert(0, 'sample_id', val_ids)
    df_wide.insert(0, 'fold', fold)

    wide_path = os.path.join(args.shap_output_dir, f"fold_{fold}_shap_wide.csv")
    df_wide.to_csv(wide_path, index=False)

    print(f"  Per-sample SHAP values saved to:\n    {long_path}\n    {wide_path}")


def combine_and_plot_shap(args):
    """Merge fold-level SHAP files and optionally draw a beeswarm plot."""
    if not SHAP_AVAILABLE:
        print("WARNING: shap is not installed. SHAP summary plotting is skipped.")
        return

    shap_files = [
        os.path.join(args.shap_output_dir, f"fold_{f}_shap_long.csv")
        for f in range(1, args.cv_folds + 1)
    ]
    shap_files = [f for f in shap_files if os.path.exists(f)]

    if len(shap_files) == 0:
        print("No fold_x_shap_long.csv files were found. SHAP merging and plotting are skipped.")
        return

    all_shap_df = pd.concat([pd.read_csv(f) for f in shap_files], axis=0, ignore_index=True)
    all_shap_path = os.path.join(args.shap_output_dir, "all_folds_shap_long.csv")
    all_shap_df.to_csv(all_shap_path, index=False)
    print(f"All per-sample SHAP values across folds saved to: {all_shap_path}")

    # Convert long-format data into matrices required by shap.summary_plot.
    X = all_shap_df.pivot_table(
        index=["fold", "sample_id"],
        columns="feature",
        values="feature_value",
        aggfunc="first"
    )

    S = all_shap_df.pivot_table(
        index=["fold", "sample_id"],
        columns="feature",
        values="shap_value",
        aggfunc="first"
    )

    # Keep sample and feature orders aligned.
    X = X.sort_index()
    S = S.reindex(index=X.index, columns=X.columns)

    shap_values = S.values
    feature_values = X.values
    feature_names = list(X.columns)

    # Save matrices for later plotting or inspection.
    shap_matrix_path = os.path.join(args.shap_output_dir, "all_folds_shap_matrix.csv")
    feature_matrix_path = os.path.join(args.shap_output_dir, "all_folds_feature_matrix.csv")

    S.reset_index().to_csv(shap_matrix_path, index=False)
    X.reset_index().to_csv(feature_matrix_path, index=False)

    print(f"SHAP matrix saved to: {shap_matrix_path}")
    print(f"Feature value matrix saved to: {feature_matrix_path}")

    if args.plot_shap_summary:
        import matplotlib.pyplot as plt

        plt.figure()
        shap.summary_plot(
            shap_values,
            features=feature_values,
            feature_names=feature_names,
            max_display=args.shap_max_display,
            show=False
        )
        fig = plt.gcf()
        fig.tight_layout()

        png_path = os.path.join(args.shap_output_dir, "all_folds_shap_summary_beeswarm.png")
        pdf_path = os.path.join(args.shap_output_dir, "all_folds_shap_summary_beeswarm.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"SHAP beeswarm plot saved to:\n  {png_path}\n  {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="ASC classification with cross-validation, per-sample SHAP export, and summary plotting")

    # Data arguments
    parser.add_argument("--abundance_csv", type=str, default='XXX', help="Path to the CSV file containing species abundance data")
    parser.add_argument("--meta_csv", type=str, default='XXX', help="Path to the CSV file containing metadata")
    parser.add_argument("--important_species_txt", type=str, default='XXX', help="Path to the text file listing important species")
    parser.add_argument("--metadata_select_txt", type=str, default='XXX', help="Path to the text file containing selected metadata columns")
    parser.add_argument("--disease_col_name", type=str, default="XXX", help="Column name in the metadata CSV that contains disease labels")
    parser.add_argument("--disease_mapping", type=str, default="XXX", help="Mapping of disease labels to integers, e.g., 'Normal Range:0,Elevated Range:1'")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_metadata", action="store_true", default=True)

    # Model arguments
    parser.add_argument("--gut_latent_dim", type=int, default=64)
    parser.add_argument("--metadata_latent_dim", type=int, default=64)
    parser.add_argument("--classifier_hidden_dims", type=str, default="128,64,32")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_batch_norm", action="store_true", default=False)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=20)

    # Class imbalance arguments
    parser.add_argument("--class_weight_0", type=float, default=1.0)
    parser.add_argument("--class_weight_1", type=float, default=1.0)

    # Cross-validation arguments
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_shuffle", action="store_true", default=True)
    parser.add_argument("--train_final_model", action="store_true", default=False)

    # Logging and saving arguments
    parser.add_argument("--save_dir", type=str, default="./checkpoints_species_cls_5_test")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="asc-classification")
    parser.add_argument("--wandb_name", type=str, default="gut-metadata-classifier")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # SHAP arguments
    parser.add_argument("--compute_shap", action="store_true", default=True,
                        help="Whether to compute SHAP values for validation samples")
    parser.add_argument("--shap_background_samples", type=int, default=100,
                        help="Number of background samples for SHAP")
    parser.add_argument("--shap_nsamples", type=int, default=100,
                        help="Number of KernelExplainer samples; larger values are more accurate but slower")
    parser.add_argument("--shap_output_dir", type=str, default="./shap_results",
                        help="Directory for SHAP outputs")
    parser.add_argument("--shap_save_detail", action="store_true", default=True,
                        help="Whether to save per-sample, per-feature SHAP values")
    parser.add_argument("--plot_shap_summary", action="store_true", default=True,
                        help="Whether to draw a beeswarm summary plot after merging SHAP results across folds")
    parser.add_argument("--shap_max_display", type=int, default=20,
                        help="Maximum number of features shown in the SHAP beeswarm plot")

    args = parser.parse_args()

    # Parse label mapping
    disease_mapping = {}
    if args.disease_mapping:
        try:
            for pair in args.disease_mapping.split(','):
                label, value = pair.split(':')
                disease_mapping[label.strip()] = int(value.strip())
        except ValueError:
            print(f"WARNING: failed to parse disease mapping '{args.disease_mapping}'. Using the default mapping")
            disease_mapping = {'Normal Range': 0, 'Elevated Range': 1}

    print(f"Disease label mapping: {disease_mapping}")
    print(f"Use metadata: {args.use_metadata}")

    args.classifier_hidden_dims = [int(x) for x in args.classifier_hidden_dims.split(",")]

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    cv_dir = os.path.join(args.save_dir, "cv_results")
    os.makedirs(cv_dir, exist_ok=True)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    print("Loading the full dataset...")
    full_dataset = ASCClassificationDataset(
        abundance_csv=args.abundance_csv,
        meta_csv=args.meta_csv,
        important_species_txt=args.important_species_txt,
        metadata_select_txt=args.metadata_select_txt if args.use_metadata else None,
        disease_col_name=args.disease_col_name,
        disease_mapping=disease_mapping,
        transform_clr=True,
        clr_eps=1e-6,
        remove_missing_disease=True,
        remove_disease_3=True,
        use_metadata=args.use_metadata,
        device=None,
    )

    y_full = full_dataset.y.numpy().astype(int)
    info = getattr(full_dataset, "info", {})
    info.update({
        "n_samples": len(full_dataset),
        "gut_dim": full_dataset.X_gut.shape[1],
        "metadata_dim": full_dataset.X_metadata.shape[1] if args.use_metadata else 0,
    })

    print("\nDataset information:")
    print(f"  Number of samples: {info['n_samples']}")
    print(f"  Number of selected species: {info['gut_dim']}")
    print(f"  Metadata dimension: {info['metadata_dim']}")
    print(f"  Class distribution: 0={np.sum(y_full==0)}, 1={np.sum(y_full==1)}")

    skf = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=args.cv_shuffle,
        random_state=args.seed if args.cv_shuffle else None
    )

    fold_metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_full)), y_full), start=1):
        print("\n" + "="*50)
        print(f"Fold {fold}/{args.cv_folds}")
        print("="*50)

        train_idx = np.array(train_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, drop_last=False
        )

        model = ASCClassifier(
            gut_dim=info['gut_dim'],
            metadata_dim=info['metadata_dim'],
            gut_latent_dim=args.gut_latent_dim,
            metadata_latent_dim=args.metadata_latent_dim,
            classifier_hidden_dims=args.classifier_hidden_dims,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm,
            use_metadata=args.use_metadata,
        ).to(device)

        pos_weight = torch.tensor([args.class_weight_1 / max(args.class_weight_0, 1e-8)]).to(device)
        criterion = ASCClassificationLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=False)

        best_val_auc = -1.0
        patience_counter = 0
        best_checkpoint_path = None

        for epoch in range(1, args.epochs + 1):
            print(f"\nFold {fold} - Epoch {epoch}/{args.epochs}")

            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args)
            val_loss, val_metrics, val_ids, val_true, val_probs = validate_with_ids(model, val_loader, criterion, device, args)

            scheduler.step(val_metrics['auc'])

            print(f"Training - loss: {train_loss:.4f}, AUC: {train_metrics['auc']:.6f}")
            print(f"Validation - loss: {val_loss:.4f}, AUC: {val_metrics['auc']:.6f} | "
                  f"Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, "
                  f"Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"fold_{fold}/epoch": epoch,
                    f"fold_{fold}/train_loss": train_loss,
                    f"fold_{fold}/val_loss": val_loss,
                    f"fold_{fold}/train_auc": train_metrics['auc'],
                    f"fold_{fold}/val_auc": val_metrics['auc'],
                    f"fold_{fold}/val_accuracy": val_metrics['accuracy'],
                    f"fold_{fold}/val_precision": val_metrics['precision'],
                    f"fold_{fold}/val_recall": val_metrics['recall'],
                    f"fold_{fold}/val_f1": val_metrics['f1'],
                })

            if val_metrics['auc'] > best_val_auc + 1e-12:
                best_val_auc = float(val_metrics['auc'])
                patience_counter = 0

                fold_model_path = os.path.join(cv_dir, f"fold_{fold}_best.pt")

                cpu_state = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                             for k, v in model.state_dict().items()}

                fold_pred_path_saved = os.path.join(cv_dir, f"fold_{fold}_predictions_saved_at_epoch_{epoch}.csv")
                _ = save_predictions_csv(val_ids, val_true, val_probs, fold_pred_path_saved, verbose=False)

                checkpoint = {
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': cpu_state,
                    'val_auc': best_val_auc,
                    'val_metrics': val_metrics,
                    'val_ids': val_ids,
                    'val_true': val_true,
                    'val_probs': val_probs,
                    'val_idx': val_idx.tolist(),
                    'args': vars(args),
                }
                torch.save(checkpoint, fold_model_path)
                best_checkpoint_path = fold_model_path

                print(f"  Saved the best model to {fold_model_path}, AUC: {best_val_auc:.6f}")
                print(f"  Validation CSV at the current epoch saved to: {fold_pred_path_saved}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping triggered after {args.patience} epochs without improvement")
                    break

        print(f"\nFold {fold} completed. Best validation AUC (recorded): {best_val_auc:.6f}")

        if best_checkpoint_path is None:
            print("  WARNING: no best model was saved for this fold. Skipping this fold.")
            continue

        # Reload the best checkpoint and recompute validation metrics.
        ck = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print(f"  Loaded model from checkpoint: fold={ck.get('fold')}, epoch={ck.get('epoch')}, recorded val_auc={ck.get('val_auc'):.6f}")

        ck_val_idx = np.array(ck['val_idx'], dtype=np.int64)
        val_subset_re = torch.utils.data.Subset(full_dataset, ck_val_idx)
        val_loader_re = torch.utils.data.DataLoader(
            val_subset_re, batch_size=args.batch_size, shuffle=False,
            num_workers=0, drop_last=False
        )

        val_loss_re, val_metrics_re, val_ids_re, val_true_re, val_probs_re = validate_with_ids(
            model, val_loader_re, criterion, device, args
        )

        print("  Recomputed validation metrics:")
        print(f"    val_auc (recomputed): {val_metrics_re['auc']:.6f}")
        print(f"    val_auc (saved in ckpt): {ck.get('val_auc'):.6f}")
        print(f"    saved ids len: {len(ck.get('val_ids', []))}, recomputed ids len: {len(val_ids_re)}")

        saved_df = pd.DataFrame({'ID': ck['val_ids'], 'True': ck['val_true'], 'Prob_saved': ck['val_probs']})
        recomputed_df = pd.DataFrame({'ID': val_ids_re, 'True_re': val_true_re, 'Prob_re': val_probs_re})
        merged = saved_df.merge(recomputed_df, on='ID', how='inner')
        if not merged.empty:
            max_abs_err = np.max(np.abs(merged['Prob_saved'].to_numpy() - merged['Prob_re'].to_numpy()))
            print(f"    Merged sample count: {len(merged)}, maximum absolute probability error: {max_abs_err:.8f}")
            try:
                auc_re_merged = roc_auc_score(merged['True'].to_numpy().astype(int), merged['Prob_re'].to_numpy())
                print(f"    AUC after ID-based merge: {auc_re_merged:.6f}")
            except Exception as e:
                print(f"    Failed to compute merged AUC: {e}")
        else:
            print("    WARNING: ID-based merge is empty, indicating inconsistent ID sets.")

        fold_pred_path_re = os.path.join(cv_dir, f"fold_{fold}_predictions_recomputed.csv")
        save_predictions_csv(val_ids_re, val_true_re, val_probs_re, fold_pred_path_re, verbose=True)

        # Compute SHAP values for validation samples in this fold.
        if args.compute_shap:
            print("  Start computing SHAP values for validation samples...")
            compute_shap_for_validation(
                model, train_loader, val_loader_re, full_dataset, device, args, fold
            )

        # Record fold summary.
        fold_metrics = {
            'fold': fold,
            'best_val_auc': float(ck.get('val_auc', best_val_auc)),
            'val_loss': float(val_loss_re),
            'val_accuracy': float(val_metrics_re['accuracy']),
            'val_auc': float(val_metrics_re['auc']),
            'val_precision': float(val_metrics_re['precision']),
            'val_recall': float(val_metrics_re['recall']),
            'val_f1': float(val_metrics_re['f1']),
            'val_tp': int(val_metrics_re['tp']),
            'val_tn': int(val_metrics_re['tn']),
            'val_fp': int(val_metrics_re['fp']),
            'val_fn': int(val_metrics_re['fn']),
        }
        fold_metrics_list.append(fold_metrics)

    # Cross-validation summary
    print("\n" + "="*50)
    print("Cross-validation summary")
    print("="*50)
    metrics_df = pd.DataFrame(fold_metrics_list)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
        print("\nMean metrics (mean ± standard deviation):")
        for col in ['val_auc', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']:
            mean_val = metrics_df[col].mean()
            std_val = metrics_df[col].std()
            print(f"{col:15s}: {mean_val:.4f} ± {std_val:.4f}")

        summary_path = os.path.join(args.save_dir, "cv_summary.csv")
        metrics_df.to_csv(summary_path, index=False)
        print(f"\nCross-validation summary saved to: {summary_path}")
    else:
        print("No fold metrics are available. All folds may have failed to save a model.")

    # Merge SHAP results across folds and draw the beeswarm plot.
    if args.compute_shap:
        combine_and_plot_shap(args)

    # Optional final model.
    if args.train_final_model:
        print("\n" + "="*50)
        print("Training the final model on all data")
        print("="*50)
        full_loader = torch.utils.data.DataLoader(
            full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        final_model = ASCClassifier(
            gut_dim=info['gut_dim'],
            metadata_dim=info['metadata_dim'],
            gut_latent_dim=args.gut_latent_dim,
            metadata_latent_dim=args.metadata_latent_dim,
            classifier_hidden_dims=args.classifier_hidden_dims,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm,
            use_metadata=args.use_metadata,
        ).to(device)
        optimizer = optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=False)
        best_final_auc = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_epoch(final_model, full_loader, criterion, optimizer, device, args)
            print(f"Epoch {epoch}: training loss {train_loss:.4f}, training AUC {train_metrics['auc']:.6f}")
            if train_metrics['auc'] > best_final_auc + 1e-12:
                best_final_auc = float(train_metrics['auc'])
        final_model_path = os.path.join(args.save_dir, "final_model.pt")
        torch.save({
            'model_state_dict': {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                                 for k, v in final_model.state_dict().items()},
            'args': vars(args),
            'info': info,
        }, final_model_path)
        print(f"Final model saved to: {final_model_path}")

    print("\nAll tasks completed！")

if __name__ == "__main__":
    main()