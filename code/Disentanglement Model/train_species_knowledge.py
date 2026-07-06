# networks/species/train_species.py
import os
import argparse
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import sys

from Disentangle_knowledge import (
    ImprovedCompositeModelWithAttention,
    MultiHeadAdversary,
    orthogonality_loss_cosine,
    sparse_loss_delta,
    reconstruction_loss,
    compute_metrics,
    weighted_reconstruction_loss,
    extract_labels_from_C,
    positive_prior_loss,
    negative_prior_loss,
    neutral_prior_loss,
    prior_group_sparsity_loss,
)

from dataloader_knowledge import build_dataloaders

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def build_model(info, device, args):
    x_dim = info["X_dim"]; c_dim = info["C_dim"]
    cont_dim = info["cont_dim"]; cont_miss_dim = info["cont_miss_dim"]
    num_disc_dim = info["num_disc_dim"]; onehot_disc_dim = info["onehot_disc_dim"]
    c_names = info["c_names"]
    disease_dim = info["disease_dim"]; others_dim = info["others_dim"]

    adversary = None
    if args.use_adversary and c_dim > 0 and (onehot_disc_dim > 0 or num_disc_dim > 0):
        print(f"Building adversary: onehot_dim={onehot_disc_dim}, num_disc_dim={num_disc_dim}")
        adversary = MultiHeadAdversary(
            z_dim=args.z_dim, c_names=c_names, cont_dim=cont_dim, cont_miss_dim=cont_miss_dim,
            num_disc_dim=num_disc_dim, onehot_disc_dim=onehot_disc_dim,
            hidden_dims=[128, 64], dropout=args.dropout
        ).to(device)

    # Load prior information
    prior_weight = torch.from_numpy(info.get("prior_weight", np.zeros(info["X_dim"], dtype=np.float32))).to(device)
    pos_mask = torch.from_numpy(info.get("prior_pos_mask", np.zeros(info["X_dim"], dtype=np.float32))).to(device)
    neg_mask = torch.from_numpy(info.get("prior_neg_mask", np.zeros(info["X_dim"], dtype=np.float32))).to(device)
    neu_mask = torch.from_numpy(info.get("prior_neu_mask", np.zeros(info["X_dim"], dtype=np.float32))).to(device)
    
    prior_info = {
        'prior_weight': prior_weight,
        'pos_mask': pos_mask.cpu().numpy(),
        'neg_mask': neg_mask.cpu().numpy(),
        'neu_mask': neu_mask.cpu().numpy(),
    }

    model = ImprovedCompositeModelWithAttention(
        x_dim=x_dim, c_dim=c_dim, disease_dim=disease_dim, others_dim=others_dim,
        z_dim=args.z_dim, u_dim=args.u_dim,
        enc_hidden_dims=[args.hid, args.hid // 2],
        dec_hidden_dims=[args.hid // 2, args.hid],
        dropout=args.dropout, shift_dropout=args.shift_dropout,
        grl_lambda=args.grl_lambda, adversary=adversary,
        prior_info=prior_info,
        use_attention=args.use_prior_attention,
        attention_heads=args.attention_heads,
        attn_beta=args.attn_beta,
        use_unknown=args.use_unknown,
    ).to(device)

    print(f"Model built: X_dim={x_dim}, C_dim={c_dim}, disease_dim={disease_dim}, others_dim={others_dim}")
    if args.use_prior_attention:
        print(f"Prior attention enabled, heads: {args.attention_heads}, beta={args.attn_beta}")
    print(f"Unknown residual module: {'enabled' if args.use_unknown else 'disabled'}")
    
    return model, adversary, prior_weight, pos_mask, neg_mask, neu_mask

def topk_prior_diagnostics(delta_d, prior_mask_any, k=30):
    """Compute the fraction of prior species among the top-k absolute delta_d features."""
    if prior_mask_any is None or prior_mask_any.numel() == 0:
        return 0.0
    with torch.no_grad():
        B, X = delta_d.shape
        has_prior = (prior_mask_any > 0)
        if has_prior.sum() == 0:
            return 0.0
        hit_rates = []
        for b in range(B):
            d = delta_d[b]
            top_idx = torch.topk(d.abs(), k=min(k, X)).indices
            hit = has_prior[top_idx].float().mean().item()
            hit_rates.append(hit)
        return float(np.mean(hit_rates))

def train_phase1_known_only(model, adversary, train_loader, opt_main, opt_adv, device, args, epoch, info,
                          prior_weight, pos_mask, neg_mask, neu_mask):
    model.train()
    if adversary is not None:
        adversary.train()

    disease_start = info["disease_start"]; disease_dim = info["disease_dim"]
    lambda_adv = args.lambda_adv * 0.5 if epoch > args.adv_warmup_epochs else 0.0
    class_weights = torch.tensor([args.class_weight_0, args.class_weight_1], device=device)

    meters = {
        "total": 0.0, "recon": 0.0,
        "ortho_base_d": 0.0, "ortho_base_o": 0.0, "ortho_d_o": 0.0,
        "sparse_d": 0.0, "sparse_o": 0.0,
        "adv_total": 0.0,
        "pos_prior": 0.0,
        "neg_prior": 0.0,
        "neu_prior": 0.0,
        "group_sparsity": 0.0,
        "prior_hit@k": 0.0,
    }
    n_samples = 0

    pbar = tqdm(train_loader, desc=f"Phase1 Epoch {epoch} [Known Only]")
    for batch_idx, batch in enumerate(pbar):
        x = batch["x_clr"].to(device)
        C = batch["c"].to(device)
        bs = x.size(0)
        n_samples += bs
        labels = extract_labels_from_C(C, disease_start, disease_dim).long().to(device)

        if adversary is not None and batch_idx % args.adv_update_freq == 0 and lambda_adv > 0:
            for p in model.parameters(): p.requires_grad = False
            for p in adversary.parameters(): p.requires_grad = True
            opt_adv.zero_grad()
            with torch.no_grad():
                z_base = model.intrinsic_encoder(x)
            adv_losses = adversary.compute_losses(z_base, C, device)
            adv_loss = adv_losses["adv_total"]
            if adv_loss.item() > 0:
                adv_loss.backward()
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), args.grad_clip)
                opt_adv.step()

        for name, param in model.named_parameters():
            if 'unknown_factor_encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        if adversary is not None:
            for p in adversary.parameters(): p.requires_grad = False

        opt_main.zero_grad()
        out = model(x, C, disease_start=disease_start, disease_dim=disease_dim, training_stage="known_only")

        x_base, delta_d, delta_o = out["x_base"], out["delta_d"], out["delta_o"]
        x_hat = out["x_hat"]

        recon_loss = weighted_reconstruction_loss(
            x, x_hat, class_weights, labels,
            mse_weight=args.mse_weight, mae_weight=args.mae_weight
        )
        total_loss = recon_loss

        l_bd = orthogonality_loss_cosine(x_base, delta_d)
        l_bo = orthogonality_loss_cosine(x_base, delta_o)
        l_do = orthogonality_loss_cosine(delta_d, delta_o)
        total_loss += args.lambda_ortho_base_d * l_bd + args.lambda_ortho_base_o * l_bo + args.lambda_ortho_d_o * l_do
        meters["ortho_base_d"] += l_bd.item() * bs
        meters["ortho_base_o"] += l_bo.item() * bs
        meters["ortho_d_o"] += l_do.item() * bs

        group_sparsity = prior_group_sparsity_loss(
            delta_d, model.prior_mask_any,
            l1_prior=args.l1_prior_group, l1_nonprior=args.l1_non_group
        )
        total_loss += group_sparsity
        meters["group_sparsity"] += group_sparsity.item() * bs

        sparse_o = sparse_loss_delta(delta_o, l1_weight=args.lambda_l1, l2_weight=args.lambda_l2)
        total_loss += args.lambda_sparse_o * sparse_o
        meters["sparse_o"] += sparse_o.item() * bs

        if args.lambda_pos_prior_phase1 > 0 and pos_mask.sum() > 0:
            # pos_loss = positive_prior_loss(
            #     delta_d, delta_o, x_base, labels,
            #     pos_mask, margin=args.pos_margin
            # )
            pos_loss = positive_prior_loss(delta_d, labels, pos_mask, margin=args.pos_margin)
            total_loss += args.lambda_pos_prior_phase1 * pos_loss
            meters["pos_prior"] += pos_loss.item() * bs
            
        if args.lambda_neg_prior_phase1 > 0 and neg_mask.sum() > 0:
            # neg_loss = negative_prior_loss(
            #     delta_d, delta_o, x_base, labels,
            #     neg_mask, margin=args.neg_margin
            # )
            neg_loss = negative_prior_loss(delta_d, labels, neg_mask, margin=args.neg_margin)
            total_loss += args.lambda_neg_prior_phase1 * neg_loss
            meters["neg_prior"] += neg_loss.item() * bs
            
        if args.lambda_neu_prior_phase1 > 0 and neu_mask.sum() > 0:
            # neu_loss = neutral_prior_loss(
            #     delta_d, delta_o, x_base, labels,
            #     neu_mask, margin=args.neu_margin
            # )
            neu_loss = neutral_prior_loss(delta_d, labels, neu_mask, margin=args.neu_margin)
            total_loss += args.lambda_neu_prior_phase1 * neu_loss
            meters["neu_prior"] += neu_loss.item() * bs
    
        if adversary is not None and out.get("adv_losses") is not None and lambda_adv > 0:
            adv_ce = out["adv_losses"]["adv_ce"]
            adv_mse = out["adv_losses"]["adv_mse"]
            total_loss += lambda_adv * (adv_ce + adv_mse)
            meters["adv_total"] += (adv_ce + adv_mse).item() * bs

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt_main.step()

        hitk = topk_prior_diagnostics(delta_d.detach(), model.prior_mask_any.detach(), k=args.topk_diag)
        meters["prior_hit@k"] += hitk * bs

        meters["total"] += total_loss.item() * bs
        meters["recon"] += recon_loss.item() * bs

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}", 
            "recon": f"{recon_loss.item():.4f}",
            "pos": f"{pos_loss.item() if 'pos_loss' in locals() else 0:.4f}",
            "neg": f"{neg_loss.item() if 'neg_loss' in locals() else 0:.4f}",
            "neu": f"{neu_loss.item() if 'neu_loss' in locals() else 0:.4f}",
            "hit@k": f"{hitk:.2f}"
        })

    for k in meters:
        meters[k] /= max(1, n_samples)
    return meters, opt_main, opt_adv

def train_phase2_add_unknown(model, adversary, train_loader, opt_unknown, device, args, epoch, info):
    if not args.use_unknown or model.unknown_factor_encoder is None:
        return {
            "total": 0.0, "recon": 0.0, "sparse_u": 0.0,
            "ortho_base_u": 0.0, "ortho_d_u": 0.0, "ortho_o_u": 0.0
        }, opt_unknown

    model.train()
    for p in model.intrinsic_encoder.parameters(): p.requires_grad = False
    for p in model.intrinsic_decoder.parameters(): p.requires_grad = False
    for p in model.disease_shift_encoder.parameters(): p.requires_grad = False
    for p in model.other_shift_encoder.parameters(): p.requires_grad = False
    for p in model.unknown_factor_encoder.parameters(): p.requires_grad = True
    if hasattr(model, 'prior_attention') and model.prior_attention is not None:
        for p in model.prior_attention.parameters(): p.requires_grad = True

    disease_start = info["disease_start"]; disease_dim = info["disease_dim"]
    lambda_unknown_sparse = args.lambda_unknown_sparse * min(1.0, epoch / (args.phase2_epochs * 0.5))

    meters = {"total": 0.0, "recon": 0.0, "sparse_u": 0.0, 
              "ortho_base_u": 0.0, "ortho_d_u": 0.0, "ortho_o_u": 0.0}
    n_samples = 0

    pbar = tqdm(train_loader, desc=f"Phase2 Epoch {epoch} [Add Unknown]")
    for batch_idx, batch in enumerate(pbar):
        x = batch["x_clr"].to(device)
        C = batch["c"].to(device)
        bs = x.size(0)
        n_samples += bs

        opt_unknown.zero_grad()
        out = model(x, C, disease_start=disease_start, disease_dim=disease_dim, training_stage="add_unknown")

        x_hat = out["x_hat"]
        delta_u = out["delta_u"]
        
        recon_loss = reconstruction_loss(x, x_hat, mse_weight=args.mse_weight, mae_weight=args.mae_weight)
        l_bu = orthogonality_loss_cosine(out["x_base"], out["delta_u"])
        l_du = orthogonality_loss_cosine(out["delta_d"], out["delta_u"])
        l_ou = orthogonality_loss_cosine(out["delta_o"], out["delta_u"])
        total_loss = recon_loss + args.lambda_ortho_base_u * l_bu + args.lambda_ortho_d_u * l_du + args.lambda_ortho_o_u * l_ou

        sparse_u = sparse_loss_delta(delta_u, l1_weight=args.lambda_l1 * 0.5, l2_weight=args.lambda_l2 * 0.5)
        total_loss += lambda_unknown_sparse * sparse_u
        meters["sparse_u"] += sparse_u.item() * bs

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt_unknown.step()

        meters["total"] += total_loss.item() * bs
        meters["recon"] += recon_loss.item() * bs
        meters["ortho_base_u"] += l_bu.item() * bs
        meters["ortho_d_u"] += l_du.item() * bs
        meters["ortho_o_u"] += l_ou.item() * bs

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "recon": f"{recon_loss.item():.4f}",
            "sparse_u": f"{sparse_u.item():.4f}",
        })

    for k in meters:
        meters[k] /= max(1, n_samples)
    return meters, opt_unknown

def train_phase3_joint(model, adversary, train_loader, opt_joint, opt_adv, device, args, epoch, info,
                     prior_weight, pos_mask, neg_mask, neu_mask):
    model.train()
    if adversary is not None:
        adversary.train()

    disease_start = info["disease_start"]; disease_dim = info["disease_dim"]
    lambda_adv = args.lambda_adv * 0.8
    lambda_unknown_sparse = args.lambda_unknown_sparse

    class_weights = torch.tensor([args.class_weight_0, args.class_weight_1], device=device)

    meters = {
        "total": 0.0, "recon": 0.0,
        "ortho_base_d": 0.0, "ortho_base_o": 0.0, "ortho_d_o": 0.0,
        "sparse_d": 0.0, "sparse_o": 0.0, "sparse_u": 0.0,
        "adv_total": 0.0,  "ortho_base_u": 0.0, "ortho_d_u": 0.0, "ortho_o_u": 0.0,
        "pos_prior": 0.0,
        "neg_prior": 0.0,
        "neu_prior": 0.0,
        "group_sparsity": 0.0,
        "prior_hit@k": 0.0,
    }
    n_samples = 0

    pbar = tqdm(train_loader, desc=f"Phase3 Epoch {epoch} [Joint]")
    for batch_idx, batch in enumerate(pbar):
        x = batch["x_clr"].to(device)
        C = batch["c"].to(device)
        bs = x.size(0)
        n_samples += bs
        labels = extract_labels_from_C(C, disease_start, disease_dim).long().to(device)

        if adversary is not None and batch_idx % args.adv_update_freq == 0 and lambda_adv > 0:
            for p in model.parameters(): p.requires_grad = False
            for p in adversary.parameters(): p.requires_grad = True
            opt_adv.zero_grad()
            with torch.no_grad():
                z_base = model.intrinsic_encoder(x)
            adv_losses = adversary.compute_losses(z_base, C, device)
            adv_loss = adv_losses["adv_total"]
            if adv_loss.item() > 0:
                adv_loss.backward()
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), args.grad_clip)
                opt_adv.step()

        for p in model.parameters(): p.requires_grad = True
        if adversary is not None:
            for p in adversary.parameters(): p.requires_grad = False

        opt_joint.zero_grad()
        out = model(x, C, disease_start=disease_start, disease_dim=disease_dim, training_stage="full")

        x_base, delta_d, delta_o, delta_u = out["x_base"], out["delta_d"], out["delta_o"], out.get("delta_u")
        x_hat = out["x_hat"]
        
        recon_loss = weighted_reconstruction_loss(
            x, x_hat, class_weights, labels,
            mse_weight=args.mse_weight, mae_weight=args.mae_weight
        )
        total_loss = recon_loss

        l_bd = orthogonality_loss_cosine(x_base, delta_d) * args.lambda_ortho_base_d * 0.5
        l_bo = orthogonality_loss_cosine(x_base, delta_o) * args.lambda_ortho_base_o * 0.5
        l_do = orthogonality_loss_cosine(delta_d, delta_o) * args.lambda_ortho_d_o * 0.5
        if delta_u is not None:
            l_bu = orthogonality_loss_cosine(x_base, delta_u) * args.lambda_ortho_base_u * 0.5
            l_du = orthogonality_loss_cosine(delta_d, delta_u) * args.lambda_ortho_d_u * 0.5
            l_ou = orthogonality_loss_cosine(delta_o, delta_u) * args.lambda_ortho_o_u * 0.5
        else:
            l_bu = torch.tensor(0.0, device=device)
            l_du = torch.tensor(0.0, device=device)
            l_ou = torch.tensor(0.0, device=device)
        total_loss += l_bd + l_bo + l_do + l_bu + l_du + l_ou
        meters["ortho_base_d"] += l_bd.item() * bs
        meters["ortho_base_o"] += l_bo.item() * bs
        meters["ortho_d_o"] += l_do.item() * bs
        meters["ortho_base_u"] += l_bu.item() * bs
        meters["ortho_d_u"] += l_du.item() * bs
        meters["ortho_o_u"] += l_ou.item() * bs

        group_sparsity = prior_group_sparsity_loss(
            delta_d, model.prior_mask_any,
            l1_prior=args.l1_prior_group_phase3, l1_nonprior=args.l1_non_group_phase3
        )
        total_loss += group_sparsity
        meters["group_sparsity"] += group_sparsity.item() * bs

        sparse_o = sparse_loss_delta(delta_o, l1_weight=args.lambda_l1, l2_weight=args.lambda_l2) * args.lambda_sparse_o * 0.5
        total_loss += sparse_o
        meters["sparse_o"] += sparse_o.item() * bs

        if delta_u is not None:
            sparse_u = sparse_loss_delta(delta_u, l1_weight=args.lambda_l1 * 0.5, l2_weight=args.lambda_l2 * 0.5)
            total_loss += lambda_unknown_sparse * sparse_u
            meters["sparse_u"] += sparse_u.item() * bs

        if args.lambda_pos_prior_phase3 > 0 and pos_mask.sum() > 0:
            # pos_loss = positive_prior_loss(
            #     delta_d, delta_o, x_base, labels,
            #     pos_mask, margin=args.pos_margin_phase3
            # )
            pos_loss = positive_prior_loss(delta_d, labels, pos_mask, margin=args.pos_margin)
            total_loss += args.lambda_pos_prior_phase3 * pos_loss
            meters["pos_prior"] += pos_loss.item() * bs
            
        if args.lambda_neg_prior_phase3 > 0 and neg_mask.sum() > 0:
            # neg_loss = negative_prior_loss(
            #     delta_d, delta_o, x_base, labels,
            #     neg_mask, margin=args.neg_margin_phase3
            # )
            neg_loss = negative_prior_loss(delta_d, labels, neg_mask, margin=args.neg_margin)
            total_loss += args.lambda_neg_prior_phase3 * neg_loss
            meters["neg_prior"] += neg_loss.item() * bs
            
        if args.lambda_neu_prior_phase3 > 0 and neu_mask.sum() > 0:
            # neu_loss = neutral_prior_loss(
            #     delta_d, delta_o, x_base, labels,
            #     neu_mask, margin=args.neu_margin_phase3
            # )
            neu_loss = neutral_prior_loss(delta_d, labels, neu_mask, margin=args.neu_margin)
            total_loss += args.lambda_neu_prior_phase3 * neu_loss
            meters["neu_prior"] += neu_loss.item() * bs
        
        if adversary is not None and out.get("adv_losses") is not None and lambda_adv > 0:
            adv_ce = out["adv_losses"]["adv_ce"]
            adv_mse = out["adv_losses"]["adv_mse"]
            total_loss += lambda_adv * (adv_ce + adv_mse)
            meters["adv_total"] += (adv_ce + adv_mse).item() * bs

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt_joint.step()

        hitk = topk_prior_diagnostics(delta_d.detach(), model.prior_mask_any.detach(), k=args.topk_diag)
        meters["prior_hit@k"] += hitk * bs

        meters["total"] += total_loss.item() * bs
        meters["recon"] += recon_loss.item() * bs

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}", 
            "recon": f"{recon_loss.item():.4f}",
            "pos": f"{pos_loss.item() if 'pos_loss' in locals() else 0:.4f}",
            "neg": f"{neg_loss.item() if 'neg_loss' in locals() else 0:.4f}",
            "neu": f"{neu_loss.item() if 'neu_loss' in locals() else 0:.4f}",
            "hit@k": f"{hitk:.2f}"
        })

    for k in meters:
        meters[k] /= max(1, n_samples)
    return meters, opt_joint, opt_adv

@torch.no_grad()
def evaluate_model(model, val_loader, device, args, info, phase_name="phase1"):
    model.eval()
    disease_start = info["disease_start"]; disease_dim = info["disease_dim"]
    class_weights = torch.tensor([args.class_weight_0, args.class_weight_1], device=device)

    meters = {
        "total": 0.0, "recon": 0.0,
        "ortho_base_d": 0.0, "ortho_base_o": 0.0, "ortho_d_o": 0.0,
        "sparse_d": 0.0, "sparse_o": 0.0, "sparse_u": 0.0,
    }
    recon_metrics_all = {"mse": [], "mae": [], "correlation": [], "r2": []}
    n_samples = 0

    for batch in val_loader:
        x = batch["x_clr"].to(device)
        C = batch["c"].to(device)
        bs = x.size(0); n_samples += bs
        labels = extract_labels_from_C(C, disease_start, disease_dim).long().to(device)
        
        if phase_name == "phase1":
            out = model(x, C, disease_start=disease_start, disease_dim=disease_dim, training_stage="known_only")
            recon_weight = 1.0; ortho_weight = 1.0; lambda_unknown_sparse = 0.0
        elif phase_name == "phase2":
            out = model(x, C, disease_start=disease_start, disease_dim=disease_dim, training_stage="add_unknown")
            recon_weight = 1.0; ortho_weight = 0.0; lambda_unknown_sparse = args.lambda_unknown_sparse
        else:
            out = model(x, C, disease_start=disease_start, disease_dim=disease_dim, training_stage="full")
            recon_weight = 1.0; ortho_weight = 0.5; lambda_unknown_sparse = args.lambda_unknown_sparse
        
        x_base, delta_d, delta_o, x_hat = out["x_base"], out["delta_d"], out["delta_o"], out["x_hat"]
        
        recon_loss = weighted_reconstruction_loss(
            x, x_hat, class_weights, labels,
            mse_weight=args.mse_weight, mae_weight=args.mae_weight
        )
        total_loss = recon_loss * recon_weight

        if ortho_weight > 0:
            l_bd = orthogonality_loss_cosine(x_base, delta_d) * args.lambda_ortho_base_d * ortho_weight
            l_bo = orthogonality_loss_cosine(x_base, delta_o) * args.lambda_ortho_base_o * ortho_weight
            l_do = orthogonality_loss_cosine(delta_d, delta_o) * args.lambda_ortho_d_o * ortho_weight
            total_loss += l_bd + l_bo + l_do
            meters["ortho_base_d"] += l_bd.item() * bs
            meters["ortho_base_o"] += l_bo.item() * bs
            meters["ortho_d_o"] += l_do.item() * bs

        sd = sparse_loss_delta(delta_d, l1_weight=args.lambda_l1, l2_weight=args.lambda_l2) * ortho_weight
        so = sparse_loss_delta(delta_o, l1_weight=args.lambda_l1, l2_weight=args.lambda_l2) * args.lambda_sparse_o * ortho_weight
        total_loss += sd + so
        meters["sparse_d"] += sd.item() * bs
        meters["sparse_o"] += so.item() * bs

        if phase_name in ["phase2", "phase3"] and out.get("delta_u") is not None:
            delta_u = out["delta_u"]
            su = sparse_loss_delta(delta_u, l1_weight=args.lambda_l1 * 0.5, l2_weight=args.lambda_l2 * 0.5) * lambda_unknown_sparse
            total_loss += su
            meters["sparse_u"] += su.item() * bs

        meters["total"] += total_loss.item() * bs
        meters["recon"] += recon_loss.item() * bs

        batch_metrics = compute_metrics(x, x_hat)
        for k in recon_metrics_all:
            recon_metrics_all[k].append(batch_metrics[k])

    for k in meters:
        meters[k] /= max(1, n_samples)
    avg_recon = {k: (np.mean(v) if v else 0.0) for k, v in recon_metrics_all.items()}
    return meters, avg_recon

def log_metrics_to_wandb(epoch, train_m, val_m, val_rec, stage):
    log = {"epoch": epoch}
    def put(prefix, d):
        for k, v in d.items():
            log[f"{prefix}/{k}"] = v
    put(f"{stage}/train", train_m)
    put(f"{stage}/val", val_m)
    for k, v in val_rec.items():
            log[f"{stage}/val_recon/{k}"] = v
    wandb.log(log)

def main():
    parser = argparse.ArgumentParser(description="Three-stage training with species-level priors and prior attention")
    parser.add_argument("--abundance_csv", type=str, default='XXX')
    parser.add_argument("--meta_csv", type=str, default='XXX')
    parser.add_argument("--metadata_txt", type=str, default='XXX')
    parser.add_argument("--species_prior_csv", type=str, default='XXX')
    parser.add_argument("--abundance_transform", type=str, default="arcsinh",
                        choices=["arcsinh", "asinh", "clr", "none", "raw"],
                        help="Abundance transformation. Default: arcsinh.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_main", type=float, default=1e-3)
    parser.add_argument("--lr_adv", type=float, default=1e-3)
    parser.add_argument("--weight_decay_main", type=float, default=1e-5)
    parser.add_argument("--weight_decay_adv", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=20.0)
    parser.add_argument("--z_dim", type=int, default=512)
    parser.add_argument("--u_dim", type=int, default=32)
    parser.add_argument("--hid", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--shift_dropout", type=float, default=0.4)
    parser.add_argument("--use_unknown", dest="use_unknown", action="store_true", default=True,
                        help="Enable the unknown residual factor module. Enabled by default.")
    parser.add_argument("--no_unknown", dest="use_unknown", action="store_false",
                        help="Disable the unknown residual factor module and skip Phase 2.")
    parser.add_argument("--use_adversary", action="store_true", default=True)
    parser.add_argument("--grl_lambda", type=float, default=0.1)
    parser.add_argument("--lambda_adv", type=float, default=0.2)
    parser.add_argument("--adv_warmup_epochs", type=int, default=80)
    parser.add_argument("--adv_update_freq", type=int, default=1)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--mae_weight", type=float, default=0.0)
    parser.add_argument("--lambda_l1", type=float, default=0.3)
    parser.add_argument("--lambda_l2", type=float, default=0.3)
    parser.add_argument("--lambda_sparse_o", type=float, default=0.5)
    parser.add_argument("--lambda_unknown_sparse", type=float, default=0.002)
    parser.add_argument("--lambda_ortho_base_d", type=float, default=0.5)
    parser.add_argument("--lambda_ortho_base_o", type=float, default=0.5)
    parser.add_argument("--lambda_ortho_d_o", type=float, default=0.5)
    parser.add_argument("--lambda_ortho_base_u", type=float, default=0.5)
    parser.add_argument("--lambda_ortho_d_u", type=float, default=0.5)
    parser.add_argument("--lambda_ortho_o_u", type=float, default=0.5)
    parser.add_argument("--use_prior_attention", action="store_true", default=True)
    parser.add_argument("--attention_heads", type=int, default=5)
    parser.add_argument("--attn_beta", type=float, default=0.7)
    parser.add_argument("--lambda_pos_prior_phase1", type=float, default=0.4)
    parser.add_argument("--lambda_pos_prior_phase3", type=float, default=0.2)
    parser.add_argument("--pos_margin", type=float, default=0.02)
    parser.add_argument("--pos_margin_phase3", type=float, default=0.02)
    parser.add_argument("--lambda_neg_prior_phase1", type=float, default=0.4)
    parser.add_argument("--lambda_neg_prior_phase3", type=float, default=0.2)
    parser.add_argument("--neg_margin", type=float, default=0.02)
    parser.add_argument("--neg_margin_phase3", type=float, default=0.02)
    parser.add_argument("--lambda_neu_prior_phase1", type=float, default=0.4)
    parser.add_argument("--lambda_neu_prior_phase3", type=float, default=0.2)
    parser.add_argument("--neu_margin", type=float, default=0.02)
    parser.add_argument("--neu_margin_phase3", type=float, default=0.02)
    parser.add_argument("--lambda_zero_prior", type=float, default=0.5)

    parser.add_argument("--l1_prior_group", type=float, default=0.15)
    parser.add_argument("--l1_non_group", type=float, default=0.12)
    parser.add_argument("--l1_prior_group_phase3", type=float, default=0.12)
    parser.add_argument("--l1_non_group_phase3", type=float, default=0.10)
    parser.add_argument("--phase1_epochs", type=int, default=300)
    parser.add_argument("--phase2_epochs", type=int, default=300)
    parser.add_argument("--phase3_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="XXX")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="microbiome-disentangle")
    parser.add_argument("--wandb_name", type=str, default="three-phase-species-attn-three-priors")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--class_weight_0", type=float, default=1.0)
    parser.add_argument("--class_weight_1", type=float, default=1.0)
    parser.add_argument("--topk_diag", type=int, default=30)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    print("Loading data...")
    print(f"Abundance transform: {args.abundance_transform}")
    train_loader, val_loader, info = build_dataloaders(
        abundance_csv=args.abundance_csv, meta_csv=args.meta_csv, metadata_txt=args.metadata_txt,
        save_species_csv="selected_species.csv",
        species_prior_csv=args.species_prior_csv,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
        val_split=args.val_split, seed=args.seed,
        abundance_transform=args.abundance_transform, subject_intersection_only=True, prevalence_threshold=0.1,
        min_nonzero_species=20, standardize_continuous=True, include_cont_missing_indicator=True,
        numeric_discrete_mode="minmax", device=None,
    )
    print(f"Training set: {len(train_loader.dataset)}, validation set: {len(val_loader.dataset)}")
    print(f"X_dim: {info['X_dim']}, C_dim: {info['C_dim']}, disease_dim: {info['disease_dim']}")

    original_dataset = train_loader.dataset.dataset  # train_loader.dataset is a Subset
    X_raw = original_dataset.X_raw
    subject_ids = original_dataset.subject_ids
    species = original_dataset.selected_species
    df_filtered = pd.DataFrame(X_raw, index=subject_ids, columns=species)
    out_path = os.path.join(args.save_dir, "filtered_abundance.csv")
    df_filtered.to_csv(out_path)
    print(f"Saved filtered abundance table: {out_path} (samples {df_filtered.shape[0]}, species {df_filtered.shape[1]})")
    
    prior_idx_pos = info.get("prior_idx_pos", np.array([], dtype=np.int64))
    prior_idx_neg = info.get("prior_idx_neg", np.array([], dtype=np.int64))
    prior_idx_neu = info.get("prior_idx_neu", np.array([], dtype=np.int64))
    print(f"Prior species: positive={len(prior_idx_pos)}, negative={len(prior_idx_neg)}, neutral={len(prior_idx_neu)}")

    print("\nBuilding model...")
    model, adversary, prior_weight, pos_mask, neg_mask, neu_mask = build_model(info, device, args)

    print("\nBuilding optimizers...")
    known_params = [
        {"params": model.intrinsic_encoder.parameters(), "weight_decay": args.weight_decay_main},
        {"params": model.intrinsic_decoder.parameters(), "weight_decay": args.weight_decay_main},
        {"params": model.disease_shift_encoder.parameters(), "weight_decay": args.weight_decay_main * 2.0},
        {"params": model.other_shift_encoder.parameters(), "weight_decay": args.weight_decay_main * 2.0},
    ]
    if hasattr(model, 'prior_attention') and model.prior_attention is not None:
        known_params.append({"params": model.prior_attention.parameters(), "weight_decay": args.weight_decay_main * 0.5})
    
    opt_main = optim.AdamW(known_params, lr=args.lr_main)
    opt_adv = optim.AdamW(adversary.parameters(), lr=args.lr_adv, weight_decay=args.weight_decay_adv) if adversary is not None else None

    if args.use_unknown and model.unknown_factor_encoder is not None:
        unknown_params = [{"params": model.unknown_factor_encoder.parameters(), "weight_decay": args.weight_decay_main * 2.0}]
        opt_unknown = optim.AdamW(unknown_params, lr=args.lr_main * 0.5)
    else:
        opt_unknown = None

    joint_params = [
        {"params": model.intrinsic_encoder.parameters(), "weight_decay": args.weight_decay_main * 0.5},
        {"params": model.intrinsic_decoder.parameters(), "weight_decay": args.weight_decay_main * 0.5},
        {"params": model.disease_shift_encoder.parameters(), "weight_decay": args.weight_decay_main},
        {"params": model.other_shift_encoder.parameters(), "weight_decay": args.weight_decay_main},
    ]
    if args.use_unknown and model.unknown_factor_encoder is not None:
        joint_params.append({"params": model.unknown_factor_encoder.parameters(), "weight_decay": args.weight_decay_main})
    if hasattr(model, 'prior_attention') and model.prior_attention is not None:
        joint_params.append({"params": model.prior_attention.parameters(), "weight_decay": args.weight_decay_main * 0.3})
    
    opt_joint = optim.AdamW(joint_params, lr=args.lr_main * 0.3)

    global_epoch = 0

    print("\n====== Phase 1: Known factors ======")
    phase1_best_R2 = -1e9
    phase1_patience = 0
    for ep in range(1, args.phase1_epochs + 1):
        global_epoch += 1
        tr_m, opt_main, opt_adv = train_phase1_known_only(
            model, adversary, train_loader, opt_main, opt_adv, device, args, global_epoch, info,
            prior_weight, pos_mask, neg_mask, neu_mask
        )
        va_m, va_rec = evaluate_model(model, val_loader, device, args, info, "phase1")
        print(f"[P1 {ep}/{args.phase1_epochs}] loss={tr_m['total']:.4f} recon={tr_m['recon']:.4f} pos={tr_m['pos_prior']:.4f} neg={tr_m['neg_prior']:.4f} neu={tr_m['neu_prior']:.4f} | Val R2={va_rec['r2']:.4f}")
        if args.use_wandb:
            log_metrics_to_wandb(global_epoch, tr_m, va_m, va_rec, stage="phase1")
        if va_rec["r2"] > phase1_best_R2:
            phase1_best_R2 = va_rec["r2"]; phase1_patience = 0
            state = {
                "epoch": global_epoch, "model_state": model.state_dict(),
                "adversary_state": adversary.state_dict() if adversary else None,
                "info": info, "args": vars(args), "phase": "phase1", "val_r2": va_rec["r2"],
            }
            p = os.path.join(args.save_dir, "best_phase1.pt")
            torch.save(state, p)
            print(f"   -> Saved best Phase 1 model: R²={va_rec['r2']:.4f}")
        else:
            phase1_patience += 1
            if phase1_patience >= args.patience:
                print(f"Phase 1 early stopping: {phase1_patience}epochs without improvement")
                break

    print("\n====== Phase 2: Unknown factor ======")
    phase1_checkpoint_path = os.path.join(args.save_dir, "best_phase1.pt")
    if os.path.exists(phase1_checkpoint_path):
        checkpoint = torch.load(phase1_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if adversary is not None and checkpoint.get('adversary_state') is not None:
            adversary.load_state_dict(checkpoint['adversary_state'])
        print(f"Loaded best Phase 1 model (epoch {checkpoint['epoch']}, R²={checkpoint.get('val_r2', 'N/A'):.4f})")
    else:
        print("Warning: best Phase 1 checkpoint not found. Continuing with current parameters.")

    if args.use_unknown:
        phase2_best_R2 = -1e9
        phase2_patience = 0
        for ep in range(1, args.phase2_epochs + 1):
            global_epoch += 1
            tr_m, opt_unknown = train_phase2_add_unknown(model, adversary, train_loader, opt_unknown, device, args, global_epoch, info)
            va_m, va_rec = evaluate_model(model, val_loader, device, args, info, "phase2")
            print(f"[P2 {ep}/{args.phase2_epochs}] loss={tr_m['total']:.4f} recon={tr_m['recon']:.4f} sparse_u={tr_m['sparse_u']:.4f} | Val R2={va_rec['r2']:.4f}")
            if args.use_wandb:
                log_metrics_to_wandb(global_epoch, tr_m, va_m, va_rec, stage="phase2")
            if va_rec["r2"] > phase2_best_R2:
                phase2_best_R2 = va_rec["r2"]; phase2_patience = 0
                state = {
                    "epoch": global_epoch, "model_state": model.state_dict(),
                    "adversary_state": adversary.state_dict() if adversary else None,
                    "info": info, "args": vars(args), "phase": "phase2", "val_r2": va_rec["r2"],
                }
                p = os.path.join(args.save_dir, "best_phase2.pt")
                torch.save(state, p)
                print(f"   -> Saved best Phase 2 model: R²={va_rec['r2']:.4f}")
            else:
                phase2_patience += 1
                if phase2_patience >= args.patience:
                    print(f"Phase 2 early stopping: {phase2_patience}epochs without improvement")
                    break
    else:
        print("Unknown residual module is disabled. Skipping Phase 2.")

    for p in model.parameters(): p.requires_grad = True

    print("\n====== Phase 3: Joint fine-tuning ======")
    phase2_checkpoint_path = os.path.join(args.save_dir, "best_phase2.pt")
    phase3_init_checkpoint_path = phase2_checkpoint_path if args.use_unknown else phase1_checkpoint_path
    if os.path.exists(phase3_init_checkpoint_path):
        checkpoint = torch.load(phase3_init_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if adversary is not None and checkpoint.get('adversary_state') is not None:
            adversary.load_state_dict(checkpoint['adversary_state'])
        source_phase = "Phase 2" if args.use_unknown else "Phase 1"
        print(f"Loaded best {source_phase} checkpoint (epoch {checkpoint['epoch']}, R²={checkpoint.get('val_r2', 'N/A'):.4f})")
    else:
        print("Warning: Phase 3 initialization checkpoint not found. Continuing with current parameters.")

    phase3_best_R2 = -1e9
    phase3_patience = 0
    for ep in range(1, args.phase3_epochs + 1):
        global_epoch += 1
        tr_m, opt_joint, opt_adv = train_phase3_joint(
            model, adversary, train_loader, opt_joint, opt_adv, device, args, global_epoch, info,
            prior_weight, pos_mask, neg_mask, neu_mask
        )
        va_m, va_rec = evaluate_model(model, val_loader, device, args, info, "phase3")
        print(f"[P3 {ep}/{args.phase3_epochs}] loss={tr_m['total']:.4f} recon={tr_m['recon']:.4f} pos={tr_m['pos_prior']:.4f} neg={tr_m['neg_prior']:.4f} neu={tr_m['neu_prior']:.4f} | Val R2={va_rec['r2']:.4f}")
        if args.use_wandb:
            log_metrics_to_wandb(global_epoch, tr_m, va_m, va_rec, stage="phase3")
        if va_rec["r2"] > phase3_best_R2:
            phase3_best_R2 = va_rec["r2"]; phase3_patience = 0
            state = {
                "epoch": global_epoch, "model_state": model.state_dict(),
                "adversary_state": adversary.state_dict() if adversary else None,
                "info": info, "args": vars(args), "phase": "phase3", "val_r2": va_rec["r2"],
            }
            best_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(state, best_path)
            print(f"   -> Saved best Phase 3 model: R²={va_rec['r2']:.4f}")
        else:
            phase3_patience += 1
            if phase3_patience >= args.patience:
                print(f"Phase 3 early stopping: {phase3_patience}epochs without improvement")
                break

    final_path = os.path.join(args.save_dir, "final_model.pt")
    if 'best_path' in locals() and best_path and os.path.exists(best_path):
        import shutil
        shutil.copy2(best_path, final_path)
    else:
        state = {
            "epoch": global_epoch,
            "model_state": model.state_dict(),
            "adversary_state": adversary.state_dict() if adversary else None,
            "info": info, "args": vars(args),
            "phase": "final",
        }
        torch.save(state, final_path)
    
    print(f"\nTraining complete. Best model saved to: {final_path}")

if __name__ == "__main__":
    main()
