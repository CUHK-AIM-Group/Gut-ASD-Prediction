import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def mlp(in_dim, hidden_dims: List[int], out_dim, dropout=0.0, 
        act=nn.LeakyReLU, bn=True, last_activation=False):
    layers = []
    dim = in_dim
    for i, h in enumerate(hidden_dims):
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

# ==================== Prior attention layer ====================
class PriorAttentionLayer(nn.Module):
    """
    Gated enhancement: delta = delta_raw * (1 + beta * gate).
    Gate input combines summary statistics of |delta_raw| and prior confidence.
    """
    def __init__(self, num_features: int, 
                 prior_weight: torch.Tensor = None,
                 beta: float = 0.7,
                 hidden: int = 64):
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
        # Use prior confidence as part of the gate input
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
            PriorAttentionLayer(
                num_features,
                prior_weight,
                beta=beta
            )
            for _ in range(num_heads)
        ])
        self.head_logits = nn.Parameter(torch.zeros(num_heads))
        
    def forward(self, delta_raw: torch.Tensor, x_base: torch.Tensor = None):
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
    """
    Disease encoder that outputs delta_d_raw for optional attention-based enhancement.
    Use prior confidence as a magnitude hint.
    """
    def __init__(self, disease_dim: int, output_dim: int,
                 latent_dim: int = 32, hidden_dims: List[int] = [128, 64], dropout: float = 0.3,
                 prior_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.output_dim = output_dim
        
        self.encoder = mlp(disease_dim, hidden_dims, latent_dim, dropout=dropout, last_activation=True)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, output_dim),
            # nn.ReLU()  # Ensure non-negative output for later gate enhancement
        )
        self.apply(self._init)
        
        device = prior_weight.device if prior_weight is not None else torch.device('cpu')
        self.register_buffer('prior_weight', prior_weight if prior_weight is not None else torch.zeros(output_dim, device=device))
        # Learnable confidence scaling
        self.alpha_prior = nn.Parameter(torch.full((output_dim,), 0.05))
        
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
                
    def forward(self, c_d):
        h = self.encoder(c_d)
        delta_raw = self.head(h)
        # Inject a confidence-based magnitude hint
        delta_raw = delta_raw + (self.alpha_prior * self.prior_weight)
        return delta_raw

class OtherMetadataShiftEncoder(nn.Module):
    def __init__(self, others_dim: int, output_dim: int,
                 latent_dim: int = 64, hidden_dims: List[int] = [256, 128], dropout: float = 0.4):
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
            self.apply(self._init)
        else:
            self.encoder = None
            self.head = None
            
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, c_o):
        if self.encoder is None or c_o.shape[1] == 0:
            return torch.zeros(c_o.shape[0], self.output_dim, device=c_o.device)
        return self.head(self.encoder(c_o))

class UnknownFactorEncoder(nn.Module):
    def __init__(self, x_dim: int, u_dim: int = 64,
                 hidden_dims: List[int] = [256, 128], dropout: float = 0.4):
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
        if self.delta_predictor[-1].bias is not None:
            nn.init.constant_(self.delta_predictor[-1].bias, 0)
            
    def forward(self, residual):
        u = self.encoder(residual)
        delta_u = self.delta_predictor(u)
        return u, delta_u

class MultiHeadAdversary(nn.Module):
    def __init__(
        self, z_dim: int, c_names: List[str], cont_dim: int, cont_miss_dim: int,
        num_disc_dim: int, onehot_disc_dim: int, hidden_dims: List[int] = [256, 128], dropout: float = 0.2,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_names = c_names
        self.cont_dim = cont_dim
        self.cont_miss_dim = cont_miss_dim
        self.num_disc_dim = num_disc_dim
        self.onehot_disc_dim = onehot_disc_dim
        self.feature_extractor = mlp(z_dim, hidden_dims, hidden_dims[-1], dropout=dropout, last_activation=True)
        self.onehot_offsets, self.onehot_field_sizes = self._build_onehot_field_slices(c_names)
        self.onehot_classifiers = nn.ModuleList([nn.Linear(hidden_dims[-1], n) for n in self.onehot_field_sizes])
        if num_disc_dim > 0:
            self.numeric_regressor = nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, num_disc_dim)
            )
        else:
            self.numeric_regressor = None
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def _build_onehot_field_slices(self, c_names: List[str]) -> Tuple[List[Tuple[int, int]], List[int]]:
        start = self.cont_dim + self.cont_miss_dim + self.num_disc_dim
        if self.onehot_disc_dim > 0:
            onehot_names = c_names[start: start + self.onehot_disc_dim]
        else:
            return [], []
        field_dict = {}
        for i, nm in enumerate(onehot_names):
            if "=" in nm:
                field = nm.split("=")[0]
            else:
                field = nm.split("_")[0] if "_" in nm else nm
            field_dict.setdefault(field, []).append(i)
        field_sizes, offsets, pos = [], [], 0
        for field, indices in field_dict.items():
            size = len(indices)
            field_sizes.append(size)
            offsets.append((pos, pos + size))
            pos += size
        return offsets, field_sizes
    
    def forward(self, z):
        features = self.feature_extractor(z)
        onehot_outputs = [clf(features) for clf in self.onehot_classifiers]
        numeric_output = self.numeric_regressor(features) if self.numeric_regressor is not None else None
        return onehot_outputs, numeric_output
    
    def compute_losses(self, z, C, device):
        onehot_outputs, numeric_output = self(z)
        losses = {}
        adv_ce = torch.tensor(0.0, device=device)
        if self.onehot_disc_dim > 0 and len(self.onehot_classifiers) > 0:
            start = self.cont_dim + self.cont_miss_dim + self.num_disc_dim
            C_onehot = C[:, start: start + self.onehot_disc_dim]
            ce_losses = []
            for i, (l, r) in enumerate(self.onehot_offsets):
                seg = C_onehot[:, l:r]
                valid_mask = seg.sum(dim=1) > 0.5
                if valid_mask.any():
                    target = seg[valid_mask].argmax(dim=1)
                    ce = F.cross_entropy(onehot_outputs[i][valid_mask], target, reduction='mean')
                    ce_losses.append(ce)
            if ce_losses:
                adv_ce = torch.stack(ce_losses).mean()
        adv_mse = torch.tensor(0.0, device=device)
        if self.num_disc_dim > 0 and self.numeric_regressor is not None:
            start = self.cont_dim + self.cont_miss_dim
            C_numeric = C[:, start: start + self.num_disc_dim]
            valid_mask = ~torch.isnan(C_numeric).any(dim=1)
            if valid_mask.any():
                adv_mse = F.mse_loss(numeric_output[valid_mask], C_numeric[valid_mask], reduction='mean')
        losses["adv_ce"] = adv_ce
        losses["adv_mse"] = adv_mse
        losses["adv_total"] = adv_ce + adv_mse
        return losses

# ==================== Basic loss functions ====================
def orthogonality_loss_cosine(x1, x2, eps: float = 1e-8):
    x1_norm = torch.norm(x1, dim=1, keepdim=True) + eps
    x2_norm = torch.norm(x2, dim=1, keepdim=True) + eps
    x1n, x2n = x1 / x1_norm, x2 / x2_norm
    cosine = (x1n * x2n).sum(dim=1)
    return cosine.abs().mean()

def sparse_loss_delta(delta, l1_weight: float = 1.0, l2_weight: float = 0.1):
    l1_loss = delta.abs().mean()
    l2_loss = delta.pow(2).mean()
    return l1_weight * l1_loss + l2_weight * l2_loss

def reconstruction_loss(x_true, x_pred, mse_weight: float = 0.7, mae_weight: float = 0.3):
    mse_loss = F.mse_loss(x_pred, x_true, reduction='mean')
    mae_loss = F.l1_loss(x_pred, x_true, reduction='mean')
    return mse_weight * mse_loss + mae_weight * mae_loss

def compute_metrics(x_true, x_pred):
    with torch.no_grad():
        mse = F.mse_loss(x_pred, x_true).item()
        mae = F.l1_loss(x_pred, x_true).item()
        x_true_np = x_true.cpu().numpy()
        x_pred_np = x_pred.cpu().numpy()
        correlations = []
        for i in range(x_true.shape[0]):
            corr_matrix = np.corrcoef(x_true_np[i], x_pred_np[i])
            if corr_matrix.shape[0] == 2:
                corr = corr_matrix[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        avg_correlation = np.mean(correlations) if correlations else 0.0
        ss_res = ((x_true - x_pred) ** 2).sum(dim=1)
        ss_tot = ((x_true - x_true.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        r2_per = 1 - ss_res / (ss_tot + 1e-8)
        r2 = r2_per.mean().item()
        return {"mse": mse, "mae": mae, "correlation": avg_correlation, "r2": r2}

def weighted_reconstruction_loss(x_true, x_pred, class_weights, labels, 
                                mse_weight: float = 0.7, mae_weight: float = 0.3):
    mse_per_sample = ((x_pred - x_true) ** 2).mean(dim=1)
    mae_per_sample = torch.abs(x_pred - x_true).mean(dim=1)
    loss_per_sample = mse_weight * mse_per_sample + mae_weight * mae_per_sample
    weights = class_weights[labels.long()]
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss

def extract_labels_from_C(C, disease_start: int, disease_dim: int) -> torch.Tensor:
    c_d = C[:, disease_start: disease_start + disease_dim]
    return torch.argmax(c_d, dim=1).float()


def delta_d_group_diff(delta_d, labels, eps=1e-8):
    """
    Compute the group mean difference of delta_d: ASD mean minus control mean.
    Returns [F].
    """
    y = labels.view(-1, 1).float()
    asd_mask = (y > 0.5).float()
    ctrl_mask = 1.0 - asd_mask
    
    asd_mean = (delta_d * asd_mask).sum(dim=0) / (asd_mask.sum() + eps)
    ctrl_mean = (delta_d * ctrl_mask).sum(dim=0) / (ctrl_mask.sum() + eps)
    diff = asd_mean - ctrl_mean
    return diff


def positive_prior_loss(delta_d, labels, pos_mask, margin=0.1):
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=delta_d.device)
    diff = delta_d_group_diff(delta_d, labels)
    pos_diff = diff * pos_mask.view(-1)
    loss = F.relu(margin - pos_diff) * pos_mask.view(-1)
    weighted_loss = (loss * pos_mask.view(-1)).sum() / (pos_mask.sum() + 1e-8)
    return weighted_loss


def negative_prior_loss(delta_d, labels, neg_mask, margin=0.1):
    if neg_mask.sum() == 0:
        return torch.tensor(0.0, device=delta_d.device)
    diff = delta_d_group_diff(delta_d, labels)
    neg_diff = diff * neg_mask.view(-1)
    loss = F.relu(neg_diff + margin) * neg_mask.view(-1)
    weighted_loss = (loss * neg_mask.view(-1)).sum() / (neg_mask.sum() + 1e-8)
    return weighted_loss


def neutral_prior_loss(delta_d, labels, neu_mask, margin=0.1):
    if neu_mask.sum() == 0:
        return torch.tensor(0.0, device=delta_d.device)
    diff = delta_d_group_diff(delta_d, labels)
    neu_abs = diff.abs() * neu_mask.view(-1)
    loss = F.relu(neu_abs - margin) * neu_mask.view(-1)  # Penalize values above the neutral margin.
    weighted_loss = (loss * neu_mask.view(-1)).sum() / (neu_mask.sum() + 1e-8)
    return weighted_loss

def prior_group_sparsity_loss(delta_d, prior_mask_any, 
                             l1_prior: float = 0.15, l1_nonprior: float = 0.12):
    """
    Group sparsity loss with separate L1 weights for prior and non-prior species.
    """
    pm = prior_mask_any.unsqueeze(0)
    l_prior = (delta_d.abs() * pm).mean() * l1_prior
    l_non = (delta_d.abs() * (1.0 - pm)).mean() * l1_nonprior
    return l_prior + l_non

# ==================== Composite model ====================
class ImprovedCompositeModelWithAttention(nn.Module):
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        disease_dim: int,
        others_dim: int,
        z_dim: int = 128,
        u_dim: int = 32,
        enc_hidden_dims: List[int] = [1024, 512],
        dec_hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.3,
        shift_dropout: float = 0.4,
        grl_lambda: float = 1.0,
        adversary: Optional[MultiHeadAdversary] = None,
        prior_info: dict = None,
        use_attention: bool = True,
        attention_heads: int = 5,
        attn_beta: float = 0.7,
        use_unknown: bool = True,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.use_attention = use_attention
        self.use_unknown = use_unknown

        self.intrinsic_encoder = IntrinsicEncoder(x_dim, z_dim, enc_hidden_dims, dropout)
        self.intrinsic_decoder = IntrinsicDecoder(z_dim, x_dim, dec_hidden_dims, dropout)

        # Load prior information
        prior_weight = prior_info.get('prior_weight') if prior_info is not None else None
        
        self.disease_shift_encoder = DiseaseShiftEncoder(
            disease_dim, x_dim, latent_dim=16, hidden_dims=[64,32], dropout=shift_dropout*0.8,
            prior_weight=prior_weight
        )
        
        self.other_shift_encoder = OtherMetadataShiftEncoder(
            others_dim, x_dim, latent_dim=64, hidden_dims=[256,128], dropout=shift_dropout
        )
        
        self.unknown_factor_encoder = UnknownFactorEncoder(x_dim, u_dim, [256,128], dropout) if use_unknown else None

        # Prior attention layer
        if use_attention and prior_info is not None:
            self.prior_attention = MultiHeadPriorAttention(
                x_dim, num_heads=attention_heads, prior_info={'prior_weight': prior_weight}, beta=attn_beta
            )
        else:
            self.prior_attention = None

        self.grl = GRL(grl_lambda)
        self.adversary = adversary

        self.apply(self._init_weights)

        # Cache prior information for losses
        self.register_buffer('prior_weight', prior_weight if prior_weight is not None else torch.zeros(x_dim))
        
        # Load direction masks
        pos_mask = torch.from_numpy(prior_info.get('pos_mask', np.zeros(x_dim))).to(prior_weight.device) if prior_info else torch.zeros(x_dim)
        neg_mask = torch.from_numpy(prior_info.get('neg_mask', np.zeros(x_dim))).to(prior_weight.device) if prior_info else torch.zeros(x_dim)
        neu_mask = torch.from_numpy(prior_info.get('neu_mask', np.zeros(x_dim))).to(prior_weight.device) if prior_info else torch.zeros(x_dim)
        
        self.register_buffer('pos_mask', pos_mask)
        self.register_buffer('neg_mask', neg_mask)
        self.register_buffer('neu_mask', neu_mask)
        
        # Mask for any prior
        self.register_buffer('prior_mask_any', ((pos_mask + neg_mask + neu_mask) > 0).float())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, c, disease_start: int, disease_dim: int, 
                training_stage="full", apply_attention: bool = True):
        c_d = c[:, disease_start: disease_start + disease_dim]
        c_o = torch.cat([c[:, :disease_start], c[:, disease_start + disease_dim:]], dim=1) if c.shape[1] > disease_dim else c.new_zeros(c.size(0), 0)

        z_base = self.intrinsic_encoder(x)
        x_base = self.intrinsic_decoder(z_base)

        delta_d_raw = self.disease_shift_encoder(c_d)

        attn_gate = None
        if self.use_attention and self.prior_attention is not None and apply_attention:
            delta_d, attn_gate = self.prior_attention(delta_d_raw, x_base)
        else:
            delta_d = delta_d_raw

        delta_o = self.other_shift_encoder(c_o)
        x_base = torch.abs(x_base)  # Keep baseline contributions non-negative for downstream analysis
        # print(x_base.shape, delta_d.shape, delta_o.shape)
        if training_stage == "known_only" or not self.use_unknown:
            x_hat = x_base + delta_d + delta_o
            adv_losses = None
            if self.adversary is not None and self.training:
                adv_losses = self.adversary.compute_losses(self.grl(z_base), c, device=x.device)
            return {
                "x_hat": x_hat, 
                "x_base": x_base, 
                "delta_d": delta_d, 
                "delta_d_raw": delta_d_raw,
                "delta_o": delta_o,
                "z_base": z_base,
                "delta_u": None,
                "u": None,
                "residual": None if training_stage == "known_only" else x - x_hat,
                "adv_losses": adv_losses,
                "attention_weights": attn_gate,
            }
        elif training_stage == "add_unknown":
            with torch.no_grad():
                x_base_det = x_base.detach()
                delta_d_det = delta_d.detach()
                delta_o_det = delta_o.detach()
                z_base_det = z_base.detach()
            residual = x - (x_base_det + delta_d_det + delta_o_det)
            u, delta_u = self.unknown_factor_encoder(residual)
            x_hat = x_base_det + delta_d_det + delta_o_det + delta_u
            return {
                "x_hat": x_hat, 
                "x_base": x_base_det, 
                "delta_d": delta_d_det, 
                "delta_o": delta_o_det,
                "z_base": z_base_det,
                "delta_u": delta_u,
                "u": u,
                "residual": residual,
                "adv_losses": None,
                "attention_weights": attn_gate,
            }
        else:
            residual = x - (x_base + delta_d + delta_o)
            u, delta_u = self.unknown_factor_encoder(residual)
            x_hat = x_base + delta_d + delta_o + delta_u
            adv_losses = None
            if self.adversary is not None and self.training:
                adv_losses = self.adversary.compute_losses(self.grl(z_base), c, device=x.device)
            return {
                "x_hat": x_hat, 
                "x_base": x_base, 
                "delta_d": delta_d, 
                "delta_d_raw": delta_d_raw,
                "delta_o": delta_o,
                "z_base": z_base,
                "delta_u": delta_u,
                "u": u,
                "residual": residual,
                "adv_losses": adv_losses,
                "attention_weights": attn_gate,
            }
