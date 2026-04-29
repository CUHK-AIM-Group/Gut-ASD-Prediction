import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

def mlp(in_dim, hidden_dims: List[int], out_dim, dropout=0.0, 
        act=nn.LeakyReLU, bn=True, last_activation=False):
    """通用的MLP构建函数"""
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
        if out_dim == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)

class GutEncoder(nn.Module):
    """肠道菌群特征编码器（基于关键菌种）"""
    def __init__(self, input_dim: int, latent_dim: int = 128, 
                 hidden_dims: List[int] = [256, 128], dropout: float = 0.3):
        super().__init__()
        self.encoder = mlp(
            input_dim, hidden_dims, latent_dim, 
            dropout=dropout, last_activation=True
        )
    
    def forward(self, x):
        return self.encoder(x)

class MetadataEncoder(nn.Module):
    """元数据编码器（除疾病外的其他元数据）"""
    def __init__(self, input_dim: int, latent_dim: int = 64,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.4):
        super().__init__()
        if input_dim > 0:
            self.encoder = mlp(
                input_dim, hidden_dims, latent_dim,
                dropout=dropout, last_activation=True
            )
            self.latent_dim = latent_dim
        else:
            self.encoder = None
            self.latent_dim = 0
    
    def forward(self, x):
        if self.encoder is None or x.shape[1] == 0:
            # 返回空张量，维度与batch大小一致
            return torch.zeros(x.shape[0], 0, device=x.device)
        return self.encoder(x)

class ASCClassifier(nn.Module):
    """ASC分类器 - 可选的肠道菌群和元数据特征"""
    def __init__(
        self,
        gut_dim: int,           # 关键菌种数量
        metadata_dim: int,      # 元数据维度（可为0）
        gut_latent_dim: int = 128,
        metadata_latent_dim: int = 64,
        classifier_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_metadata: bool = True,  # 新增：是否使用元数据
    ):
        super().__init__()
        
        self.use_metadata = use_metadata
        
        # 肠道菌群编码器（始终使用）
        self.gut_encoder = GutEncoder(
            input_dim=gut_dim,
            latent_dim=gut_latent_dim,
            hidden_dims=[gut_dim * 2, gut_dim],  # 自适应调整
            dropout=dropout * 0.8  # 稍低的dropout
        )
        
        # 元数据编码器（可选）
        if use_metadata and metadata_dim > 0:
            self.metadata_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                latent_dim=metadata_latent_dim,
                hidden_dims=[max(64, metadata_dim // 2), 64],
                dropout=dropout
            )
            self.total_latent_dim = gut_latent_dim + metadata_latent_dim
        else:
            self.metadata_encoder = None
            self.total_latent_dim = gut_latent_dim
            print(f"模型配置: 不使用元数据，仅使用肠道菌群特征")
        
        # 分类器
        self.classifier = mlp(
            in_dim=self.total_latent_dim,
            hidden_dims=classifier_hidden_dims,
            out_dim=1,  # 二分类输出单个值
            dropout=dropout,
            bn=use_batch_norm,
            last_activation=False  # 将在forward中使用sigmoid
        )
        
        # 注意力机制（可选，用于可视化重要特征）
        if gut_dim > 0:
            self.attention = nn.Sequential(
                nn.Linear(gut_latent_dim, gut_latent_dim // 2),
                nn.Tanh(),
                nn.Linear(gut_latent_dim // 2, gut_dim),  # 映射回原始菌种维度
                nn.Softmax(dim=1)
            )
        else:
            self.attention = None
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x_gut, x_metadata, return_attention=False):
        """
        前向传播
        
        Args:
            x_gut: 肠道菌群数据 [batch_size, gut_dim]
            x_metadata: 元数据 [batch_size, metadata_dim]
            return_attention: 是否返回注意力权重
        
        Returns:
            分类概率 [batch_size, 1]
        """
        # 编码肠道菌群特征
        gut_features = self.gut_encoder(x_gut)
        
        # 编码元数据特征（如果可用）
        if self.use_metadata and self.metadata_encoder is not None:
            metadata_features = self.metadata_encoder(x_metadata)
            # 特征融合
            combined_features = torch.cat([gut_features, metadata_features], dim=1)
        else:
            # 仅使用肠道菌群特征
            combined_features = gut_features
        
        # 分类
        logits = self.classifier(combined_features)
        
        # Sigmoid激活（二分类）
        probabilities = torch.sigmoid(logits)
        
        outputs = {"probabilities": probabilities, "logits": logits}
        
        # 如果需要，计算并返回注意力权重
        if return_attention and self.attention is not None:
            attention_weights = self.attention(gut_features)
            outputs["attention_weights"] = attention_weights
        
        return outputs
    
    def get_feature_importance(self, x_gut):
        """
        获取菌种重要性评分（用于可解释性）
        """
        if self.attention is None:
            return None
        
        with torch.no_grad():
            gut_features = self.gut_encoder(x_gut)
            importance = self.attention(gut_features)
        
        return importance

class MultiTaskASCClassifier(nn.Module):
    """
    多任务分类器（可选）：
    1. 主要任务：ASC分类（二分类）
    2. 辅助任务：重建关键菌种（帮助学习更好的表示）
    """
    def __init__(
        self,
        gut_dim: int,
        metadata_dim: int,
        gut_latent_dim: int = 128,
        metadata_latent_dim: int = 64,
        classifier_hidden_dims: List[int] = [256, 128, 64],
        reconstruct_hidden_dims: List[int] = [128, 256],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_metadata: bool = True,  # 新增：是否使用元数据
    ):
        super().__init__()
        
        self.use_metadata = use_metadata
        
        # 共享的肠道菌群编码器
        self.gut_encoder = GutEncoder(
            input_dim=gut_dim,
            latent_dim=gut_latent_dim,
            hidden_dims=[gut_dim * 2, gut_dim],
            dropout=dropout * 0.8
        )
        
        # 元数据编码器（可选）
        if use_metadata and metadata_dim > 0:
            self.metadata_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                latent_dim=metadata_latent_dim,
                dropout=dropout
            )
            self.total_latent_dim = gut_latent_dim + metadata_latent_dim
        else:
            self.metadata_encoder = None
            self.total_latent_dim = gut_latent_dim
        
        # 分类头
        self.classifier = mlp(
            in_dim=self.total_latent_dim,
            hidden_dims=classifier_hidden_dims,
            out_dim=1,
            dropout=dropout,
            bn=use_batch_norm,
            last_activation=False
        )
        
        # 重建头（辅助任务）
        self.reconstructor = mlp(
            in_dim=gut_latent_dim,  # 只从肠道特征重建
            hidden_dims=reconstruct_hidden_dims,
            out_dim=gut_dim,
            dropout=dropout * 0.5,
            bn=use_batch_norm,
            last_activation=False
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x_gut, x_metadata):
        # 编码肠道菌群特征
        gut_features = self.gut_encoder(x_gut)
        
        # 编码元数据特征（如果可用）
        if self.use_metadata and self.metadata_encoder is not None:
            metadata_features = self.metadata_encoder(x_metadata)
            # 融合特征
            combined_features = torch.cat([gut_features, metadata_features], dim=1)
        else:
            # 仅使用肠道菌群特征
            combined_features = gut_features
        
        # 分类
        logits = self.classifier(combined_features)
        probabilities = torch.sigmoid(logits)
        
        # 重建（辅助任务）
        reconstructed = self.reconstructor(gut_features)
        
        return {
            "probabilities": probabilities,
            "logits": logits,
            "reconstructed": reconstructed,
            "gut_features": gut_features
        }

# 损失函数定义
class ASCClassificationLoss(nn.Module):
    """ASC分类的损失函数"""
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, outputs, targets):
        logits = outputs["logits"].squeeze()
        return self.bce_loss(logits, targets)

class MultiTaskLoss(nn.Module):
    """多任务损失（分类 + 重建）"""
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, 
                 reconstruction_weight: float = 0.1):
        super().__init__()
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.reconstruction_loss = nn.MSELoss()
        self.reconstruction_weight = reconstruction_weight
    
    def forward(self, outputs, targets, reconstruction_targets):
        # 分类损失
        logits = outputs["logits"].squeeze()
        cls_loss = self.classification_loss(logits, targets)
        
        # 重建损失
        recon_loss = self.reconstruction_loss(
            outputs["reconstructed"], 
            reconstruction_targets
        )
        
        # 总损失
        total_loss = cls_loss + self.reconstruction_weight * recon_loss
        
        return {
            "total_loss": total_loss,
            "classification_loss": cls_loss,
            "reconstruction_loss": recon_loss
        }