import torch
import torch.nn as nn
from typing import List, Optional


def mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    dropout: float = 0.0,
    act=nn.LeakyReLU,
    bn: bool = True,
    last_activation: bool = False,
) -> nn.Sequential:
    """Build a standard multilayer perceptron."""
    layers = []
    dim = in_dim

    for h in hidden_dims:
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
    """Encoder for selected gut microbiome species features."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = mlp(
            input_dim,
            hidden_dims,
            latent_dim,
            dropout=dropout,
            last_activation=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MetadataEncoder(nn.Module):
    """Encoder for non-disease metadata covariates."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.4,
    ):
        super().__init__()
        if input_dim > 0:
            self.encoder = mlp(
                input_dim,
                hidden_dims,
                latent_dim,
                dropout=dropout,
                last_activation=True,
            )
            self.latent_dim = latent_dim
        else:
            self.encoder = None
            self.latent_dim = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is None or x.shape[1] == 0:
            return torch.zeros(x.shape[0], 0, device=x.device)
        return self.encoder(x)


class ASCClassifier(nn.Module):
    """Binary ASC classifier using gut microbiome features and optional metadata."""

    def __init__(
        self,
        gut_dim: int,
        metadata_dim: int,
        gut_latent_dim: int = 128,
        metadata_latent_dim: int = 64,
        classifier_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_metadata: bool = True,
    ):
        super().__init__()

        self.use_metadata = use_metadata

        self.gut_encoder = GutEncoder(
            input_dim=gut_dim,
            latent_dim=gut_latent_dim,
            hidden_dims=[gut_dim * 2, gut_dim],
            dropout=dropout * 0.8,
        )

        if use_metadata and metadata_dim > 0:
            self.metadata_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                latent_dim=metadata_latent_dim,
                hidden_dims=[max(64, metadata_dim // 2), 64],
                dropout=dropout,
            )
            self.total_latent_dim = gut_latent_dim + metadata_latent_dim
        else:
            self.metadata_encoder = None
            self.total_latent_dim = gut_latent_dim
            print("Model configuration: metadata is disabled; only gut microbiome features are used.")

        self.classifier = mlp(
            in_dim=self.total_latent_dim,
            hidden_dims=classifier_hidden_dims,
            out_dim=1,
            dropout=dropout,
            bn=use_batch_norm,
            last_activation=False,
        )

        if gut_dim > 0:
            self.attention = nn.Sequential(
                nn.Linear(gut_latent_dim, gut_latent_dim // 2),
                nn.Tanh(),
                nn.Linear(gut_latent_dim // 2, gut_dim),
                nn.Softmax(dim=1),
            )
        else:
            self.attention = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x_gut: torch.Tensor,
        x_metadata: torch.Tensor,
        return_attention: bool = False,
    ) -> dict:
        gut_features = self.gut_encoder(x_gut)

        if self.use_metadata and self.metadata_encoder is not None:
            metadata_features = self.metadata_encoder(x_metadata)
            combined_features = torch.cat([gut_features, metadata_features], dim=1)
        else:
            combined_features = gut_features

        logits = self.classifier(combined_features)
        probabilities = torch.sigmoid(logits)

        outputs = {
            "probabilities": probabilities,
            "logits": logits,
        }

        if return_attention and self.attention is not None:
            outputs["attention_weights"] = self.attention(gut_features)

        return outputs

    def get_feature_importance(self, x_gut: torch.Tensor) -> Optional[torch.Tensor]:
        if self.attention is None:
            return None

        with torch.no_grad():
            gut_features = self.gut_encoder(x_gut)
            importance = self.attention(gut_features)

        return importance


class MultiTaskASCClassifier(nn.Module):
    """Optional multitask classifier with an auxiliary gut-feature reconstruction head."""

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
        use_metadata: bool = True,
    ):
        super().__init__()

        self.use_metadata = use_metadata

        self.gut_encoder = GutEncoder(
            input_dim=gut_dim,
            latent_dim=gut_latent_dim,
            hidden_dims=[gut_dim * 2, gut_dim],
            dropout=dropout * 0.8,
        )

        if use_metadata and metadata_dim > 0:
            self.metadata_encoder = MetadataEncoder(
                input_dim=metadata_dim,
                latent_dim=metadata_latent_dim,
                dropout=dropout,
            )
            self.total_latent_dim = gut_latent_dim + metadata_latent_dim
        else:
            self.metadata_encoder = None
            self.total_latent_dim = gut_latent_dim

        self.classifier = mlp(
            in_dim=self.total_latent_dim,
            hidden_dims=classifier_hidden_dims,
            out_dim=1,
            dropout=dropout,
            bn=use_batch_norm,
            last_activation=False,
        )

        self.reconstructor = mlp(
            in_dim=gut_latent_dim,
            hidden_dims=reconstruct_hidden_dims,
            out_dim=gut_dim,
            dropout=dropout * 0.5,
            bn=use_batch_norm,
            last_activation=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x_gut: torch.Tensor, x_metadata: torch.Tensor) -> dict:
        gut_features = self.gut_encoder(x_gut)

        if self.use_metadata and self.metadata_encoder is not None:
            metadata_features = self.metadata_encoder(x_metadata)
            combined_features = torch.cat([gut_features, metadata_features], dim=1)
        else:
            combined_features = gut_features

        logits = self.classifier(combined_features)
        probabilities = torch.sigmoid(logits)
        reconstructed = self.reconstructor(gut_features)

        return {
            "probabilities": probabilities,
            "logits": logits,
            "reconstructed": reconstructed,
            "gut_features": gut_features,
        }


class ASCClassificationLoss(nn.Module):
    """Binary classification loss for ASC prediction."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, outputs: dict, targets: torch.Tensor) -> torch.Tensor:
        logits = outputs["logits"].squeeze()
        return self.bce_loss(logits, targets)
