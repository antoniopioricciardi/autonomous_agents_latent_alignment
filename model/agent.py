import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_cnn_encoder(input_channels=3):
    """
    Standard CNN encoder for Atari (fast, optimized for 84x84 RGB)
    Output: 512-dim embeddings
    Speed: ~5000+ SPS on typical hardware

    Args:
        input_channels: Number of input channels
            - 3 for RGB (no frame stacking)
            - 12 for RGB with 4-frame stacking
            - 4 for grayscale with 4-frame stacking (standard Atari)
    """
    return nn.Sequential(
        layer_init(nn.Conv2d(input_channels, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )


def build_efficientnet_encoder(variant='b0', pretrained=True, freeze_backbone=False):
    """
    EfficientNet encoder (slower but higher capacity, for 84x84 RGB)

    Args:
        variant: 'b0', 'b1', 'b2', etc. (b0 is fastest, b1 used in SCIL)
        pretrained: Use ImageNet weights (recommended)
        freeze_backbone: Freeze backbone weights (faster training, less flexible)

    Output dimensions:
        - b0: 1280-dim
        - b1: 1280-dim
        - b2: 1408-dim

    Speed: ~200-500 SPS (20-30× slower than CNN, but better representation quality)

    Note: Expects RGB input (3 channels). Grayscale conversion handled in _preprocess_input().
    """
    from torchvision import models

    # Map variant to torchvision model
    efficientnet_models = {
        'b0': models.efficientnet_b0,
        'b1': models.efficientnet_b1,
        'b2': models.efficientnet_b2,
    }

    if variant not in efficientnet_models:
        raise ValueError(f"Unsupported variant: {variant}. Choose from {list(efficientnet_models.keys())}")

    # Load pretrained model
    efficientnet = efficientnet_models[variant](pretrained=pretrained)

    # Remove classifier head (keep feature extractor)
    backbone = nn.Sequential(*list(efficientnet.children())[:-1])

    # Optionally freeze backbone
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print(f"EfficientNet-{variant.upper()} backbone frozen - only training LSTM and heads")

    # Get output dimension
    embedding_dim = {
        'b0': 1280,
        'b1': 1280,
        'b2': 1408,
    }[variant]

    return backbone, embedding_dim


class Agent(nn.Module):
    def __init__(self, envs, encoder_type='cnn', efficientnet_variant='b0',
                 efficientnet_pretrained=True, efficientnet_freeze=False):
        """
        PPO Agent with configurable encoder architecture.

        Args:
            envs: Vectorized gym environments
            encoder_type: 'cnn' (fast, default) or 'efficientnet' (slower, higher quality)
            efficientnet_variant: 'b0' (fastest), 'b1' (SCIL default), 'b2' (largest)
            efficientnet_pretrained: Use ImageNet pretrained weights (recommended)
            efficientnet_freeze: Freeze EfficientNet backbone (faster training)
        """
        super().__init__()

        self.encoder_type = encoder_type

        # Determine input channels from observation space
        # Handle frame stacking: observation space might be [F, C, H, W] or [C, H, W]
        obs_shape = envs.single_observation_space.shape
        if len(obs_shape) == 4:
            # Frame stacking: [F, C, H, W]
            num_frames, channels_per_frame, _, _ = obs_shape
            input_channels = num_frames * channels_per_frame
        elif len(obs_shape) == 3:
            # No frame stacking: [C, H, W]
            input_channels, _, _ = obs_shape
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")

        print(f"Observation space shape: {obs_shape} → {input_channels} input channels")

        # Build encoder based on type
        if encoder_type == 'cnn':
            self.encoder = build_cnn_encoder(input_channels=input_channels)
            embedding_dim = 512
            print(f"Using standard CNN encoder (512-dim, {input_channels} channels, optimized for speed)")

        elif encoder_type == 'efficientnet':
            # EfficientNet expects 3 channels (RGB), so we need an adapter layer
            # if input has different number of channels
            if input_channels != 3:
                print(f"⚠️  EfficientNet expects 3 channels, but input has {input_channels}")
                print(f"    Adding 1x1 conv adapter: {input_channels} → 3 channels")
                adapter = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1)
                efficientnet_backbone, embedding_dim = build_efficientnet_encoder(
                    variant=efficientnet_variant,
                    pretrained=efficientnet_pretrained,
                    freeze_backbone=efficientnet_freeze
                )
                self.encoder = nn.Sequential(adapter, efficientnet_backbone)
            else:
                self.encoder, embedding_dim = build_efficientnet_encoder(
                    variant=efficientnet_variant,
                    pretrained=efficientnet_pretrained,
                    freeze_backbone=efficientnet_freeze
                )
            print(f"Using EfficientNet-{efficientnet_variant.upper()} encoder ({embedding_dim}-dim)")
            print(f"⚠️  Expected speed: ~20-30× slower than CNN, but better representations")

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.input_channels = input_channels

        # Recurrent layer: embeddings → latent state (128-dim)
        self.lstm = nn.LSTM(embedding_dim, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Policy and value heads
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def _preprocess_input(self, x):
        """
        Preprocess input for both encoders (both use RGB).
        Handles frame stacking and grayscale-to-RGB conversion.
        """
        x = x / 255.0  # Normalize to [0, 1]

        # Handle frame stacking: [B, F, C, H, W] → [B, F*C, H, W]
        # where F=num_frames, C=channels (1 for grayscale, 3 for RGB)
        if x.dim() == 5:
            batch_size, num_frames, channels, height, width = x.shape
            x = x.reshape(batch_size, num_frames * channels, height, width)
            # Note: For stack=4, RGB input becomes 12 channels (4 frames × 3 RGB)

        # Convert grayscale to RGB if input is single channel (no frame stacking)
        elif x.dim() == 4 and x.shape[1] == 1:
            print("\n\n!! WARNING, INPUT IS GRAYSCALE - CONVERTING TO RGB !!\n\n")
            x = x.repeat(1, 3, 1, 1)  # [B, 1, H, W] → [B, 3, H, W]

        return x

    def get_embeddings(self, x):
        """
        Extract encoder embeddings before LSTM.

        Output dimensions:
        - CNN: 512-dim
        - EfficientNet-B0/B1: 1280-dim
        - EfficientNet-B2: 1408-dim

        Use for:
        - Supervised contrastive loss (SupCon)
        - SAPS transformation estimation
        - Representation analysis
        """
        x = self._preprocess_input(x)
        h = self.encoder(x)

        # EfficientNet outputs [B, C, 1, 1], need to flatten
        if self.encoder_type == 'efficientnet':
            h = h.view(h.size(0), -1)

        return h

    def get_states(self, x, lstm_state, done):
        """Get LSTM latent state (128-dim) - the main representation for RL"""
        hidden = self.get_embeddings(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state