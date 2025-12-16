# Encoder Architecture Guide

The agent now supports two encoder architectures: standard CNN (fast) and EfficientNet (higher quality).

## Quick Start

### Option 1: Standard CNN (Default - Fast)
```bash
python ppo_atari_lstm.py --encoder-type cnn
```
- **Speed**: ~5000+ SPS
- **Embedding dim**: 512
- **Best for**: Fast iteration, standard RL training

### Option 2: EfficientNet-B0 (Balanced)
```bash
python ppo_atari_lstm.py \
    --encoder-type efficientnet \
    --efficientnet-variant b0 \
    --efficientnet-pretrained
```
- **Speed**: ~500-1000 SPS (5-10× slower)
- **Embedding dim**: 1280
- **Best for**: Better representations, worth the slowdown

### Option 3: EfficientNet-B1 (SCIL Validated)
```bash
python ppo_atari_lstm.py \
    --encoder-type efficientnet \
    --efficientnet-variant b1 \
    --efficientnet-pretrained
```
- **Speed**: ~200-500 SPS (20-30× slower)
- **Embedding dim**: 1280
- **Best for**: Maximum alignment quality (proven in your SCIL experiments)

### Option 4: Frozen EfficientNet (Faster)
```bash
python ppo_atari_lstm.py \
    --encoder-type efficientnet \
    --efficientnet-variant b0 \
    --efficientnet-pretrained \
    --efficientnet-freeze
```
- **Speed**: ~1000-2000 SPS (faster than unfrozen)
- Only trains LSTM + heads, uses pretrained ImageNet features
- Good for transfer learning scenarios

## Speed vs Quality Tradeoff

| Encoder | SPS | Params | Embedding Dim | When to Use |
|---------|-----|--------|---------------|-------------|
| CNN | 5000+ | 3.1M | 512 | Fast prototyping, standard RL |
| EfficientNet-B0 | 500-1000 | 5.3M | 1280 | Balanced speed/quality |
| EfficientNet-B1 | 200-500 | 7.8M | 1280 | Best alignment (SCIL proven) |
| EfficientNet-B2 | 100-300 | 9.2M | 1408 | Maximum capacity |

## Implementation Details

### Architecture Modularity
```python
# Both architectures follow the same pattern:
Input (84×84 grayscale → converted to RGB internally)
    ↓
encoder → embeddings (512 or 1280-dim)  ← SupCon loss applied here
    ↓
LSTM → latent state (128-dim)           ← R3L anchors/SAPS alignment here
    ↓
actor/critic heads → outputs
```

### Key Methods

**`get_embeddings(x)`**:
- Returns encoder output before LSTM
- Use for SupCon loss and SAPS
- CNN: 512-dim, EfficientNet: 1280/1408-dim

**`get_states(x, lstm_state, done)`**:
- Returns LSTM hidden state
- Use for RL value/policy computation
- Always 128-dim regardless of encoder

### Preprocessing (Unified)

**Both CNN and EfficientNet**:
- Input: 84×84 grayscale (1 channel) → automatically converted to RGB (3 channels)
- Normalized to [0, 1]
- Uses RGB for consistency and better feature learning
- Seamless encoder switching without environment changes

## Recommendations

### For Initial Experiments
Start with **CNN** to iterate quickly. Add SupCon loss and validate it works.

### For Alignment Quality
Once you're ready for serious training, use **EfficientNet-B1**:
- Proven in your SCIL experiments (0.53 silhouette score)
- Better action-based clustering
- Worth the 20× slowdown for better transfer

### For Limited Compute
Use **frozen EfficientNet-B0**:
- Leverages ImageNet features (good general representations)
- 2-3× faster than unfrozen
- Still better than CNN for SupCon clustering

## Example Training Commands

### Quick validation (CNN + SupCon)
```bash
python ppo_atari_lstm.py \
    --encoder-type cnn \
    --total-timesteps 1000000 \
    --track  # Add when you integrate SupCon
```

### Full training (EfficientNet-B1 like SCIL)
```bash
python ppo_atari_lstm.py \
    --encoder-type efficientnet \
    --efficientnet-variant b1 \
    --efficientnet-pretrained \
    --total-timesteps 10000000 \
    --track
```

### Fast transfer learning (frozen backbone)
```bash
python ppo_atari_lstm.py \
    --encoder-type efficientnet \
    --efficientnet-variant b0 \
    --efficientnet-pretrained \
    --efficientnet-freeze \
    --total-timesteps 5000000
```

## Notes

- **ImageNet pretraining helps**: Always use `--efficientnet-pretrained` unless you have a specific reason not to
- **Memory usage**: EfficientNet requires ~2-3× more GPU memory due to RGB conversion and larger architecture
- **Batch size**: You might need to reduce `--num-envs` or `--num-steps` if you run out of memory with EfficientNet
- **Compatibility**: Both encoders work identically with SupCon loss and SAPS - the interface is the same

## Next Steps

1. Test basic PPO training with CNN encoder (baseline)
2. Add SupCon loss (works with both encoders)
3. Compare action clustering quality (CNN vs EfficientNet)
4. If EfficientNet shows better clustering, use it for final training runs
5. Use SAPS for post-training alignment or zero-shot stitching
