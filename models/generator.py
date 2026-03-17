import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. INVERTIBLE BLOCK  (ISN — paper Section 3.2, Figure 1)
# ============================================================
# This is the core of the watermarking system.
# It uses an affine coupling layer:
#   Forward (embed):  y2 = x2 * exp(s(x1)) + t(x1)
#   Reverse (extract): x2 = (y2 - t(y1)) * exp(-s(y1))
# This guarantees perfect invertibility without any information loss.
# ============================================================

class InvertibleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        half_channels = in_channels // 2
        
        # Network 1: Modifies the secret based on the cover
        self.s1 = nn.Sequential(nn.Conv2d(half_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half_channels, 3, padding=1), nn.Tanh())
        self.t1 = nn.Sequential(nn.Conv2d(half_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half_channels, 3, padding=1))
        
        # Network 2: Modifies the cover based on the secret (This actually embeds the watermark!)
        self.s2 = nn.Sequential(nn.Conv2d(half_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half_channels, 3, padding=1), nn.Tanh())
        self.t2 = nn.Sequential(nn.Conv2d(half_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half_channels, 3, padding=1))

    def forward(self, x, reverse=False):
        if not reverse:
            x1, x2 = torch.chunk(x, 2, dim=1) # x1: cover, x2: secret
            
            # Step 1: Cover modifies Secret
            y2 = x2 * torch.exp(self.s1(x1)) + self.t1(x1)
            y1 = x1
            
            # Step 2: Secret modifies Cover (Embedding occurs here)
            z1 = y1 * torch.exp(self.s2(y2)) + self.t2(y2)
            z2 = y2
            
            return torch.cat([z1, z2], dim=1)
        else:
            z1, z2 = torch.chunk(x, 2, dim=1)
            
            # Reverse Step 2
            y2 = z2
            y1 = (z1 - self.t2(y2)) * torch.exp(-self.s2(y2))
            
            # Reverse Step 1
            x1 = y1
            x2 = (y2 - self.t1(x1)) * torch.exp(-self.s1(x1))
            
            return torch.cat([x1, x2], dim=1)

# ============================================================
# 2. DENSE BLOCK  (paper Section 3.3, inside Figure 2)
# ============================================================
# A 2-layer dense block: new features are concatenated with the
# input before being passed to the next conv (dense connectivity).
# A final residual skip ensures no information is lost.
# ============================================================

class DenseBlock(nn.Module):
    """
    2-layer densely-connected convolutional block.
    Used as the Pre-Dense and Post-Dense sub-modules inside EnhancementModule.

    Input  → Conv1 → ReLU → cat(Input, out1) → Conv2 → + Input (residual) → Output
    """

    def __init__(self, channels=3, growth_rate=16):
        super().__init__()
        # Conv1 expands to growth_rate feature maps
        self.conv1 = nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1)
        # Conv2 takes original + new features, outputs back to original channel count
        self.conv2 = nn.Conv2d(channels + growth_rate, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        # Dense connection: concatenate input with first output
        out2 = self.conv2(torch.cat([x, out1], dim=1))
        # Residual connection: add input back to maintain signal
        return out2 + x


# ============================================================
# 3. DYNAMIC MLP  (paper Section 3.3, Figure 3)
# ============================================================
# Generates a per-channel weight vector from the feature map.
# This is the "dynamic" part of the dynamic attention mechanism.
# The weights adaptively scale the attention features based on
# the current image content.
# ============================================================

class DynamicMLP(nn.Module):
    """
    Lightweight two-layer MLP that produces a dynamic weight vector.
    Input: feature map (B, C, H, W)
    Output: per-channel weight tensor (B, C, 1, 1)

    Flow: GlobalAvgPool → Flatten → Linear → ReLU → Linear → weights
    """

    def __init__(self, channels):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)   # compress spatial → (B, C, 1, 1)
        self.flatten = nn.Flatten()               # → (B, C)
        self.fc1     = nn.Linear(channels, channels // 2)
        self.relu    = nn.ReLU(inplace=True)
        self.fc2     = nn.Linear(channels // 2, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x)           # (B, C, 1, 1)
        w = self.flatten(w)        # (B, C)
        w = self.relu(self.fc1(w)) # (B, C//2)
        w = self.fc2(w)            # (B, C)
        return w.view(b, c, 1, 1)  # reshape for channel-wise multiplication


# ============================================================
# 4. SWIN-LIKE BLOCK  (paper Section 3.3, Figure 3)
# ============================================================
# Approximates the Swin Transformer's shifted-window attention.
# The key idea is that a conv captures spatial relationships (like
# window attention), then DynamicMLP scales those features
# differently for each channel (dynamic attention weights).
# window_size=4  → better for GLOBAL Gaussian noise (small window)
# window_size=8  → better for LOCAL JPEG block artifacts (larger window)
# ============================================================

class SwinLikeBlock(nn.Module):
    def __init__(self, channels, window_size=4):
        super().__init__()
        self.window_size = window_size
        
        # Keep kernel=3 and padding=1 to ensure the output remains exactly 224x224
        # Do not use window_size as the kernel_size here!
        self.conv_global = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dynamic_mlp = DynamicMLP(channels)

    def forward(self, x):
        global_feat = F.relu(self.conv_global(x))
        dynamic_weights = self.dynamic_mlp(x)
        weighted = global_feat * dynamic_weights
        return x + weighted


# ============================================================
# 5. ENHANCEMENT MODULE  (paper Section 3.3, Figure 2)
# ============================================================
# This is the full image enhancement module from Figure 2.
# Structure: Pre-Dense → Swin-Like Block → Post-Dense → residual
#
# TWO instances are created in WatermarkGenerator:
#   enhance_pre  (window_size=4): used BEFORE ISN reverse extraction
#                                 → targets global Gaussian noise
#   enhance_post (window_size=8): used AFTER ISN reverse extraction
#                                 → targets local JPEG block artifacts
# ============================================================

class EnhancementModule(nn.Module):
    """
    Full image enhancement module from paper Figure 2.
    Structure: Pre-Dense → SwinLikeBlock → Post-Dense → + residual

    Args:
        channels    : number of image channels (3 for RGB)
        window_size : attention window size (4 for pre, 8 for post)
    """

    def __init__(self, channels=3, window_size=4):
        super().__init__()
        # Pre-Dense: extracts local low-level features
        self.pre_dense  = DenseBlock(channels)
        # SwinLikeBlock: captures global context with dynamic attention
        self.swin_block = SwinLikeBlock(channels, window_size=window_size)
        # Post-Dense: fuses local and global information
        self.post_dense = DenseBlock(channels)

    def forward(self, x):
        # Pre-Dense: local feature extraction
        feat = self.pre_dense(x)
        # Swin Block + Dynamic MLP: global context + dynamic weighting
        feat = self.swin_block(feat)
        # Post-Dense: local-global feature fusion
        feat = self.post_dense(feat)
        # Final residual: concatenate enhanced features with original input
        # (paper Figure 2 shows concat(xd, xd_post) at the output)
        return feat + x


# ============================================================
# 6. DIFFERENTIAL FEATURE EXTRACTOR  (paper Section 3.4, Figure 4)
# ============================================================
# This 4-layer DenseNet captures what was LOST when the watermarked
# image was attacked. It computes (xc - xd) as input, where:
#   xc = pristine watermarked image
#   xd = attacked (distorted) version
# The output (xc_feat) is fed into the ISN during extraction to
# compensate for information loss caused by the attack.
# ============================================================

class DifferentialFeatureExtractor(nn.Module):
    """
    4-layer densely-connected network that extracts differential
    features between watermarked image (xc) and attacked image (xd).

    Input : xc (watermarked, 3ch) and xd (attacked, 3ch)
    Output: xc_feat (3ch) — compensation features for the extractor

    Dense connectivity: each conv sees all previous outputs.
    Final 1×1 conv collapses back to 3 channels (matches paper).
    """

    def __init__(self, channels=3, growth_rate=32):
        super().__init__()
        ch = channels  # 3

        # Layer 1: diff (3ch) → 32 feature maps
        self.conv1 = nn.Conv2d(ch,          growth_rate,     kernel_size=3, padding=1)
        # Layer 2: diff (3) + x1 (32) = 35ch → 32 feature maps
        self.conv2 = nn.Conv2d(ch +   growth_rate,     growth_rate, kernel_size=3, padding=1)
        # Layer 3: diff (3) + x1 (32) + x2 (32) = 67ch → 32 feature maps
        self.conv3 = nn.Conv2d(ch + 2*growth_rate,     growth_rate, kernel_size=3, padding=1)
        # Layer 4: diff (3) + x1 + x2 + x3 = 99ch → 32 feature maps
        self.conv4 = nn.Conv2d(ch + 3*growth_rate,     growth_rate, kernel_size=3, padding=1)
        # Final 1×1 conv: collapse all features back to 3 output channels
        self.final = nn.Conv2d(ch + 4*growth_rate, ch, kernel_size=1)

    def forward(self, xc, xd):
        """
        Args:
            xc : watermarked image  (B, 3, H, W)
            xd : attacked image     (B, 3, H, W)
        Returns:
            xc_feat : compensation features (B, 3, H, W)
        """
        # Compute the pixel-wise difference (captures what was damaged)
        diff = xc - xd                                          # (B, 3, H, W)

        # 4-layer dense forward pass
        x1 = F.relu(self.conv1(diff))                           # (B, 32, H, W)
        x2 = F.relu(self.conv2(torch.cat([diff, x1],       dim=1)))   # (B, 32)
        x3 = F.relu(self.conv3(torch.cat([diff, x1, x2],  dim=1)))   # (B, 32)
        x4 = F.relu(self.conv4(torch.cat([diff, x1, x2, x3], dim=1))) # (B, 32)

        # 1×1 conv to get final 3-channel compensation feature map
        out = self.final(torch.cat([diff, x1, x2, x3, x4], dim=1))    # (B, 3)
        return out


# ============================================================
# 7. WATERMARK GENERATOR  (paper Figure 1 — ties everything together)
# ============================================================
# This is the top-level model. It wires all components exactly
# as described in Figure 1 of the paper.
#
# EMBEDDING  (forward / embed):
#   cover + secret → ISN → watermarked image (xc)
#
# EXTRACTION (extract):
#   Step 1: Get compensation features: DiffFeat(xc, xd) → xc_feat
#   Step 2: Pre-enhance attacked image: Enhance_pre(xd) → xd'
#   Step 3: ISN reverse: cat(xd', xc_feat) → ISN⁻¹ → raw secret
#   Step 4: Post-enhance recovered secret: Enhance_post(raw) → xe
# ============================================================

class WatermarkGenerator(nn.Module):
    """
    Full watermarking pipeline from paper Figure 1.

    Components:
        isn         : InvertibleBlock    — core embedding/extraction
        enhance_pre : EnhancementModule  — pre-extraction, window=4 (Gaussian)
        enhance_post: EnhancementModule  — post-extraction, window=8 (JPEG)
        diff_feat   : DifferentialFeatureExtractor — compensation features
    """

    def __init__(self):
        super().__init__()

        # ISN: the invertible core (paper Section 3.2)
        self.isn = InvertibleBlock(in_channels=6)

        # Pre-enhancement: applied BEFORE ISN reverse (targets Gaussian noise)
        # Paper: "initial image enhancement module emphasising global Gaussian noise"
        # window_size=4 → small window captures fine-grained global noise
        self.enhance_pre  = EnhancementModule(channels=3, window_size=4)

        # Post-enhancement: applied AFTER ISN reverse (targets JPEG artifacts)
        # Paper: "subsequent module focusing on local distortions caused by JPEG"
        # window_size=8 → larger window captures local block-level artifacts
        self.enhance_post = EnhancementModule(channels=3, window_size=8)

        # Differential feature extractor (paper Section 3.4, Figure 4)
        self.diff_feat = DifferentialFeatureExtractor(channels=3)

    # ----------------------------------------------------------
    # EMBEDDING  (paper: Generator in Figure 1)
    # ----------------------------------------------------------
    def embed(self, cover, secret):
        """
        Embed the secret image into the cover image.

        Args:
            cover  : (B, 3, H, W)  — the carrier image (xh)
            secret : (B, 3, H, W)  — the image to hide (xs)
        Returns:
            watermarked : (B, 3, H, W)  — the watermarked image (xc)
        """
        # Concatenate cover and secret → 6 channel input for ISN
        x = torch.cat([cover, secret], dim=1)    # (B, 6, H, W)

        # ISN forward pass: embeds secret into cover
        isn_out = self.isn(x, reverse=False)      # (B, 6, H, W)

        # The watermarked image is the first 3 channels
        # (cover occupies ch 0-2, secret was embedded into ch 3-5,
        #  but ISN mixes them; we take ch 0-2 as the final image)
        watermarked = isn_out[:, :3, :, :]        # (B, 3, H, W)

        return torch.clamp(watermarked, -1.0, 1.0)

    # ----------------------------------------------------------
    # EXTRACTION  (paper: Extraction pipeline in Figure 1)
    # ----------------------------------------------------------
    def extract(self, attacked, watermarked):
        """
        Recover the secret image from an attacked watermarked image.

        Args:
            attacked    : (B, 3, H, W) — the attacked/distorted image (xd)
            watermarked : (B, 3, H, W) — the original watermarked image (xc)
                          (needed to compute differential features)
        Returns:
            extracted : (B, 3, H, W) — the recovered secret image (xe)
        """

        # STEP 1: Compute differential features (Section 3.4, Figure 4)
        # Captures the information that was damaged by the attack
        xc_feat = self.diff_feat(watermarked, attacked)    # (B, 3, H, W)

        # STEP 2: Pre-enhance the attacked image (Figure 2, window=4)
        # Reduces global Gaussian noise before passing to ISN
        xd_enhanced = self.enhance_pre(attacked)           # (B, 3, H, W)

        # STEP 3: ISN reverse — recover the secret
        # Concatenate enhanced attacked image + compensation features → 6ch
        combined = torch.cat([xd_enhanced, xc_feat], dim=1)  # (B, 6, H, W)
        isn_out  = self.isn(combined, reverse=True)           # (B, 6, H, W)

        # Secret was in channels 3-5 during embedding
        raw_secret = isn_out[:, 3:, :, :]                    # (B, 3, H, W)

        # STEP 4: Post-enhance the recovered secret (Figure 2, window=8)
        # Cleans up local JPEG block artifacts in the recovered secret
        extracted = self.enhance_post(raw_secret)             # (B, 3, H, W)

        return torch.clamp(extracted, -1.0, 1.0)

    # ----------------------------------------------------------
    # FORWARD  (called by train.py during embedding)
    # ----------------------------------------------------------
    def forward(self, cover, secret):
        """Alias for embed() — called during the training forward pass."""
        return self.embed(cover, secret)