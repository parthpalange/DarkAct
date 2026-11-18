# DarkAct-Net: Paper vs. Code Differences

This document compares the paper ([papers/DarkAct.pdf](papers/DarkAct.pdf), CVPR 2026, Section 4 "DarkAct-Net", Section 5.1 "Implementation Details", Figure 3) with the code in this repository.

For every difference it states what the paper says, what the code does, why the code may have been written that way, and what effect the difference has.

The paper refers to an **Appendix** for several details (the mapping network ϕ, the class list and extra training details). The appendix is **not in the PDF** in this repo. Where the paper is silent, the code's choice is marked as *"paper unspecified"* and is an interpretation, not necessarily a deviation.

---

## Summary

| # | Area | Paper | Code | Severity |
|---|------|-------|------|----------|
| 1 | Training loss | Standard cross-entropy only | CE + Sobel + MSP + Contrastive + SSIM | **Major** |
| 2 | Patch Embedding | Separate block between MAA and Stage 1 | Merged into `ModalEmbedding` (before MAA) | **Major** |
| 3 | Backbone attention | MViTv2 (pooling attention, relative position) | Divided space-time attention, no positional info | **Major** |
| 4 | Backbone depth | MViTv2-B: depths [2, 3, 16, 3] | [2, 3, 6, 2] | **Major** |
| 5 | Stage downsampling | MViTv2 Q-pooling inside attention | Strided Conv3d before each stage | Medium |
| 6 | MAA output (Eq. 3) | `Y = Softmax(EQ·ESᵀ/√C)·ES`, plus G (Fig. 3) | Sigmoid gate: `Y = G·σ(conv(attn)) + G` | Medium |
| 7 | MAA query mapping (Eq. 2) | `EQ = Ē·W` (right-multiply) | `EQ = W·Ē` (left-multiply, C×C) | Medium |
| 8 | MAA temporal pooling | Paper unspecified | E^S and E^Q are averaged/pooled over T | Minor |
| 9 | LAF temporal handling | Paper unspecified (N = tokens) | LAF works on time-averaged features; F_LAF is broadcast over T | **Major** |
| 10 | LAF key block (Eq. 5) | 3×3 dilated conv + Sigmoid | Dilated conv + BN + PReLU + 1×1 conv + Sigmoid | Minor |
| 11 | LAF cross-attention | `Att(F^r, F^r, F^t)` | Multi-head attention with learned Q/K/V projections | Minor |
| 12 | LAF value pooling (Eq. 6) | "Hierarchical" Max/Avg pooling | Local 3×3 Max/Avg pooling, stride 1 | Minor |
| 13 | LAF output | `F_LAF = MLP(F_d^r) + MLP(F_d^t)` | Extra LayerNorm on F_LAF | Minor |
| 14 | Y resizing in LAF | Paper unspecified | Trilinear interpolation + one shared 1×1 conv | Minor |
| 15 | Weight sharing | Fig. 3 draws Patch Embed / Stages as single boxes spanning both streams | Fully separate RGB and thermal weights | Unclear |
| 16 | Classifier | Final features "flattened and concatenated" | Global average pooling, then concat + Dropout 0.5 | Medium |
| 17 | Optimizer / schedule | AdamW, wd 5e-3, 20 warm-up + 100 cosine epochs, 1e-3 → 1e-4, bs 8 | Same defaults | ✅ Matches |
| 18 | Input sampling / augmentation | Paper unspecified | 16 frames, resize to 224, horizontal flip only | Minor |
| 19 | Class names | README/paper list includes "clapping" | Code list has `"slap"` | Check |
| 20 | Benchmark baselines | Real models (MRFS, CMX, Conv2Former, …) | Generic placeholders built from this backbone | **Major** |

---

## 1. Training loss: the biggest difference

**Paper (end of Sec. 4.2):**
> "The network is trained using the standard cross-entropy loss."

**Code ([models/losses.py:125-217](models/losses.py#L125-L217), [main.py:99-110](main.py#L99-L110)):** `DarkActLoss` combines five terms:

```
L = 1.0·CE + 0.1·Sobel + 0.1·MSP + 0.05·MC + 0.05·SSIM
```

The [README](README.md) describes these losses (Sobel-fcc, MSP, Modality Contrastive, LPIPS/MS-SSIM), but the paper does not mention them at all.

**Why it may differ:**
- The README says the framework's features are also meant for "multi-modal video semantic embedding" and alignment. The extra losses look like they come from a later or extended version of the method (the README mentions DarkAct++), or from a pre-training setup.
- The weights (0.1 / 0.05) are not given anywhere, so they were probably chosen by hand.

**Differences between the README formulas and the code itself:**
- The README specifies **LPIPS + MS-SSIM**. The code uses only **single-scale SSIM** (no LPIPS, no multi-scale).
- SSIM compares `F_LAF` from the **last stage** (768 channels, 7×7) with `(Y^r+Y^t)/2` (96 channels, 56×56). The code upsamples F_LAF and keeps **only the first 96 of its 768 channels** ([losses.py:195-197](models/losses.py#L195-L197)). This is arbitrary and not defined in any formula.
- MSP reconstructs the 56×56 `Y` from 7×7 `F_LAF` using trilinear upsampling. That is a very lossy reconstruction target.

**Impact:** results from this code are not a reproduction of the paper's training. To match the paper, train with `weight_sobel = weight_msp = weight_mc = weight_ssim = 0`.

---

## 2. Patch Embedding position

**Paper (Fig. 3):**
```
Video → Modal Embedding → G → MAA (⊕G) → Y → Patch Embedding → Transformer Stage 1 → LAF …
```

**Code ([models/darkact_net.py:99-118](models/darkact_net.py#L99-L118)):**
```
Video → ModalEmbedding → G → MAA (+G) → Y ───────────────────→ Transformer Stage 1 → LAF …
```

`ModalEmbedding` ([backbone.py:14-20](models/backbone.py#L14-L20)) is `Conv3d(3→96, kernel=(3,7,7), stride=(2,4,4))`, which is exactly MViTv2's own `patch_embed`. The code has therefore **merged** "Modal Embedding" and "Patch Embedding" into one layer. After MAA, `TransformerStage` only flattens the grid into tokens; there is no learned layer (stage 1 uses `nn.Identity`).

**Why it may differ:**
- **Memory.** If MAA ran before downsampling, it would process 16×224×224 positions instead of 8×56×56, which is about 32 times more. MAA's frame differences, MLPs and C×C attention would become very expensive.
- The paper never defines what "Modal Embedding" contains, so the code chose the simplest version.

**Impact:** in the code, MAA works on coarse 4×4-pixel patches rather than finer features. Small motions may be harder to detect.

---

## 3. Backbone attention is not MViTv2

**Paper:** "we adopt MViTv2 [30] as the backbone".

**Code ([backbone.py:27-80](models/backbone.py#L27-L80)):** `EfficientSpatioTemporalAttention` uses **divided space-time attention**: first attention within each frame over H×W, then attention across T at each position. This is the TimeSformer design, not MViTv2.

Missing MViTv2 parts:
- **Pooling attention** (K/V pooled to reduce the token count)
- **Decomposed relative positional embeddings**
- **Residual pooling connection**
- **Any positional embedding at all.** The tokens only carry position indirectly, through the convolutions.

**Why it may differ:** with 8×56×56 = 25,088 tokens, full attention is impossible on GPU memory. The docstring says it was done to prevent "1000+ GB memory spikes". MViTv2 solves the same problem with pooling attention. This code solves it with factorized attention, which is simpler to write.

**Impact:** the backbone is a different model with the same name. The paper's numbers for MViTv2-B (Table 2) are not directly comparable with results from this code.

---

## 4. Backbone depth

| Variant | Real MViTv2 | Code ([backbone.py:177-191](models/backbone.py#L177-L191)) |
|---|---|---|
| MViTv2-B | depths [2, 3, 16, 3], 24 blocks | depths [2, 3, 6, 2], 13 blocks |
| MViTv2-S | depths [1, 2, 11, 2], 16 blocks | depths [1, 2, 4, 2], 9 blocks |
| Heads | [1, 2, 4, 8] | [2, 4, 8, 16] |

**Why it may differ:** the code comment says: *"To allow fast and scalable training on various environments, default depths can be customized"*. It was reduced for speed and memory.

**Impact:** `--backbone mvitv2_b` is about half as deep as the paper's model.

---

## 5. Downsampling between stages

**MViTv2:** reduces resolution by pooling the query inside the first attention block of each stage.

**Code ([backbone.py:126-130](models/backbone.py#L126-L130)):** a separate `Conv3d(kernel=(1,3,3), stride=(1,2,2)) + BN` before stages 2–4 (patch merging, Swin-style).

**Why:** this is simpler and works with the divided attention, which has no pooling.

---

## 6. MAA output (Eq. 3)

**Paper:**
```
Y = Softmax(E^Q (E^S)ᵀ / √C) · E^S          (Eq. 3)
```
Fig. 3 also shows `G` added (⊕) after "Salient Region Attention".

**Code ([maa.py:112-123](models/maa.py#L112-L123)):**
```python
salient = Softmax(E_Q @ E_Sᵀ / √C) @ E_S        # Eq. 3, shape (B, C, H, W)
gate    = Sigmoid(BN(Conv1x1(salient)))         # NOT in paper
Y       = G * gate + G                          # gated residual
```

**Why it may differ:**
- Eq. 3 produces a tensor of shape **C×N (no time axis)**, but `Y` must be C×T×H×W to feed the transformer. The paper doesn't say how T is restored. The code solves this by turning the result into a gate (one value per channel and pixel) and multiplying it with the full `G`, which still has T.
- The sigmoid gate keeps values in [0, 1], which makes training more stable than adding raw attention output.

---

## 7. MAA query mapping (Eq. 2)

**Paper:** `E^Q = Ē · W_M^A`. Since Ē ∈ ℝ^{C×N}, right-multiplying means **W is N×N**, which depends on the spatial resolution.

**Code ([maa.py:33](models/maa.py#L33), [maa.py:109](models/maa.py#L109)):** `W` is **C×C** and is left-multiplied: `E_Q = W @ Ē`.

**Why:** an N×N matrix (3136×3136 at 56×56) would be large, and it would break if the input resolution changed. A C×C matrix is small and works at any resolution. The code comment notes this choice.

---

## 8. MAA: how the time axis is removed

**Paper:** E^S and E^Q are in ℝ^{C×N}, so T must be removed somehow, but the method is unspecified.

**Code:**
- E^S: `Delta.mean(dim=2)` averages the motion maps over time, then applies the MLP ([maa.py:88](models/maa.py#L88)).
- E^Q: `G.max(dim=2)` and `G.mean(dim=2)`: Max/Avg pooling over **time** ([maa.py:99-100](models/maa.py#L99-L100)).

Eq. 1 (`N(|sin(G_t − G_{t−1})|)`, plus the first-last boundary difference) **matches the paper exactly** ([maa.py:50-72](models/maa.py#L50-L72)).

**Impact:** averaging over time means MAA produces **one spatial attention map per clip**, not per frame. A motion at the start and another at the end are blended together.

---

## 9. LAF works on time-averaged features

**Paper:** F^Q, F^K and F^V are in ℝ^{C×N}. The handling of T is unspecified.

**Code ([laf.py:139-142](models/laf.py#L139-L142), [laf.py:193-194](models/laf.py#L193-L194)):**
```python
F_r_2d = F_r.mean(dim=2)    # average over time
...
F_LAF = F_LAF_2d.unsqueeze(2).expand(-1, -1, T, -1, -1)   # copy same map to every frame
```

**Why:** this is cheaper (attention over H×W instead of T×H×W) and avoids memory issues in early stages.

**Impact:** this is significant for an *action* recognition model. The fusion signal added back to the stream is **identical for every frame**, so LAF cannot fuse modalities differently at different moments (for example when lighting changes mid-clip). Temporal modelling is left entirely to the backbone.

---

## 10–14. Smaller LAF details

| Item | Paper | Code | Likely reason |
|---|---|---|---|
| Key block (Eq. 5) | "3×3 dilated conv + Sigmoid" | Dilated conv + BN + PReLU + 1×1 conv + Sigmoid ([laf.py:63-76](models/laf.py#L63-L76)) | Fig. 3's legend shows "DilConv+PReLU+BN", so the code follows the figure. The 1×1 conv adds capacity. |
| ϕ mapping (Eq. 4) | "details in Appendix" (missing) | Conv3×3 + BN + PReLU + Conv1×1 ([laf.py:55-60](models/laf.py#L55-L60)) | Guess at a "light-weight mapping subnetwork" |
| Cross-attention (Eq. 6) | `Att(F^r, F^r, F^t)` | Multi-head (4 heads), learned Q/K/V/out projections ([laf.py:7-37](models/laf.py#L7-L37)) | Standard practice. Without projections, Q = K means attention is just self-similarity. |
| Value pooling (Eq. 6) | "hierarchical pooling" | 3×3 Max + Avg pooling, stride 1 ([laf.py:169-170](models/laf.py#L169-L170)) | Global pooling would destroy the N dimension that F^V needs (C×N), so local pooling keeps it. |
| Output norm | Not mentioned | `LayerNorm` on F_LAF ([laf.py:190](models/laf.py#L190)) | Keeps the residual addition at a stable scale |
| Residual | "F_LAF is added to the Transformer stage output" | Added to **both** F_r and F_t ([laf.py:197-198](models/laf.py#L197-L198)) | Consistent with the paper (it has two streams) |
| Y resizing | Not mentioned | Y (56×56, 96 ch) trilinear-resized to each stage's size + one 1×1 conv **shared** for Y^r and Y^t ([laf.py:126-135](models/laf.py#L126-L135)) | Y and F have different shapes at stages 2–4, so some resizing is unavoidable |

---

## 15. Weight sharing between streams

In Fig. 3, "Patch Embedding" and each "Transformer Stage" are drawn as **one box spanning both the RGB and thermal streams**, while "Modal Embedding" and "MAA" are drawn as two separate boxes. This could mean shared weights, or it could just be how the figure is drawn.

**Code:** everything is separate: `rgb_backbone` and `thermal_backbone` ([darkact_net.py:48-49](models/darkact_net.py#L48-L49)).

**Impact:** if the paper shares weights, the code has about twice as many backbone parameters as the paper's model. This is unclear from the paper.

---

## 16. Classifier head

**Paper:** "the output features of each modality from the final Transformer stage are **flattened** and concatenated, followed by a linear classifier".

**Code ([darkact_net.py:128-135](models/darkact_net.py#L128-L135), [darkact_net.py:77-80](models/darkact_net.py#L77-L80)):** global **average pooling** to (B, 768) per stream, concatenation to (B, 1536), then `Dropout(0.5)` and `Linear`.

**Why:** literal flattening would give 768 × 8 × 7 × 7 × 2 ≈ 600k features. That linear layer would have about 16M parameters and would overfit badly. Average pooling is what MViTv2 itself does. "Flattened" in the paper probably means "pooled and flattened into a vector".

---

## 17. Training hyper-parameters ✅

The defaults in [main.py:13-56](main.py#L13-L56) match Sec. 5.1: AdamW, weight decay 5e-3, 20 warm-up epochs + 100 cosine epochs, LR 1e-3 → 1e-4, batch size 8, 2 GPUs (DataParallel).

Small differences:
- Warm-up starts at `lr/20` and rises linearly. The paper doesn't specify the start value.
- The scheduler steps **per epoch**, not per iteration (unspecified in the paper).
- DataParallel splits batch 8 across 2 GPUs, so 4 per GPU. The paper's "batch size 8" may mean per GPU (the `--batch_size` help text says "per GPU", but the code uses it as the total batch).
- The model is trained from scratch ("random init."), which **matches** Table 2.
- The best checkpoint is chosen using the **test set** (evaluated every 5 epochs). The paper doesn't describe model selection, but this gives optimistic accuracy.

---

## 18. Input pipeline (paper unspecified)

[datasets/darkact_dataset.py](datasets/darkact_dataset.py):
- 16 frames per clip at 224×224. Frames are **resized** without cropping, so the aspect ratio is distorted, even though the argument is called `crop_size`.
- The only augmentation is horizontal flip, synchronized between RGB and thermal.
- Thermal uses the same normalization as RGB (mean 0.45, std 0.225, from Kinetics).
- Thermal is read as 3 channels, and `ModalEmbedding` is applied with `in_channels=3`.

These are reasonable defaults, but the paper's appendix may use stronger augmentation. The current train/test gap (about 96% vs about 62% Top-1 in `outputs/train.log`) suggests more augmentation is needed.

---

## 19. Class list

The README's list (and presumably the paper's appendix) has **"clapping"**. `DARKACT_CLASSES` in the code has **`"slap"`** instead, and no "clap" ([darkact_dataset.py:31](datasets/darkact_dataset.py#L31)). If the dataset folder is named `clap`/`clapping`, folder discovery will not find it in `CLASS_TO_IDX` and will **silently label it as class 0 (`call`)** ([darkact_dataset.py:155](datasets/darkact_dataset.py#L155)). Check the real folder name. With `annotations/*.txt` files present, labels come from the file and this doesn't matter.

---

## 20. Benchmark scripts are not the paper's baselines

**Paper Table 2:** compares real published models: Conv2Former, MViTv2-B, VideoSwin, MRFS, CMX, DFormerv2, and others.

**Code:**
- [scripts/benchmark_unimodal.py](scripts/benchmark_unimodal.py): `--backbone conv2former` is only a name. The model is always this repo's simplified backbone.
- [scripts/benchmark_multimodal.py](scripts/benchmark_multimodal.py): `GenericMultimodalBaseline("mrfs")` is a two-stream backbone with late concatenation. The `model_name` argument is **not used** to build anything different.

**Why:** re-implementing more than 20 external models is out of scope. These scripts are placeholders.

**Impact:** you cannot reproduce Table 2 with them. The multimodal baseline is effectively the "w/o MAA + LAF" ablation.

---

## Related implementation issues (not paper differences)

These are not paper-vs-code differences, but they affect whether the code can reproduce the paper.

1. **RGB and thermal frames are sampled at different times during training.** `read_video_frames(..., sample_mode="random")` is called separately for RGB and for thermal ([darkact_dataset.py:219-230](datasets/darkact_dataset.py#L219-L230)). Each call draws its own `random.randint` indices ([darkact_dataset.py:67](datasets/darkact_dataset.py#L67)), so the two streams get **different frames**. This breaks the frame-level RGB–T pairing that the paper emphasizes, and it also adds noise to MAA's temporal differences and the Sobel/contrastive losses. The fix is to sample the indices once and use them for both videos.
2. **Missing thermal videos are filled with the RGB video.** In folder-discovery mode, if no thermal match is found the sample uses the RGB video as "thermal" ([darkact_dataset.py:169](datasets/darkact_dataset.py#L169)), and no warning is printed.
3. **The train/test split depends on annotations.** Without `annotations/train.txt` and `annotations/test.txt`, the code makes a random 80/20 split instead of the official 9,040 / 3,738 split. An earlier run in `outputs/train.log` used 10,222 / 2,556 (random split); the latest run uses the official 9,040 / 3,738.
4. **The latest run's LR config is inverted.** The last run in the log uses `lr=1e-5` with `min_lr=1e-4`. The cosine schedule then **increases** the LR from 1e-5 to 1e-4 instead of decaying it, and it doesn't match the paper's 1e-3 → 1e-4.

---

## Matching the paper more closely

In rough order of impact on reproduction:

1. Train with CE only (set the auxiliary loss weights to 0).
2. Fix the synchronized frame sampling for RGB and thermal.
3. Use the real MViTv2-B depths [2, 3, 16, 3], or better, the official MViTv2 implementation (e.g. from `pytorchvideo` / `timm`) with pooling attention and relative position.
4. Make LAF operate per frame instead of on the time average.
5. Add a separate Patch Embedding after MAA, if memory allows.
6. Use the official annotation split and the paper's LR (1e-3 → 1e-4).

Several of these will make existing checkpoints in `outputs/checkpoints/` incompatible.
