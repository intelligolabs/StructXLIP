# StructXLIP — Enhancing Vision-Language Models with Multimodal Structural Cues

<p align="center">
  <img src="images/logo2.png" width="380" alt="StructXLIP">
</p>

<p align="center">
  <b>CVPR 2026</b> &nbsp;·&nbsp;
  <a href="README_ZH.md">简体中文</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.20089"><img src="https://img.shields.io/badge/arXiv-2602.20089-b31b1b.svg"></a>
  <a href="https://eveleslie.github.io/structxlip-web/"><img src="https://img.shields.io/badge/Project-Page-4a90d9.svg"></a>
  <a href="https://huggingface.co/zanxii/StructXlip"><img src="https://img.shields.io/badge/🤗_Weights-zanxii/StructXlip-f5a623.svg"></a>
  <a href="https://huggingface.co/datasets/zanxii/StructXlip"><img src="https://img.shields.io/badge/🤗_Dataset-zanxii/StructXlip-e07b00.svg"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-3776ab.svg"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg"></a>
</p>

---

<p align="center">
  <img src="images/method.png" width="860" alt="Method overview">
</p>

StructXLIP augments CLIP-style contrastive learning with **structural cues** (scribble maps, edge maps etc) alongside RGB images. 
Three new training objectives are implemented in [`structxlip/losses.py`](structxlip/losses.py) and packaged as drop-in PyTorch modules in [`plug_and_play_loss.py`](plug_and_play_loss.py):

| Objective | Function |
|---|---|
| Structure-Centric Alignment | `compute_structure_centric_loss` | Contrastive alignment: global structure ↔ caption |
| RGB–Structure Consistency | `compute_rgb_structure_consistency_loss` | Cosine consistency: RGB features ↔ structure features |
| Local Structure-Centric Alignment | `compute_local_structure_centric_loss` |

---

## 🗂️ Repository Structure

```
StructXLIP/
├── plug_and_play_loss.py        # ⭐ standalone loss modules
├── structxlip/
│   ├── train.py                 # python -m structxlip.train
│   ├── retrieval.py             # python -m structxlip.retrieval
│   ├── dataloader.py            # JSON → tensors (RGB + structure)
│   ├── losses.py                # full loss implementations
│   ├── text_filters.py          # caption filtering utilities
│   └── utils/func.py            # long-token positional embedding, etc.
├── scripts/
│   ├── finetune.sh
│   ├── eval.sh
│   └── package_sketchy_to_hf.py
├── datasets/                    # example JSON lists
├── weights/                     # local checkpoints
└── requirements.txt
```

---
## Quick Start — Retrieval Evaluation on Sketchy

**Step 1.** Install

```bash
conda create -n structxlip python=3.10 && conda activate structxlip
pip install -r requirements.txt
```

**Step 2.** Download weights and test data

```bash
hf download zanxii/StructXlip Sketchy.pth --repo-type model --local-dir weights
hf download zanxii/StructXlip sketchy_test.zip --repo-type dataset --local-dir data/structxlip
```

**Step 3.** Unzip and rewrite local paths

```bash
mkdir -p data/structxlip/sketchy_test_images
unzip -q data/structxlip/sketchy_test.zip -d data/structxlip/sketchy_test_images

python - <<'PY'
import json
from pathlib import Path
src  = Path("datasets/test/Sketchy.json")
out  = Path("datasets/test/Sketchy_local.json")
imgs = Path("data/structxlip/sketchy_test_images").resolve()
data = json.loads(src.read_text())
for r in data:
    r["original_filename"] = str(imgs / r["file_name"])
out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print(f"Written: {out}  ({len(data)} items)")
PY
```

**Step 4.** Run evaluation

```bash
python -m structxlip.retrieval \
  --dataset datasets/test/Sketchy_local.json \
  --ckpt    weights/Sketchy.pth \
  --model   B \
  --eval_batch_size 32
```

---

## Training

```bash
python -m structxlip.train \
  --dataset    /path/to/train.json \
  --model      openai/clip-vit-base-patch16 \
  --output_dir outputs/ckpt \
  --epochs     10 \
  --batch_size 16
```

Or via the launcher script:

```bash
DATASET_JSON=/path/to/train.json \
OUTPUT_DIR=outputs/ckpt \
WANDB_PROJECT=StructXLIP \
bash scripts/finetune.sh
```

<details>
<summary><b>Key training arguments</b></summary>

| Argument | Description |
|---|---|
| `--lambda_global` | Weight for standard CLIP loss |
| `--lambda_structure_centric` | Weight for Structure-Centric Alignment |
| `--lambda_rgb_scribble_consistency` | Weight for RGB–Structure Consistency |
| `--lambda_local_structure_centric` | Weight for Local Structure-Centric Alignment |
| `--chunk_top_k / --chunk_tau / --chunk_base_window / --chunk_stride` | Local alignment controls |
| `--remove_colors / --remove_materials / --remove_textures / --remove_insect` | Caption filtering (see `text_filters.py`) |
| `--warmup_sketch_epochs` | Warm up with structure losses only |
| `--new_max_token` | Extend CLIP text positional embeddings |

</details>

<details>
<summary><b>Training JSON format</b></summary>

```json
[
  {
    "original_filename": "/abs/path/rgb.jpg",
    "original_caption": "full caption",
    "original_filename_structure": "/abs/path/global_structure.png",
    "segment": [
      {
        "similarity_score": 0.87,
        "filename": "/abs/path/local_crop_rgb.jpg",
        "caption": "local region caption",
        "filename_structure_cropped": "/abs/path/local_structure.png",
        "bbox_coordinates": { "x1": 0, "y1": 0, "x2": 0, "y2": 0, "width": 0, "height": 0 }
      }
    ]
  }
]
```


</details>

## 🔌 Plug-and-Play Losses

All three objectives are available as self-contained PyTorch modules in [`plug_and_play_loss.py`](plug_and_play_loss.py) — **only PyTorch required** for Structure-Centric Alignment and RGB–Structure Consistency.

> **Recommended starting point:** begin with `StructureCentricAlignmentLoss` — it is the simplest to integrate and gives the most direct structural signal.

```python
from plug_and_play_loss import (
    StructureCentricAlignmentLoss,   # recommended first
    RGBStructureConsistencyLoss,
    LocalStructureCentricLoss,       # requires model + tokenizer
    cosine_anneal_warm_decay,        # optional loss weight schedule
)

structure_centric = StructureCentricAlignmentLoss()
rgb_consistency   = RGBStructureConsistencyLoss()
local_structure   = LocalStructureCentricLoss(chunk_base_window=3, chunk_tau=0.07)

loss = (loss_clip
        + λ1 * structure_centric(scribble_emb, text_emb, has_struct, logit_scale)
        + λ2 * rgb_consistency(image_emb, scribble_emb, has_struct)
        + λ3 * local_structure(model, text_tokens, captions, edge_emb_flat, edge_mask, tokenizer)[0])
```

Smoke-test with random tensors:

```bash
python plug_and_play_loss.py
```

---



## Weights & Data

**Weights** — [zanxii/StructXlip](https://huggingface.co/zanxii/StructXlip)

| Checkpoint | Dataset |
|---|---|
| `Sketchy.pth` | Sketchy |
| `DCI.pth` | DCI |
| `DOCCI.pth` | DOCCI |
| `Insect.pth` | Insect |

```bash
hf download zanxii/StructXlip <checkpoint>.pth --repo-type model --local-dir weights
```

**Dataset** — [zanxii/StructXlip](https://huggingface.co/datasets/zanxii/StructXlip)

Test set images for all four benchmarks are available on Hugging Face. See the dataset repo for the full file listing.

```bash
hf download zanxii/StructXlip <file>.zip --repo-type dataset --local-dir data/structxlip
```

---


## Release Status

**Code**
- [x] Training code (`structxlip/train.py`, `losses.py`)
- [x] Retrieval evaluation code (`structxlip/retrieval.py`)
- [ ] Structure edge extraction & preprocessing scripts

**Checkpoints**
- [x] Model weights: Sketchy / DCI / DOCCI / Insect

**Data**

| Dataset | Train | Test |
|---------|:-----:|:----:|
| Sketchy | ✅ | ✅ |
| DOCCI   | —  | ✅ |
| DCI     | —  | ✅ |
| Insect  | —  | — |

**Plug-and-play losses**
- [x] Structure-Centric Alignment
- [ ] RGB–Structure Consistency
- [ ] Local Structure-Centric Alignment

---

## 🎉 Acknowledgements

We thank the authors of [CLIP](https://github.com/openai/CLIP), [LongCLIP](https://github.com/beichenzbc/Long-CLIP), for their excellent open-source work, which this project builds upon.

This section is being updated as we continue to open-source components of this project. *(More acknowledgements coming soon.)*

---

## Citation

```bibtex
@inproceedings{ruan2026StruXLIP,
  title     = {StructXLIP: Enhancing Vision-Language Models
               with Multimodal Structural Cues},
  author    = {Ruan, Zanxi and Gao, Songqun and Kong, Qiuyu
               and Wang, Yiming and Cristani, Marco},
  booktitle = {Proceedings of the IEEE/CVF Conference on
               Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
