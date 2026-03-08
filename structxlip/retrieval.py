import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from structxlip.utils.func import longclip_pos_embeddings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ALIAS: Dict[str, str] = {
    "L-336": "openai/clip-vit-large-patch14-336",
    "L": "openai/clip-vit-large-patch14",
    "B": "openai/clip-vit-base-patch16",
    "G": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
}

DATASET_PRESETS: Dict[str, Tuple[str, Optional[str]]] = {
    "docci": ("datasets/docci_test.json", None),
    "coco": ("datasets/coco_test.json", None),
    "flickr30k": ("datasets/flickr30k_test.json", None),
    "DCI": ("datasets/DCI_test.json", None),
    "urban": ("datasets/urban_dataset_test.json", None),
    "sharegpt4v": ("datasets/sharegpt4v_test.json", None),
    "sketchy_global": (
        "/mnt/data/zruan/workspace_novel/zruan/LOST/fashionpedia_clip_dataset_test/global/metadata_with_scribbles_v1.0_updated.json",
        "/mnt/data/zruan/workspace_novel/zruan/LOST/fashionpedia_clip_dataset_test/",
    ),
    "sketchy_local": (
        "/mnt/data/zruan/workspace_novel/zruan/LOST/fashionpedia_clip_dataset_test/local/metadata_modified.json",
        "/mnt/data/zruan/workspace_novel/zruan/LOST/fashionpedia_clip_dataset_test/",
    ),
}


class QueryDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        processor,
        max_token_length: int,
        base_path: Optional[str] = None,
    ):
        self.records = records
        self.processor = processor
        self.base_path = base_path
        self.max_token_length = max_token_length

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, idx: int) -> str:
        image_path = self.records[idx]["original_filename"]
        if self.base_path is not None and not os.path.isabs(image_path):
            image_path = os.path.join(self.base_path, image_path)
        return image_path

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()

        image_path = self._resolve_image_path(idx)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        caption = self.records[idx]["original_caption"]
        encoded = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_length,
        )
        return encoded.pixel_values[0], encoded.input_ids[0]


def resolve_model_name(model_arg: str) -> str:
    return MODEL_ALIAS.get(model_arg, model_arg)


def resolve_dataset(dataset_arg: str) -> Tuple[str, Optional[str]]:
    if dataset_arg in DATASET_PRESETS:
        return DATASET_PRESETS[dataset_arg]

    if os.path.exists(dataset_arg):
        return dataset_arg, infer_base_path(dataset_arg)

    raise FileNotFoundError(
        f"Dataset '{dataset_arg}' is neither a known preset nor a valid JSON file path."
    )


def infer_base_path(dataset_json_path: str) -> Optional[str]:
    try:
        records = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
        if not records or "original_filename" not in records[0]:
            return None

        probe = records[0]["original_filename"]
        dataset_dir = Path(dataset_json_path).resolve().parent

        if (dataset_dir / probe).exists():
            return str(dataset_dir)
        if (dataset_dir.parent / probe).exists():
            return str(dataset_dir.parent)
    except Exception:
        return None

    return None


def extract_state_dict(ckpt_data: dict) -> dict:
    if "model_state_dict" in ckpt_data:
        return ckpt_data["model_state_dict"]

    if any(k.startswith("clip_model.") for k in ckpt_data.keys()):
        state = {}
        for k, v in ckpt_data.items():
            if k.startswith("clip_model."):
                state[k[len("clip_model.") :]] = v
            else:
                state[k] = v
        return state

    return ckpt_data


def load_checkpoint_if_needed(model, ckpt_path: str, zero_shot: bool) -> None:
    if zero_shot:
        return

    if not ckpt_path:
        raise ValueError("Please provide --ckpt or enable --zero_shot.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint_data = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint_data)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(
        f"Loaded checkpoint: {ckpt_path} | missing_keys={len(missing_keys)} | "
        f"unexpected_keys={len(unexpected_keys)}"
    )


def compute_recall(ranks: torch.Tensor, k: int) -> float:
    return (ranks < k).float().mean().item() * 100.0


@torch.no_grad()
def evaluate(fabric: L.Fabric, model, query_loader: DataLoader) -> None:
    fabric.print("Running retrieval evaluation...")

    image_embeds_all: List[torch.Tensor] = []
    text_embeds_all: List[torch.Tensor] = []

    for image_pixels, input_ids in query_loader:
        image_pixels = image_pixels.to(fabric.device)
        input_ids = input_ids.to(fabric.device)

        outputs = model(pixel_values=image_pixels, input_ids=input_ids)
        image_embeds_all.append(F.normalize(outputs.image_embeds).cpu())
        text_embeds_all.append(F.normalize(outputs.text_embeds).cpu())

    if not image_embeds_all or not text_embeds_all:
        fabric.print("No embeddings extracted. Abort evaluation.")
        return

    image_embeds = torch.cat(image_embeds_all, dim=0).to(fabric.device)
    text_embeds = torch.cat(text_embeds_all, dim=0).to(fabric.device)

    if image_embeds.size(0) != text_embeds.size(0):
        fabric.print(
            "Warning: image/text embedding counts do not match. "
            "Recall assumes paired ordering."
        )

    similarity = image_embeds @ text_embeds.t()
    i2t_scores = similarity
    t2i_scores = similarity.t()

    num_samples = image_embeds.shape[0]
    gt = torch.arange(num_samples, device=fabric.device)

    i2t_sorted = torch.argsort(i2t_scores, dim=1, descending=True)
    t2i_sorted = torch.argsort(t2i_scores, dim=1, descending=True)

    i2t_ranks = (i2t_sorted == gt[:, None]).nonzero(as_tuple=True)[1]
    t2i_ranks = (t2i_sorted == gt[:, None]).nonzero(as_tuple=True)[1]

    ks = [1, 5, 10, 25, 50]
    i2t_recalls = {k: compute_recall(i2t_ranks, k) for k in ks}
    t2i_recalls = {k: compute_recall(t2i_ranks, k) for k in ks}

    fabric.print(
        "Text-to-Image: "
        + " | ".join([f"R@{k} {t2i_recalls[k]:.2f}" for k in ks])
    )
    fabric.print(
        "Image-to-Text: "
        + " | ".join([f"R@{k} {i2t_recalls[k]:.2f}" for k in ks])
    )


def main(args):
    fabric = L.Fabric(accelerator="cuda", devices=1, precision=args.precision)
    fabric.launch()
    fabric.seed_everything(args.seed + fabric.global_rank)

    model_name_hf = resolve_model_name(args.model)
    print(f"Resolved model: {model_name_hf}")

    processor = transformers.AutoProcessor.from_pretrained(model_name_hf)
    model = transformers.AutoModel.from_pretrained(
        model_name_hf,
        attn_implementation="eager",
    ).bfloat16()

    longclip_pos_embeddings(model, args.new_max_token)
    load_checkpoint_if_needed(model, args.ckpt, args.zero_shot)

    model = model.to(fabric.device)
    model.eval()

    query_list_path, base_path = resolve_dataset(args.dataset)
    if not os.path.exists(query_list_path):
        raise FileNotFoundError(f"Dataset JSON not found: {query_list_path}")

    query_records = json.loads(Path(query_list_path).read_text(encoding="utf-8"))
    query_dataset = QueryDataset(
        records=query_records,
        processor=processor,
        max_token_length=args.new_max_token,
        base_path=base_path,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    evaluate(fabric, model, query_loader)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="StructXLIP CLIP retrieval evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sketchy_global",
        help="Dataset preset name or a JSON file path.",
    )
    parser.add_argument(
        "--new_max_token",
        type=int,
        default=248,
        help="Maximum text token length used in training/eval.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="B",
        help="Model alias (B, L, L-336, G) or a HF model name/path.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Fine-tuned checkpoint (.pth). Ignored if --zero_shot is enabled.",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Evaluate pretrained model without loading checkpoint.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.zero_shot and not args.ckpt:
        parser.error("Please specify --ckpt, or pass --zero_shot.")

    main(args)
