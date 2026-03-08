import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


BBOX_KEYS = ("x1", "y1", "x2", "y2", "width", "height")


def _processor_crop_size(processor) -> Tuple[int, int]:
    crop = getattr(processor.image_processor, "crop_size", None)
    if isinstance(crop, dict):
        height = int(crop.get("height", crop.get("shortest_edge", 224)))
        width = int(crop.get("width", crop.get("shortest_edge", 224)))
        return height, width
    if isinstance(crop, int):
        return crop, crop
    return 224, 224


class StructXLIPDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        processor,
        max_token_length: int,
        chunk_top_k: int,
    ):
        self.records = records
        self.processor = processor
        self.max_token_length = max_token_length
        self.chunk_top_k = max(1, int(chunk_top_k))

        height, width = _processor_crop_size(processor)
        self._dummy_rgb = Image.new("RGB", (width, height), (0, 0, 0))

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Optional[str], mode: str) -> Tuple[Optional[Image.Image], Tuple[int, int]]:
        if not path or not os.path.exists(path):
            return None, (0, 0)
        try:
            image = Image.open(path).convert(mode)
            return image, image.size
        except Exception:
            return None, (0, 0)

    def _build_bbox(self, segment: Dict[str, Any]) -> Dict[str, float]:
        bbox = segment.get("bbox_coordinates", {}) or {}
        return {k: float(bbox.get(k, 0.0)) for k in BBOX_KEYS}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]

        org_photo, org_photo_size = self._load_image(item.get("original_filename"), mode="RGB")
        org_caption = item.get("original_caption", "")
        llm_caption = item.get("original_caption", "")

        org_sketch, _ = self._load_image(item.get("original_filename_structure"), mode="L")

        segments = item.get("segment", [])
        if not segments:
            segments = [
                {
                    "similarity_score": 0.0,
                    "filename": None,
                    "caption": "",
                    "bbox_coordinates": {k: 0.0 for k in BBOX_KEYS},
                }
            ]

        primary_segment = max(segments, key=lambda x: x.get("similarity_score", 0.0))
        seg_photo, _ = self._load_image(primary_segment.get("filename"), mode="RGB")
        seg_caption = primary_segment.get("caption", "")
        seg_sketch, _ = self._load_image(primary_segment.get("filename_structure_cropped"), mode="L")
        bbox = self._build_bbox(primary_segment)

        ranked_segments = sorted(
            segments,
            key=lambda x: x.get("similarity_score", float("-inf")),
            reverse=True,
        )
        top_segments = ranked_segments[: self.chunk_top_k]

        org_data = self.processor(
            images=org_photo if org_photo is not None else self._dummy_rgb,
            text=org_caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_length,
        )
        seg_data = self.processor(
            images=seg_photo if seg_photo is not None else self._dummy_rgb,
            text=seg_caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_length,
        )

        if org_sketch is not None:
            org_scribble_pixels = self.processor(
                images=org_sketch.convert("RGB"),
                return_tensors="pt",
            ).pixel_values[0]
            has_org_scribble = torch.tensor(True)
        else:
            org_scribble_pixels = self.processor(images=self._dummy_rgb, return_tensors="pt").pixel_values[0]
            has_org_scribble = torch.tensor(False)

        if seg_sketch is not None:
            seg_scribble_pixels = self.processor(
                images=seg_sketch.convert("RGB"),
                return_tensors="pt",
            ).pixel_values[0]
            has_seg_scribble = torch.tensor(True)
        else:
            seg_scribble_pixels = self.processor(images=self._dummy_rgb, return_tensors="pt").pixel_values[0]
            has_seg_scribble = torch.tensor(False)

        edge_imgs: List[Image.Image] = []
        valid_flags: List[bool] = []
        for k in range(self.chunk_top_k):
            edge_path = None
            if k < len(top_segments):
                edge_path = top_segments[k].get("filename_structure_cropped")

            if edge_path and os.path.exists(edge_path):
                edge_img, _ = self._load_image(edge_path, mode="L")
                edge_imgs.append(edge_img.convert("RGB") if edge_img is not None else self._dummy_rgb)
                valid_flags.append(edge_img is not None)
            else:
                edge_imgs.append(self._dummy_rgb)
                valid_flags.append(False)

        edge_scribble_pixels = torch.stack(
            [self.processor(images=img, return_tensors="pt").pixel_values[0] for img in edge_imgs],
            dim=0,
        )
        edge_valid_mask = torch.tensor(valid_flags, dtype=torch.bool)

        return {
            "org_image": org_data.pixel_values[0],
            "org_text": org_data.input_ids[0],
            "seg_image": seg_data.pixel_values[0],
            "seg_text": seg_data.input_ids[0],
            "bbox": bbox,
            "org_caption": org_caption,
            "seg_caption": seg_caption,
            "llm_caption": llm_caption,
            "org_scribble_pixels": org_scribble_pixels,
            "seg_scribble_pixels": seg_scribble_pixels,
            "has_org_scribble": has_org_scribble,
            "has_seg_scribble": has_seg_scribble,
            "org_width": int(org_photo_size[0]),
            "org_height": int(org_photo_size[1]),
            "edge_scribble_pixels": edge_scribble_pixels,
            "edge_valid_mask": edge_valid_mask,
        }
