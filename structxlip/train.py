import argparse
import json
import math
import os
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import transformers
import wandb

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from structxlip.dataloader import StructXLIPDataset
from structxlip.losses import (
    clip_loss,
    compute_local_structure_centric_loss,
    compute_rgb_structure_consistency_loss,
    compute_structure_centric_loss,
    cosine_anneal_warm_decay,
)
from structxlip.text_filters import VisualTermFilter
from structxlip.utils.func import batch_align, longclip_pos_embeddings, print_trainable_parameters

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("StructXLIP CLIP Fine-tuning")

    parser.add_argument("--dataset", required=True, type=str, help="Path to the training JSON file")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to save checkpoints and logs")
    parser.add_argument("--model", default="openai/clip-vit-base-patch16", type=str)

    parser.add_argument("--remove_insect", action="store_true", help="Remove insect-specific visual terms")
    parser.add_argument("--remove_colors", action="store_true", help="Remove color terms")
    parser.add_argument("--remove_materials", action="store_true", help="Remove material terms")
    parser.add_argument("--remove_textures", action="store_true", help="Remove texture terms")

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--init_lr", type=float, default=5e-6)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--new_max_token", default=248, type=int)

    parser.add_argument("--lambda_global", type=float, default=1.0)
    parser.add_argument("--lambda_structure_centric", type=float, default=0.5)
    parser.add_argument("--lambda_rgb_scribble_consistency", type=float, default=0.05)
    parser.add_argument("--lambda_local_structure_centric", type=float, default=0.1)

    parser.add_argument("--lambda_scribble_text", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lambda_chunk", type=float, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--chunk_top_k", type=int, default=3)
    parser.add_argument("--chunk_tau", type=float, default=0.07)
    parser.add_argument("--chunk_base_window", type=int, default=3)
    parser.add_argument("--chunk_stride", type=int, default=1)

    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--warmup_sketch_epochs", default=0, type=int)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--wandb_project", type=str, default="StructXLIP")

    return parser


def resolve_legacy_aliases(args: argparse.Namespace) -> argparse.Namespace:
    if args.lambda_scribble_text is not None:
        args.lambda_structure_centric = args.lambda_scribble_text
    if args.lambda_chunk is not None:
        args.lambda_local_structure_centric = args.lambda_chunk

    args.lambda_scribble_text = args.lambda_structure_centric
    args.lambda_chunk = args.lambda_local_structure_centric
    return args


def _init_wandb(args: argparse.Namespace):
    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=getattr(args, "wandb_run_name", None),
            config=vars(args),
        )


def _build_train_loader(records, processor, fabric: L.Fabric, args: argparse.Namespace):
    dataset = StructXLIPDataset(
        records=records,
        processor=processor,
        max_token_length=args.new_max_token,
        chunk_top_k=args.chunk_top_k,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )
    return fabric.setup_dataloaders(loader)


def run_training(args: argparse.Namespace) -> None:
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    _init_wandb(args)

    fabric = L.Fabric(accelerator="cuda", devices=1, strategy="auto", precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(args.seed)

    if fabric.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset, "r", encoding="utf-8") as fp:
        train_records = json.load(fp)

    with fabric.device:
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.CLIPModel.from_pretrained(args.model)
        longclip_pos_embeddings(model, args.new_max_token)
        print_trainable_parameters(fabric, model)

    train_loader = _build_train_loader(train_records, processor, fabric, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    text_filter = VisualTermFilter(min_content_tokens=2)
    total_steps = len(train_loader) * args.epochs
    global_step = 0

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            lr = (args.init_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * global_step / total_steps)) + args.min_lr
            for group in optimizer.param_groups:
                group["lr"] = lr

            org_image = batch["org_image"]
            org_text = batch["org_text"]
            llm_caption = batch["llm_caption"]

            org_scribble_pixels = batch["org_scribble_pixels"]
            has_org_scribble = batch["has_org_scribble"]

            edge_scribble_pixels = batch["edge_scribble_pixels"]
            edge_valid_mask = batch["edge_valid_mask"]

            batch_size = org_image.shape[0]
            top_k = edge_scribble_pixels.shape[1]
            eps = 1e-8

            filtered_texts = text_filter.filter_batch(
                llm_caption,
                remove_insect=args.remove_insect,
                remove_colors=args.remove_colors,
                remove_materials=args.remove_materials,
                remove_textures=args.remove_textures,
            )
            enc_org_filtered = processor(
                text=filtered_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.new_max_token,
            ).input_ids.to(fabric.device)
            edge_flat = edge_scribble_pixels.reshape(batch_size * top_k, *edge_scribble_pixels.shape[2:])
            images_cat = torch.cat((org_image, org_scribble_pixels, edge_flat), dim=0)
            texts_cat = torch.cat((org_text, enc_org_filtered), dim=0)

            outputs = model(pixel_values=images_cat, input_ids=texts_cat, output_hidden_states=True, return_dict=True)
            text_outputs = outputs.text_model_output
            image_embeds_all = outputs.image_embeds
            text_embeds_all = outputs.text_embeds

            org_image_embeds = image_embeds_all[:batch_size]
            org_scribble_embeds = image_embeds_all[batch_size : 2 * batch_size]
            edge_seg_embeds_flat = image_embeds_all[2 * batch_size :]

            org_text_embeds, filtered_org_text_embeds = torch.chunk(text_embeds_all, 2)
            org_text_tokens = text_outputs.last_hidden_state[:batch_size]

            logit_scale_main = model.logit_scale.exp()
            logit_scale_aux = logit_scale_main.detach()

            x_i_org = batch_align(
                fabric,
                F.normalize(org_image_embeds + eps, dim=-1),
            )
            x_t_org = batch_align(
                fabric,
                F.normalize(org_text_embeds + eps, dim=-1),
            )

            loss_org = clip_loss(logit_scale_main * (x_i_org @ x_t_org.t()))

            loss_structure_centric = compute_structure_centric_loss(
                org_scribble_embeds,
                filtered_org_text_embeds,
                has_org_scribble,
                logit_scale=logit_scale_aux,
                eps=eps,
            )
            loss_rgb_scribble_consistency = compute_rgb_structure_consistency_loss(
                org_image_embeds,
                org_scribble_embeds,
                has_org_scribble,
                eps=eps,
            )
            loss_local_structure_centric, local_debug = compute_local_structure_centric_loss(
                model,
                org_text_tokens,
                filtered_texts,
                edge_seg_embeds_flat,
                edge_valid_mask,
                processor.tokenizer,
                chunk_base_window=args.chunk_base_window,
                chunk_stride=args.chunk_stride,
                chunk_tau=args.chunk_tau,
                eps=eps,
            )

            if (global_step < 5 or global_step % 100 == 0) and local_debug["valid_items"] > 0:
                fabric.print(
                    f"[LSC DEBUG] step={global_step} eff_scale={local_debug['effective_scale']:.2f} | "
                    f"chunks={local_debug['avg_chunks']:.1f} pos={local_debug['avg_pos']:.1f} | "
                    f"pos_mass={local_debug['pos_mass_ratio']:.3f} margin={local_debug['top1_margin']:.3f} "
                    f"entropy={local_debug['entropy']:.3f}"
                )
                if wandb.run:
                    wandb.log(
                        {
                            "lsc/effective_scale": local_debug["effective_scale"],
                            "lsc/avg_chunks": local_debug["avg_chunks"],
                            "lsc/avg_pos": local_debug["avg_pos"],
                            "lsc/pos_mass_ratio": local_debug["pos_mass_ratio"],
                            "lsc/top1_margin": local_debug["top1_margin"],
                            "lsc/entropy": local_debug["entropy"],
                        }
                    )

            lam_structure_centric = cosine_anneal_warm_decay(
                args.lambda_structure_centric,
                epoch,
                warm=1,
                decay_start=1,
                decay_end=8,
                floor=0.7,
            )
            lam_rgb = cosine_anneal_warm_decay(
                args.lambda_rgb_scribble_consistency,
                epoch,
                warm=1,
                decay_start=1,
                decay_end=8,
                floor=0.5,
            )
            lam_local_structure_centric = cosine_anneal_warm_decay(
                args.lambda_local_structure_centric,
                epoch,
                warm=1,
                decay_start=1,
                decay_end=8,
                floor=0.4,
            )

            if epoch < args.warmup_sketch_epochs:
                if fabric.global_rank == 0 and batch_idx == 0:
                    fabric.print(f"\nEpoch {epoch} - Phase 1: Sketch-only warmup\n")
                total_loss = (
                    lam_structure_centric * loss_structure_centric
                    + lam_rgb * loss_rgb_scribble_consistency
                    + lam_local_structure_centric * loss_local_structure_centric
                )
            else:
                if fabric.global_rank == 0 and batch_idx == 0:
                    fabric.print(f"\nEpoch {epoch} - Phase 2: Joint training\n")
                total_loss = (
                    args.lambda_global * loss_org
                    + lam_structure_centric * loss_structure_centric
                    + lam_rgb * loss_rgb_scribble_consistency
                    + lam_local_structure_centric * loss_local_structure_centric
                )

            fabric.backward(total_loss)
            optimizer.step()
            with torch.no_grad():
                if hasattr(model, "logit_scale"):
                    model.logit_scale.data.clamp_(0, 3.5)
            optimizer.zero_grad()

            if fabric.global_rank == 0 and wandb.run:
                wandb.log(
                    {
                        "iter": global_step,
                        "epoch": epoch,
                        "lr": lr,
                        "loss_total": total_loss.detach().item(),
                        "loss_org": loss_org.detach().item(),
                        "loss_structure_centric": loss_structure_centric.detach().item(),
                        "loss_rgb_scribble_consistency": loss_rgb_scribble_consistency.detach().item(),
                        "loss_local_structure_centric": loss_local_structure_centric.detach().item(),
                        "lambda_structure_centric_eff": float(lam_structure_centric),
                        "lambda_rgb_eff": float(lam_rgb),
                        "lambda_local_structure_centric_eff": float(lam_local_structure_centric),
                        "lambda_global": float(args.lambda_global),
                    }
                )

            if global_step % 10 == 0 or batch_idx == len(train_loader) - 1:
                fabric.print(
                    f"E{epoch} I{global_step} [{global_step / total_steps * 100:.1f}%] "
                    f"lr {lr:.2e} loss {total_loss.detach().item():.3f} | "
                    f"org {loss_org.detach().item():.3f} | "
                    f"SC {loss_structure_centric.detach().item():.3f} "
                    f"RS {loss_rgb_scribble_consistency.detach().item():.3f} "
                    f"LSC {loss_local_structure_centric.detach().item():.3f}"
                )

            global_step += 1

        save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        fabric.barrier()
        if fabric.global_rank == 0:
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, save_path)
            fabric.print(f"Model saved to {save_path}")
        fabric.barrier()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args = resolve_legacy_aliases(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_training(args)


if __name__ == "__main__":
    main()
