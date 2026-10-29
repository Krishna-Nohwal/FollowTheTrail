"""Train DINOv3-Large with fused CLS and spherical trajectory predictions.

This is the DINOv3-Large counterpart of
``train_cls_trajectory_fusion_turning.py``.  The training loss, LoRA setup,
augmentation, FF++ split, and checkpoint format are kept the same.

The Large backbone has 24 transformer blocks (zero-based indices 0..23).  The
default trajectory knots are 0, 8, and 23.  CDFv1 is evaluated at frame level
and, additionally, at video level after every epoch.  Video scores are the
mean frame score within each ``method/video_stem`` group:

* trajectory-fusion video score: mean fused fake probability;
* turning-only video score: mean valid turning cosine.

The turning cosine is reported as a score with fake=1, so its raw AUC is used
directly.  No video grouping is used for training or FF++ validation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from augmentations import augment_batch, load_and_resize, normalize


if not hasattr(transformers, "HybridCache"):
    transformers.HybridCache = getattr(
        transformers,
        "DynamicCache",
        getattr(transformers, "Cache", object),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_root",
        default="checkpoints/followthetrail",
    )
    parser.add_argument("--manifest", default="data/ffpp/manifest.csv")
    parser.add_argument("--root_dir", default="data/ffpp")
    parser.add_argument("--cdfv1_root", default="data/cdfv1")
    parser.add_argument(
        "--cdfv1_csv",
        default="data/cdfv1/manifest.csv",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.10)
    parser.add_argument("--trajectory_layers", default="0,8,23")
    parser.add_argument("--trajectory_dim", type=int, default=192)
    parser.add_argument("--trajectory_hidden_dim", type=int, default=384)
    parser.add_argument("--trajectory_dropout", type=float, default=0.20)
    parser.add_argument("--raw_ce_weight", type=float, default=0.35)
    parser.add_argument("--trajectory_ce_weight", type=float, default=1.0)
    parser.add_argument("--layer_ce_weight", type=float, default=0.08)
    parser.add_argument("--turning_loss_weight", type=float, default=0.1)
    parser.add_argument("--turning_margin", type=float, default=0.2)
    parser.add_argument("--turning_min_segment_distance", type=float, default=1e-3)
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Compatibility flag; this script does not use torch.compile.",
    )
    parser.add_argument("--natural_sampling", action="store_true")
    return parser.parse_args()


args = parse_args()
TRAJECTORY_LAYERS = tuple(
    int(value.strip()) for value in args.trajectory_layers.split(",") if value.strip()
)
if len(TRAJECTORY_LAYERS) != 3 or list(TRAJECTORY_LAYERS) != sorted(set(TRAJECTORY_LAYERS)):
    raise ValueError("trajectory_layers must contain exactly three increasing indices")
if TRAJECTORY_LAYERS[0] < 0 or TRAJECTORY_LAYERS[-1] > 23:
    raise ValueError("DINOv3-Large trajectory layers must be in [0, 23]")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
IMG_SIZE = args.image_size
torch.backends.cudnn.benchmark = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sphere_logmap(base: torch.Tensor, point: torch.Tensor):
    cosine = (base * point).sum(dim=1, keepdim=True).float()
    cosine = cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
    angle = torch.acos(cosine)
    direction = point - cosine * base
    tangent = direction * (
        angle / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
    )
    return tangent, angle


def sphere_parallel_transport(
    base: torch.Tensor,
    point: torch.Tensor,
    tangent: torch.Tensor,
):
    denominator = (1.0 + (base * point).sum(dim=1, keepdim=True)).clamp_min(1e-5)
    return tangent - (
        (tangent * point).sum(dim=1, keepdim=True) / denominator
    ) * (base + point)


def spherical_trajectory_features(points: list[torch.Tensor]):
    distances = []
    segment_end_velocities = []
    for first, second in zip(points[:-1], points[1:]):
        velocity, distance = sphere_logmap(first, second)
        distances.append(distance)
        segment_end_velocities.append(
            sphere_parallel_transport(first, second, velocity)
        )

    turns = []
    for index in range(1, len(segment_end_velocities)):
        second_velocity, _ = sphere_logmap(points[index], points[index + 1])
        turns.append(
            F.cosine_similarity(
                segment_end_velocities[index - 1], second_velocity, dim=1
            ).unsqueeze(1)
        )

    velocities_at_final = []
    for segment_index, velocity in enumerate(segment_end_velocities):
        transported = velocity
        for point_index in range(segment_index + 1, len(points) - 1):
            transported = sphere_parallel_transport(
                points[point_index], points[point_index + 1], transported
            )
        velocities_at_final.append(transported)

    return torch.cat(
        [*points, *velocities_at_final, *distances, *turns],
        dim=1,
    )


def triplet_turning_cosine(points: list[torch.Tensor]):
    first, middle, final = points
    first_velocity, first_distance = sphere_logmap(first, middle)
    second_velocity, second_distance = sphere_logmap(middle, final)
    first_at_middle = sphere_parallel_transport(first, middle, first_velocity)
    turning = F.cosine_similarity(first_at_middle, second_velocity, dim=1)
    valid = (
        (first_distance[:, 0] > args.turning_min_segment_distance)
        & (second_distance[:, 0] > args.turning_min_segment_distance)
    )
    return turning, valid


def frame_path(root: Path, value: str) -> str:
    value = str(value).replace("\\", "/")
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate / "image.png")
    return str(root / value / "image.png")


class ManifestDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, root_dir: str):
        root = Path(root_dir)
        paths = [frame_path(root, value) for value in dataframe["sample_dir"]]
        labels = dataframe["label"].astype(int).to_numpy()
        exists = np.asarray([os.path.exists(path) for path in paths], dtype=bool)
        if (~exists).any():
            print(f"Skipped {(~exists).sum()} missing FF++ frames")
        self.entries = [
            (path, int(label))
            for path, label in zip(np.asarray(paths)[exists], labels[exists])
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, label = self.entries[index]
        return normalize(load_and_resize(path, IMG_SIZE)), label


def cdf_video_id(row: pd.Series) -> str:
    if "method" in row.index and "video_stem" in row.index:
        return f"{row['method']}/{row['video_stem']}"
    parts = str(row["sample_dir"]).replace("\\", "/").strip("/").split("/")
    frame_dir = parts[-1]
    video_stem = re.sub(r"_f\d+$", "", frame_dir)
    if video_stem != frame_dir:
        return "/".join([*parts[:-1], video_stem])
    return "/".join(parts[:-1])


class CDFv1VideoDataset(Dataset):
    def __init__(self):
        dataframe = pd.read_csv(args.cdfv1_csv, sep=None, engine="python")
        if not {"sample_dir", "label"}.issubset(dataframe.columns):
            raise ValueError("CDFv1 manifest must contain sample_dir and label columns")
        root = Path(args.cdfv1_root)
        paths = [frame_path(root, value) for value in dataframe["sample_dir"]]
        # CDFv1 manifests use label=1 for real and label=0 for fake.
        labels = 1 - dataframe["label"].astype(int).to_numpy()
        video_ids = [cdf_video_id(row) for _, row in dataframe.iterrows()]
        exists = np.asarray([os.path.exists(path) for path in paths], dtype=bool)
        if (~exists).any():
            print(f"Skipped {(~exists).sum()} missing CDFv1 frames")
        self.entries = [
            (path, int(label), video_id)
            for path, label, video_id in zip(
                np.asarray(paths)[exists], labels[exists], np.asarray(video_ids)[exists]
            )
        ]
        self.video_labels = {}
        for _, label, video_id in self.entries:
            previous = self.video_labels.setdefault(video_id, label)
            if previous != label:
                raise ValueError(f"Mixed labels found within CDFv1 video group {video_id}")
        real = sum(label == 0 for _, label, _ in self.entries)
        fake = sum(label == 1 for _, label, _ in self.entries)
        print(
            f"CDFv1 frame test: real={real} fake={fake} total={len(self.entries)} | "
            f"videos={len(self.video_labels)}"
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, label, video_id = self.entries[index]
        return normalize(load_and_resize(path, IMG_SIZE)), label, video_id


def split_ffpp():
    dataframe = pd.read_csv(args.manifest)
    if not {"sample_dir", "label"}.issubset(dataframe.columns):
        raise ValueError("FF++ manifest must contain sample_dir and label columns")
    labels = dataframe["label"].astype(int)
    pools = {
        label: dataframe[labels == label].sample(frac=1.0, random_state=args.seed)
        for label in (0, 1)
    }
    train_parts, val_parts = [], []
    for label in (0, 1):
        n_val = int(len(pools[label]) * args.val_ratio)
        val_parts.append(pools[label].iloc[:n_val])
        train_parts.append(pools[label].iloc[n_val:])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=args.seed)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=args.seed)
    print(
        f"Train: {train_df['label'].value_counts().sort_index().to_dict()} | "
        f"total={len(train_df)}"
    )
    print(
        f"Val:   {val_df['label'].value_counts().sort_index().to_dict()} | "
        f"total={len(val_df)}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class CLSAndTrajectoryDINO(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=0.10,
        )
        if len(backbone.blocks) != 24:
            raise ValueError(
                f"Expected DINOv3-Large to have 24 blocks, found {len(backbone.blocks)}"
            )
        self.embed_dim = int(backbone.num_features)
        self.final_layer = len(backbone.blocks) - 1
        self.vit = get_peft_model(
            backbone,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["attn.qkv"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            ),
        )
        self.vit.base_model.model.set_grad_checkpointing(enable=True)
        for module in self.vit.base_model.model.modules():
            if isinstance(module, nn.LayerNorm):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(True)

        self.trajectory_projector = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, args.trajectory_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.trajectory_dropout),
            nn.Linear(args.trajectory_hidden_dim, args.trajectory_dim),
        )
        trajectory_input_dim = 5 * args.trajectory_dim + 3
        self.trajectory_fuser = nn.Sequential(
            nn.LayerNorm(trajectory_input_dim),
            nn.Linear(trajectory_input_dim, args.trajectory_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.trajectory_dropout),
            nn.Linear(args.trajectory_hidden_dim, args.trajectory_dim),
            nn.LayerNorm(args.trajectory_dim),
        )
        self.trajectory_classifier = nn.Linear(args.trajectory_dim, 2)
        self.raw_classifier = nn.Linear(self.embed_dim, 2)
        self.layer_classifier = nn.Linear(args.trajectory_dim, 2, bias=False)

    def forward(self, images: torch.Tensor):
        indices = sorted(set(TRAJECTORY_LAYERS + (self.final_layer,)))
        _, intermediates = self.vit.forward_intermediates(
            images,
            indices=indices,
            return_prefix_tokens=True,
            norm=True,
        )
        cls_by_layer = {
            layer: prefix_tokens[:, 0, :].float()
            for layer, (_, prefix_tokens) in zip(indices, intermediates)
        }
        final_cls = cls_by_layer[self.final_layer]
        points = [
            F.normalize(
                self.trajectory_projector(cls_by_layer[layer]),
                p=2,
                dim=1,
            )
            for layer in TRAJECTORY_LAYERS
        ]
        trajectory_input = spherical_trajectory_features(points)
        trajectory_embedding = self.trajectory_fuser(trajectory_input)
        trajectory_logits = self.trajectory_classifier(trajectory_embedding)
        raw_logits = self.raw_classifier(final_cls)
        layer_logits = torch.stack(
            [self.layer_classifier(point) for point in points],
            dim=1,
        )
        turning, turning_valid = triplet_turning_cosine(points)
        prediction_logits = 0.5 * (raw_logits + trajectory_logits)
        return prediction_logits, {
            "raw_logits": raw_logits,
            "trajectory_logits": trajectory_logits,
            "layer_logits": layer_logits,
            "trajectory_embedding": trajectory_embedding,
            "turning": turning,
            "turning_valid": turning_valid,
        }


def compute_loss(logits, features, labels):
    fused_ce = F.cross_entropy(logits, labels)
    raw_ce = F.cross_entropy(features["raw_logits"], labels)
    trajectory_ce = F.cross_entropy(features["trajectory_logits"], labels)
    repeated_labels = labels[:, None].expand(-1, features["layer_logits"].size(1))
    layer_ce = F.cross_entropy(
        features["layer_logits"].flatten(0, 1),
        repeated_labels.reshape(-1),
    )
    target_sign = labels.float().mul(2.0).sub(1.0)
    turning = features["turning"].float()
    signed_margin = target_sign * turning
    penalties = F.relu(args.turning_margin - signed_margin)
    valid = features["turning_valid"]
    turning_loss = penalties[valid].mean() if valid.any() else penalties.mean() * 0.0
    total = (
        fused_ce
        + args.raw_ce_weight * raw_ce
        + args.trajectory_ce_weight * trajectory_ce
        + args.layer_ce_weight * layer_ce
        + args.turning_loss_weight * turning_loss
    )
    return total, {
        "total": total.detach(),
        "fused_ce": fused_ce.detach(),
        "raw_ce": raw_ce.detach(),
        "trajectory_ce": trajectory_ce.detach(),
        "layer_ce": layer_ce.detach(),
        "turning": turning_loss.detach(),
    }


def report_metrics(labels, probabilities, name, epoch):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    result = {
        "auc": float(roc_auc_score(labels, probabilities)),
        "ap": float(average_precision_score(labels, probabilities)),
        "acc": float(accuracy_score(labels, probabilities >= 0.5)),
    }
    print(
        f"{name} epoch={epoch + 1}: AUC={result['auc']:.6f} "
        f"AP={result['ap']:.6f} ACC={result['acc']:.6f}"
    )
    return result


def video_auc(labels, scores, video_ids):
    grouped = defaultdict(lambda: {"labels": [], "scores": []})
    for label, score, video_id in zip(labels, scores, video_ids):
        if score is None or not np.isfinite(score):
            continue
        grouped[str(video_id)]["labels"].append(int(label))
        grouped[str(video_id)]["scores"].append(float(score))
    group_labels, group_scores = [], []
    for values in grouped.values():
        unique_labels = set(values["labels"])
        if len(unique_labels) != 1:
            continue
        group_labels.append(values["labels"][0])
        group_scores.append(float(np.mean(values["scores"])))
    if len(set(group_labels)) < 2:
        return float("nan"), 0, group_labels, group_scores
    return (
        float(roc_auc_score(group_labels, group_scores)),
        len(group_labels),
        group_labels,
        group_scores,
    )


def report_turning(turns, turn_labels, name, epoch):
    if not turns or len(set(turn_labels)) < 2:
        return float("nan")
    turns_np = np.asarray(turns, dtype=np.float64)
    labels_np = np.asarray(turn_labels, dtype=np.int64)
    turn_auc = float(roc_auc_score(labels_np, turns_np))
    print(
        f"{name} turning: real_mean={turns_np[labels_np == 0].mean():+.6f} "
        f"fake_mean={turns_np[labels_np == 1].mean():+.6f} "
        f"AUC={turn_auc:.6f} oriented={max(turn_auc, 1.0 - turn_auc):.6f}"
    )
    return turn_auc


@torch.no_grad()
def evaluate(model, loader, name, epoch, video_level=False):
    model.eval()
    labels, probabilities = [], []
    turns, turn_labels = [], []
    video_ids, video_turns, video_turn_labels = [], [], []
    for batch in tqdm(loader, desc=name, leave=False):
        if len(batch) == 3:
            images, batch_labels, batch_video_ids = batch
        else:
            images, batch_labels = batch
            batch_video_ids = None
        images = images.to(DEVICE, non_blocking=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            logits, features = model(images)
        batch_probabilities = logits.float().softmax(dim=1)[:, 1].cpu().numpy()
        batch_labels_np = batch_labels.numpy()
        labels.extend(batch_labels_np.tolist())
        probabilities.extend(batch_probabilities.tolist())

        batch_turning = features["turning"].float().view(-1).cpu().numpy()
        batch_valid = features["turning_valid"].cpu().numpy().astype(bool)
        turns.extend(batch_turning[batch_valid].tolist())
        turn_labels.extend(batch_labels_np[batch_valid].tolist())
        if batch_video_ids is not None:
            video_ids.extend(list(batch_video_ids))
            video_turns.extend(
                [float(value) if valid else None for value, valid in zip(batch_turning, batch_valid)]
            )
            video_turn_labels.extend(batch_labels_np.tolist())

    frame_metrics = report_metrics(labels, probabilities, name, epoch)
    frame_turn_auc = report_turning(turns, turn_labels, name, epoch)
    result = {
        **{f"frame_{key}": value for key, value in frame_metrics.items()},
        "frame_turn_auc": frame_turn_auc,
    }
    if video_level and video_ids:
        fusion_auc, fusion_count, group_labels, group_scores = video_auc(
            labels, probabilities, video_ids
        )
        turn_auc, turn_count, _, _ = video_auc(
            video_turn_labels, video_turns, video_ids
        )
        result["video_fusion_auc"] = fusion_auc
        result["video_turn_auc"] = turn_auc
        result["video_count"] = fusion_count
        print(
            f"{name} VIDEO epoch={epoch + 1}: "
            f"trajectory_fusion_AUC={fusion_auc:.6f} "
            f"turning_cosine_AUC={turn_auc:.6f} "
            f"videos={fusion_count}"
        )
        if len(group_labels) > 0 and len(set(group_labels)) == 2:
            print(
                f"{name} VIDEO fusion score means: "
                f"real={np.mean([score for score, label in zip(group_scores, group_labels) if label == 0]):.6f} "
                f"fake={np.mean([score for score, label in zip(group_scores, group_labels) if label == 1]):.6f}"
            )
    return result


def append_history(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(model, optimizer, scheduler, epoch, path, row):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": row,
            "args": vars(args),
        },
        path,
    )


def main():
    seed_everything(args.seed)
    print(f"Device: {DEVICE} | AMP: {AMP_ENABLED}")
    print("Backbone: DINOv3-Large (24 transformer blocks)")
    print(f"Trajectory layers: {TRAJECTORY_LAYERS}")
    print("Prediction: average of final-CLS logits and trajectory logits")
    print(f"Turning loss weight: {args.turning_loss_weight}; metric loss: disabled")
    print("Training/evaluation unit: frames; CDFv1 also reports video-level AUC")

    train_df, val_df = split_ffpp()
    train_dataset = ManifestDataset(train_df, args.root_dir)
    val_dataset = ManifestDataset(val_df, args.root_dir)
    cdf_dataset = CDFv1VideoDataset()

    if args.natural_sampling:
        sampler = None
        shuffle = True
        print("Training sampler: natural FF++ frame distribution")
    else:
        labels = np.asarray([label for _, label in train_dataset.entries], dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        weights = 1.0 / np.maximum(counts[labels], 1)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle = False
        print(f"Training sampler: class-balanced draws from counts {counts.tolist()}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=shuffle, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    cdf_loader = DataLoader(cdf_dataset, shuffle=False, **loader_kwargs)

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    model = CLSAndTrajectoryDINO().to(DEVICE)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(args.epochs * len(train_loader), 1)

    def lr_factor(step):
        minimum = 1e-6 / args.lr
        if step < args.warmup_steps:
            return minimum + (1.0 - minimum) * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    best_auc = -1.0
    history_path = str(save_root / "history.csv")

    for epoch in range(args.epochs):
        model.train()
        names = ("total", "fused_ce", "raw_ce", "trajectory_ce", "layer_ce", "turning")
        sums = {name: 0.0 for name in names}
        print(f"\n{'=' * 80}\nEPOCH {epoch + 1}/{args.epochs}\n{'=' * 80}")
        for batch_index, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False)
        ):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            images = augment_batch(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=AMP_ENABLED,
            ):
                logits, features = model(images)
                loss, parts = compute_loss(logits, features, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for name in names:
                sums[name] += float(parts[name])
            if batch_index % 128 == 0:
                print(
                    f"  batch={batch_index:4d}/{len(train_loader)} "
                    f"loss={loss.item():.4f} fused={parts['fused_ce'].item():.4f} "
                    f"turning={parts['turning'].item():.4f}"
                )

        divisor = max(len(train_loader), 1)
        means = {name: value / divisor for name, value in sums.items()}
        print(
            f"Mean losses: total={means['total']:.4f} fused={means['fused_ce']:.4f} "
            f"raw={means['raw_ce']:.4f} trajectory={means['trajectory_ce']:.4f} "
            f"layer={means['layer_ce']:.4f} turning={means['turning']:.4f}"
        )
        val_metrics = evaluate(model, val_loader, "FF++ validation", epoch)
        cdf_metrics = evaluate(model, cdf_loader, "CDFv1", epoch, video_level=True)
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value for name, value in means.items()},
            "val_frame_auc": val_metrics["frame_auc"],
            "cdf_frame_auc": cdf_metrics["frame_auc"],
            "cdf_frame_ap": cdf_metrics["frame_ap"],
            "cdf_frame_acc": cdf_metrics["frame_acc"],
            "cdf_frame_turn_auc": cdf_metrics["frame_turn_auc"],
            "cdf_video_fusion_auc": cdf_metrics["video_fusion_auc"],
            "cdf_video_turn_auc": cdf_metrics["video_turn_auc"],
            "cdf_video_count": cdf_metrics["video_count"],
        }
        append_history(history_path, row)
        save_checkpoint(model, optimizer, scheduler, epoch, str(save_root / "latest.pth"), row)
        model.vit.save_pretrained(str(save_root / "latest_lora"))
        if cdf_metrics["frame_auc"] > best_auc:
            best_auc = cdf_metrics["frame_auc"]
            save_checkpoint(model, optimizer, scheduler, epoch, str(save_root / "best.pth"), row)
            model.vit.save_pretrained(str(save_root / "best_lora"))
            print(f"New best frame-level CDFv1 AUC={best_auc:.6f}")

    print(
        f"Training complete. Best CDFv1 frame-level AUC={best_auc:.6f}; "
        f"checkpoint={save_root / 'best.pth'}"
    )


if __name__ == "__main__":
    main()
