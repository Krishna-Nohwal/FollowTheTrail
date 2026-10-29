"""Standalone evaluation of a DINOv3-Large ``traj.py`` checkpoint.

Evaluates FF++ validation, CDFv1, CDFv2, CDF++/CDFv3, DFD, DFDC, DFo, WDF,
and UADFV.  It reports fused frame/video AUC and turning-cosine frame/video
AUC, while skipping unavailable datasets.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from augmentations import load_and_resize, normalize


if not hasattr(transformers, "HybridCache"):
    transformers.HybridCache = getattr(
        transformers,
        "DynamicCache",
        getattr(transformers, "Cache", object),
    )


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FRAME_SUFFIX = re.compile(r"_(?:frame_|f)\d+$", re.IGNORECASE)
FLAT_FRAME = re.compile(r"^(.+?)_(\d+)\.(png|jpg|jpeg|webp|bmp)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_cls_trajectory_fusion_turning_large/best.pth",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--max_videos_per_class",
        type=int,
        default=0,
        help="0 evaluates all videos; otherwise caps each class per dataset.",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=0,
        help="0 uses every available frame; otherwise samples this many frames.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", default="traj_remote_all_results.csv")

    parser.add_argument("--ffpp_manifest", default="data/ffpp/manifest.csv")
    parser.add_argument("--ffpp_root", default="data/ffpp")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--cdfv1_csv", default="data/cdfv1/manifest.csv")
    parser.add_argument("--cdfv1_root", default="data/cdfv1")

    parser.add_argument(
        "--cdfv2_fake_root",
        default="data/cdfv2/fake",
    )
    parser.add_argument(
        "--cdfv2_real_root",
        default="data/cdfv2/real",
    )
    parser.add_argument(
        "--cdfpp_root",
        default="data/cdfpp",
    )
    parser.add_argument(
        "--cdfpp_csv",
        default="data/cdfpp/manifest.csv",
    )
    parser.add_argument(
        "--dfd_fake_root",
        default="data/dfd/fake",
    )
    parser.add_argument(
        "--dfd_real_root",
        default="data/dfd/real",
    )
    parser.add_argument(
        "--dfdc_fake_root",
        default="data/dfdc/fake",
    )
    parser.add_argument(
        "--dfdc_real_root",
        default="data/dfdc/real",
    )
    parser.add_argument(
        "--dfo_fake_root",
        default="data/dfo/fake",
    )
    parser.add_argument(
        "--dfo_real_root",
        default="data/dfo/real",
    )
    parser.add_argument(
        "--wdf_fake_root",
        default="data/wdf/fake",
    )
    parser.add_argument(
        "--wdf_real_root",
        default="data/wdf/real",
    )
    parser.add_argument(
        "--uadfv_fake_root",
        default="data/uadfv/fake",
    )
    parser.add_argument(
        "--uadfv_real_root",
        default="data/uadfv/real",
    )
    parser.add_argument("--no_ffpp_val", action="store_true")
    return parser.parse_args()


def clean_state_dict(state):
    cleaned = dict(state)
    for prefix in ("_orig_mod.", "module."):
        if any(key.startswith(prefix) for key in cleaned):
            cleaned = {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in cleaned.items()
            }
    return cleaned


def load_checkpoint(path: str):
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw:
        return checkpoint_path, clean_state_dict(raw["model"]), raw.get("args", {})
    if isinstance(raw, dict) and "state_dict" in raw:
        return checkpoint_path, clean_state_dict(raw["state_dict"]), raw.get("args", {})
    return checkpoint_path, clean_state_dict(raw), {}


def parse_layers(train_args):
    value = train_args.get("trajectory_layers", "0,8,23")
    if isinstance(value, str):
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        layers = tuple(int(part) for part in value)
    if len(layers) != 3:
        raise ValueError(f"Expected three trajectory layers, found {layers}")
    return layers


def image_files(root: Path):
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_png_files(root: Path):
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("image.png") if path.is_file())


def strip_frame_suffix(name: str) -> str:
    return FRAME_SUFFIX.sub("", name)


def manifest_video_id(row: pd.Series) -> str:
    if "method" in row.index and "video_stem" in row.index:
        return f"{row['method']}/{row['video_stem']}"
    parts = str(row["sample_dir"]).replace("\\", "/").strip("/").split("/")
    return "/".join([*parts[:-1], strip_frame_suffix(parts[-1])])


def build_cdfv1(csv_path: str, root_dir: str):
    csv_file, root = Path(csv_path), Path(root_dir)
    if not csv_file.is_file() or not root.is_dir():
        print("[skip] CDFv1: missing manifest/root")
        return []
    dataframe = pd.read_csv(csv_file, sep=None, engine="python")
    groups = defaultdict(list)
    labels = {}
    for _, row in dataframe.iterrows():
        relative = Path(str(row["sample_dir"]).replace("\\", "/"))
        path = root / relative / "image.png"
        if not path.is_file():
            continue
        video_id = manifest_video_id(row)
        groups[video_id].append(str(path))
        labels[video_id] = int(row["label"])
    return [
        (video_id, sorted(paths), labels[video_id])
        for video_id, paths in sorted(groups.items())
        if paths
    ]


def build_ffpp_validation(manifest_path: str, root_dir: str, val_ratio: float, seed: int):
    manifest, root = Path(manifest_path), Path(root_dir)
    if not manifest.is_file() or not root.is_dir():
        print("[skip] FF++ validation: missing manifest/root")
        return []
    dataframe = pd.read_csv(manifest)
    groups = defaultdict(list)
    labels = {}
    for label in (0, 1):
        pool = dataframe[dataframe["label"].astype(int) == label].sample(
            frac=1.0, random_state=seed
        )
        n_val = int(len(pool) * val_ratio)
        for _, row in pool.iloc[:n_val].iterrows():
            relative = Path(str(row["sample_dir"]).replace("\\", "/"))
            path = root / relative / "image.png"
            if not path.is_file():
                continue
            video_id = f"label{label}/" + strip_frame_suffix(relative.name)
            groups[video_id].append(str(path))
            labels[video_id] = label
    return [
        (video_id, sorted(paths), labels[video_id])
        for video_id, paths in sorted(groups.items())
        if paths
    ]


def build_cdfv2(fake_root: str, real_root: str):
    videos = []
    for root_str, label in ((real_root, 0), (fake_root, 1)):
        root = Path(root_str)
        if not root.is_dir():
            print(f"[skip] CDFv2 root missing: {root}")
            continue
        grouped = defaultdict(list)
        for path in image_png_files(root):
            grouped[strip_frame_suffix(path.parent.name)].append(str(path))
        for video_id, paths in sorted(grouped.items()):
            videos.append((video_id, sorted(paths), label))
    return videos


def build_cdfpp(root_str: str, csv_str: str):
    root = Path(root_str)
    manifest = Path(csv_str)
    if not root.is_dir():
        print(f"[skip] CDF++ root missing: {root}")
        return []
    if not manifest.is_file():
        print(f"[skip] CDF++ manifest missing: {manifest}")
        return []
    dataframe = pd.read_csv(manifest, sep=None, engine="python")
    required = {"sample_dir", "label"}
    if not required.issubset(dataframe.columns):
        print(f"[skip] CDF++ manifest lacks {required}: {list(dataframe.columns)}")
        return []

    grouped = defaultdict(list)
    labels = {}
    for sample_dir, raw_label in zip(dataframe["sample_dir"], dataframe["label"]):
        relative = Path(str(sample_dir).replace("\\", "/"))
        image_path = root / relative / "image.png"
        if not image_path.is_file():
            continue
        video_id = relative.parent.name
        # CDF++/CDFv3 manifest convention: 1=real, 0=fake.
        label = 0 if int(raw_label) == 1 else 1
        grouped[video_id].append(str(image_path))
        labels[video_id] = label
    return [
        (video_id, sorted(paths), labels[video_id])
        for video_id, paths in sorted(grouped.items())
        if paths
    ]


def build_nested(fake_root: str, real_root: str, name: str):
    videos = []
    for root_str, label in ((real_root, 0), (fake_root, 1)):
        root = Path(root_str)
        if not root.is_dir():
            print(f"[skip] {name} root missing: {root}")
            continue
        for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            paths = image_png_files(video_dir)
            if paths:
                videos.append((video_dir.name, [str(path) for path in paths], label))
    return videos


def build_dfdc(fake_root: str, real_root: str):
    videos = []
    for root_str, label in ((real_root, 0), (fake_root, 1)):
        root = Path(root_str)
        if not root.is_dir():
            print(f"[skip] DFDC root missing: {root}")
            continue
        grouped = defaultdict(list)
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            match = FLAT_FRAME.match(path.name)
            if match:
                grouped[match.group(1)].append((int(match.group(2)), str(path)))
        for video_id, indexed_paths in sorted(grouped.items()):
            videos.append(
                (video_id, [path for _, path in sorted(indexed_paths)], label)
            )
    return videos


def build_wdf(fake_root: str, real_root: str):
    videos = []
    for root_str, label in ((real_root, 0), (fake_root, 1)):
        root = Path(root_str)
        if not root.is_dir():
            print(f"[skip] WDF root missing: {root}")
            continue
        grouped = defaultdict(list)
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            match = FLAT_FRAME.match(path.name)
            if match:
                grouped[match.group(1)].append((int(match.group(2)), str(path)))
        for video_id, indexed_paths in sorted(grouped.items()):
            videos.append(
                (video_id, [path for _, path in sorted(indexed_paths)], label)
            )
    return videos


class FrameDataset(Dataset):
    def __init__(self, samples, image_size):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, video_id = self.samples[index]
        image = normalize(load_and_resize(path, self.image_size))
        return image, label, video_id


class FusionTurningDINO(nn.Module):
    def __init__(self, train_args, trajectory_layers):
        super().__init__()
        backbone = timm.create_model(
            train_args.get("model_name", "vit_large_patch16_dinov3.lvd1689m"),
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=0.10,
        )
        if len(backbone.blocks) != 24:
            raise ValueError(f"Expected DINOv3-Large with 24 blocks, found {len(backbone.blocks)}")
        self.embed_dim = int(backbone.num_features)
        self.final_layer = len(backbone.blocks) - 1
        self.trajectory_layers = tuple(trajectory_layers)
        self.vit = get_peft_model(
            backbone,
            LoraConfig(
                r=int(train_args.get("lora_r", 32)),
                lora_alpha=int(train_args.get("lora_alpha", 64)),
                target_modules=["attn.qkv"],
                lora_dropout=float(train_args.get("lora_dropout", 0.10)),
                bias="none",
                task_type="FEATURE_EXTRACTION",
            ),
        )
        trajectory_dim = int(train_args.get("trajectory_dim", 192))
        hidden_dim = int(train_args.get("trajectory_hidden_dim", 384))
        dropout = float(train_args.get("trajectory_dropout", 0.20))
        self.trajectory_projector = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trajectory_dim),
        )
        self.trajectory_fuser = nn.Sequential(
            nn.LayerNorm(5 * trajectory_dim + 3),
            nn.Linear(5 * trajectory_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trajectory_dim),
            nn.LayerNorm(trajectory_dim),
        )
        self.trajectory_classifier = nn.Linear(trajectory_dim, 2)
        self.raw_classifier = nn.Linear(self.embed_dim, 2)
        self.layer_classifier = nn.Linear(trajectory_dim, 2, bias=False)


def sphere_logmap(base, point):
    cosine = (base * point).sum(dim=1, keepdim=True).float().clamp(-1 + 1e-5, 1 - 1e-5)
    angle = torch.acos(cosine)
    direction = point - cosine * base
    tangent = direction * (angle / direction.norm(dim=1, keepdim=True).clamp_min(1e-6))
    return tangent, angle


def sphere_parallel_transport(base, point, tangent):
    denominator = (1.0 + (base * point).sum(dim=1, keepdim=True)).clamp_min(1e-5)
    return tangent - ((tangent * point).sum(dim=1, keepdim=True) / denominator) * (base + point)


def trajectory_features(points):
    distances, end_velocities = [], []
    for first, second in zip(points[:-1], points[1:]):
        velocity, distance = sphere_logmap(first, second)
        distances.append(distance)
        end_velocities.append(sphere_parallel_transport(first, second, velocity))
    turns = []
    for index in range(1, len(end_velocities)):
        second_velocity, _ = sphere_logmap(points[index], points[index + 1])
        turns.append(F.cosine_similarity(end_velocities[index - 1], second_velocity, dim=1).unsqueeze(1))
    velocities_at_final = []
    for segment_index, velocity in enumerate(end_velocities):
        transported = velocity
        for point_index in range(segment_index + 1, len(points) - 1):
            transported = sphere_parallel_transport(points[point_index], points[point_index + 1], transported)
        velocities_at_final.append(transported)
    return torch.cat([*points, *velocities_at_final, *distances, *turns], dim=1)


def turning_cosine(points, min_distance):
    first, middle, final = points
    first_velocity, first_distance = sphere_logmap(first, middle)
    second_velocity, second_distance = sphere_logmap(middle, final)
    first_at_middle = sphere_parallel_transport(first, middle, first_velocity)
    turning = F.cosine_similarity(first_at_middle, second_velocity, dim=1)
    valid = (first_distance[:, 0] > min_distance) & (second_distance[:, 0] > min_distance)
    return turning, valid


@torch.inference_mode()
def model_batch(model, images, device):
    indices = sorted(set(model.trajectory_layers + (model.final_layer,)))
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        _, intermediates = model.vit.forward_intermediates(
            images,
            indices=indices,
            return_prefix_tokens=True,
            norm=True,
        )
        cls_by_layer = {
            layer: prefix_tokens[:, 0, :].float()
            for layer, (_, prefix_tokens) in zip(indices, intermediates)
        }
        points = [
            F.normalize(model.trajectory_projector(cls_by_layer[layer]), p=2, dim=1)
            for layer in model.trajectory_layers
        ]
        traj_logits = model.trajectory_classifier(
            model.trajectory_fuser(trajectory_features(points))
        )
        raw_logits = model.raw_classifier(cls_by_layer[model.final_layer])
        logits = 0.5 * (raw_logits + traj_logits)
        turning, valid = turning_cosine(points, model.min_segment_distance)
    return logits.float(), turning.float(), valid


def select_videos(videos, args, seed_offset):
    if args.max_videos_per_class <= 0:
        return videos
    rng = np.random.default_rng(args.seed + seed_offset)
    selected = []
    for label in (0, 1):
        pool = [video for video in videos if video[2] == label]
        if len(pool) > args.max_videos_per_class:
            pool = [pool[int(index)] for index in rng.choice(len(pool), args.max_videos_per_class, replace=False)]
        selected.extend(pool)
    rng.shuffle(selected)
    return selected


def evaluate_dataset(model, videos, args, device, name, min_distance):
    videos = select_videos(videos, args, sum(map(ord, name)))
    samples = []
    for video_id, paths, label in videos:
        paths = sorted(paths)
        if args.max_frames_per_video > 0 and len(paths) > args.max_frames_per_video:
            indices = np.linspace(0, len(paths) - 1, args.max_frames_per_video, dtype=int)
            paths = [paths[int(index)] for index in indices]
        samples.extend((path, label, video_id) for path in paths)
    if not samples or len({label for _, label, _ in samples}) < 2:
        print(f"[skip] {name}: no usable two-class samples")
        return None

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "shuffle": False,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(FrameDataset(samples, args.image_size), **loader_kwargs)

    frame_labels, frame_scores, frame_turns, frame_turn_labels = [], [], [], []
    video_scores = defaultdict(list)
    video_turns = defaultdict(list)
    video_labels = {}
    for images, labels, video_ids in tqdm(loader, desc=f"Evaluate {name}", leave=False):
        images = images.to(device, non_blocking=True)
        logits, turns, valid = model_batch(model, images, device)
        scores = logits.softmax(dim=1)[:, 1].cpu().numpy()
        labels_np = labels.numpy()
        turns_np = turns.cpu().numpy()
        valid_np = valid.cpu().numpy().astype(bool)
        frame_labels.extend(labels_np.tolist())
        frame_scores.extend(scores.tolist())
        for label, score, turn, is_valid, video_id in zip(labels_np, scores, turns_np, valid_np, video_ids):
            video_id = str(video_id)
            video_labels[video_id] = int(label)
            video_scores[video_id].append(float(score))
            if is_valid:
                video_turns[video_id].append(float(turn))
                frame_turns.append(float(turn))
                frame_turn_labels.append(int(label))

    def auc(labels, scores):
        return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")

    frame_turn_auc = auc(frame_turn_labels, frame_turns)
    video_ids = sorted(video_scores)
    v_labels = [video_labels[video_id] for video_id in video_ids]
    v_scores = [float(np.mean(video_scores[video_id])) for video_id in video_ids]
    v_turn_ids = [video_id for video_id in video_ids if video_turns[video_id]]
    v_turn_labels = [video_labels[video_id] for video_id in v_turn_ids]
    v_turn_scores = [float(np.mean(video_turns[video_id])) for video_id in v_turn_ids]
    row = {
        "dataset": name,
        "videos": len(video_ids),
        "frames": len(frame_labels),
        "real_videos": sum(label == 0 for label in v_labels),
        "fake_videos": sum(label == 1 for label in v_labels),
        "real_frames": sum(label == 0 for label in frame_labels),
        "fake_frames": sum(label == 1 for label in frame_labels),
        "frame_auc": auc(frame_labels, frame_scores),
        "frame_ap": float(average_precision_score(frame_labels, frame_scores)),
        "frame_turning_auc": frame_turn_auc,
        "video_auc": auc(v_labels, v_scores),
        "video_turning_auc": auc(v_turn_labels, v_turn_scores),
        "real_turning_mean": float(np.mean([x for x, y in zip(frame_turns, frame_turn_labels) if y == 0])) if any(y == 0 for y in frame_turn_labels) else float("nan"),
        "fake_turning_mean": float(np.mean([x for x, y in zip(frame_turns, frame_turn_labels) if y == 1])) if any(y == 1 for y in frame_turn_labels) else float("nan"),
    }
    print(
        f"{name:<8} | videos={row['videos']:>6} frames={row['frames']:>8} "
        f"frame_AUC={row['frame_auc']:.6f} video_AUC={row['video_auc']:.6f} "
        f"frame_turn_AUC={row['frame_turning_auc']:.6f} "
        f"video_turn_AUC={row['video_turning_auc']:.6f}"
    )
    return row


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path, state, train_args = load_checkpoint(args.checkpoint)
    trajectory_layers = parse_layers(train_args)
    min_distance = float(train_args.get("turning_min_segment_distance", 1e-3))

    datasets = {}
    if not args.no_ffpp_val:
        datasets["FF++ validation"] = build_ffpp_validation(
            args.ffpp_manifest, args.ffpp_root, args.val_ratio, args.seed
        )
    datasets.update(
        {
            "CDFv1": build_cdfv1(args.cdfv1_csv, args.cdfv1_root),
            "CDFv2": build_cdfv2(args.cdfv2_fake_root, args.cdfv2_real_root),
            "CDF++": build_cdfpp(args.cdfpp_root, args.cdfpp_csv),
            "DFD": build_nested(args.dfd_fake_root, args.dfd_real_root, "DFD"),
            "DFDC": build_dfdc(args.dfdc_fake_root, args.dfdc_real_root),
            "DFo": build_nested(args.dfo_fake_root, args.dfo_real_root, "DFo"),
            "WDF": build_wdf(args.wdf_fake_root, args.wdf_real_root),
            "UADFV": build_nested(args.uadfv_fake_root, args.uadfv_real_root, "UADFV"),
        }
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Trajectory layers: {trajectory_layers}")
    print("Prediction: fused final-CLS and spherical-trajectory logits")
    print("Metrics: frame/video fused AUC and frame/video turning-cosine AUC")
    for name, videos in datasets.items():
        print(
            f"{name:<18} available videos={len(videos):>6} "
            f"real={sum(label == 0 for _, _, label in videos):>6} "
            f"fake={sum(label == 1 for _, _, label in videos):>6}"
        )

    model = FusionTurningDINO(train_args, trajectory_layers).to(device)
    model.min_segment_distance = min_distance
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.eval()

    rows = []
    for name, videos in datasets.items():
        if not videos:
            print(f"[skip] {name}: no videos found")
            continue
        row = evaluate_dataset(model, videos, args, device, name, min_distance)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("No requested datasets were available.")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
