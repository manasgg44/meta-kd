import os
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

from models import model_dict
from dataset.meta_cifar100 import get_cifar100_dataloaders
from helper.meta_loops import validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an updated/adapted teacher checkpoint on CIFAR-100 test set."
    )

    # Keep naming consistent with your training script style
    parser.add_argument("--teacher", type=str, required=True,
                        help="Teacher architecture name (must exist in models/model_dict). "
                             "Example: resnet110, resnet32x4")

    parser.add_argument("--save_folder", type=str, required=True,
                        help="Directory where checkpoints were saved (opt.save_folder from training).")

    parser.add_argument("--checkpoint", type=str, default="",
                        help="Optional: full path to a teacher checkpoint .pth. "
                             "If empty, defaults to: <save_folder>/<teacher>_teacher_last.pth")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--print_freq", type=int, default=10)

    parser.add_argument("--strict", action="store_true",
                        help="Enable strict state_dict loading (recommended).")

    parser.add_argument("--debug", action="store_true",
                        help="Print extra debug info (checkpoint keys, stats).")

    return parser


def normalize_save_folder_path(save_folder: str) -> str:
    """If user accidentally passes a .pth path as --save_folder, convert it to its directory."""
    save_folder = (save_folder or "").strip()
    if save_folder.endswith(".pth"):
        return os.path.dirname(save_folder)
    return save_folder


def resolve_teacher_ckpt_path(teacher: str, save_folder: str, checkpoint: str) -> str:
    checkpoint = (checkpoint or "").strip()
    save_folder = normalize_save_folder_path(save_folder)

    if checkpoint:
        return checkpoint

    # default expected name from train_student_meta.py
    return os.path.join(save_folder, f"{teacher}_teacher_last.pth")


def load_state_dict_from_ckpt(ckpt_path: str, debug: bool = False) -> dict:
    """
    train_student_meta.py saves:
      torch.save({'opt': opt, 'model': model_t.state_dict()}, teacher_save_file)
    So we must load ckpt['model'].
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if debug:
        print(f"[DEBUG] Loaded checkpoint object type: {type(ckpt)}")
        if isinstance(ckpt, dict):
            print(f"[DEBUG] Checkpoint keys: {list(ckpt.keys())}")

    # Expected format
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        # Fallback: sometimes people save raw state_dict under different keys
        # Try common alternatives, else treat ckpt itself as state_dict if it looks like one
        for k in ["state_dict", "net", "teacher", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        else:
            # If dict looks like a state_dict already
            state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format. Expected dict-like checkpoint.")

    # Handle DataParallel ("module.") prefix
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): val for key, val in state_dict.items()}

    if debug:
        first_key = next(iter(state_dict.keys()))
        w = state_dict[first_key]
        w = w.float()
        print(f"[DEBUG] State_dict entries: {len(state_dict)}")
        print(f"[DEBUG] First key: {first_key} | shape: {tuple(w.shape)} | mean: {w.mean().item():.6f} | std: {w.std().item():.6f}")

    return state_dict


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.teacher not in model_dict:
        raise ValueError(
            f"Unknown teacher model '{args.teacher}'. Available: {sorted(model_dict.keys())}"
        )

    args.save_folder = normalize_save_folder_path(args.save_folder)
    ckpt_path = resolve_teacher_ckpt_path(args.teacher, args.save_folder, args.checkpoint)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt_path}\n"
            f"Expected default name: {args.teacher}_teacher_last.pth inside --save_folder\n"
            f"Or pass --checkpoint with a full path."
        )

    # Build teacher (CIFAR-100 => 100 classes)
    model_t = model_dict[args.teacher](num_classes=100)

    # Load weights (always load ckpt to CPU for safety)
    state_dict = load_state_dict_from_ckpt(ckpt_path, debug=args.debug)

    strict = True if args.strict else False
    missing, unexpected = model_t.load_state_dict(state_dict, strict=strict)

    # If not strict, still report issues
    if (not strict) and (len(missing) > 0 or len(unexpected) > 0):
        print(f"[WARN] Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if args.debug:
            print("[DEBUG] Missing keys (first 20):", missing[:20])
            print("[DEBUG] Unexpected keys (first 20):", unexpected[:20])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_t = model_t.to(device)
    model_t.eval()

    # Load CIFAR-100 test loader (same pipeline used in training)
    _, _, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    criterion = nn.CrossEntropyLoss()

    # validate() expects an opt with print_freq
    opt = SimpleNamespace(print_freq=args.print_freq)

    acc1, acc5, loss = validate(test_loader, model_t, criterion, opt)

    print(f"Loaded teacher checkpoint: {ckpt_path}")
    print(f"Teacher Top-1 Acc:  {acc1:.2f}%")
    print(f"Teacher Top-5 Acc:  {acc5:.2f}%")
    print(f"Teacher Loss:       {loss:.4f}")

    # Extra: flag the “random classifier” signature clearly
    # ln(100) ~= 4.6052 and acc1 ~ 1% indicates uniform predictions.
    if loss >= 4.55 and acc1 <= 2.0:
        print("[WARN] Accuracy/loss look like a near-uniform random classifier for CIFAR-100 "
              "(loss ~ ln(100), acc1 ~ 1%). This often indicates a bad/mismatched checkpoint "
              "or a collapsed teacher during training.")


if __name__ == "__main__":
    main()