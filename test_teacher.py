import os
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

from models import model_dict
from dataset.meta_cifar100 import get_cifar100_dataloaders
from helper.meta_loops import validate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an (updated) teacher checkpoint on CIFAR-100 test data.")

    parser.add_argument("--teacher", type=str, default="resnet32x4",
                        help="Teacher model name (must exist in models/model_dict).")

    parser.add_argument("--save_folder", type=str,
                        default="./save/student_model/S:resnet8_T:resnet32x4_cifar100_mlkd_a:None_1",
                        help="Folder where the teacher checkpoint is saved.")

    parser.add_argument("--checkpoint", type=str, default="",
                        help="Full path to a checkpoint (.pth). If empty, uses: save_folder/<teacher>_teacher_last.pth")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--print_freq", type=int, default=100)

    return parser.parse_args()


def resolve_checkpoint_path(args) -> str:
    ckpt = (args.checkpoint or "").strip()
    if ckpt:
        return ckpt
    return os.path.join(args.save_folder, f"{args.teacher}_teacher_last.pth")


def load_teacher_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # Your train_student_meta.py saves: {'opt': opt, 'model': state_dict}
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Handle DataParallel checkpoints (module.xxx keys)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


def main():
    args = parse_args()

    if args.teacher not in model_dict:
        raise ValueError(
            f"Unknown teacher model '{args.teacher}'. Available: {sorted(model_dict.keys())}"
        )

    ckpt_path = resolve_checkpoint_path(args)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt_path}\n"
            f"Expected default name: {args.teacher}_teacher_last.pth inside --save_folder\n"
            f"Or pass --checkpoint with a full path."
        )

    # Build teacher
    model_t = model_dict[args.teacher](num_classes=100)

    # Load adapted teacher weights
    load_teacher_checkpoint(model_t, ckpt_path)

    # Device
    if torch.cuda.is_available():
        model_t = model_t.cuda()

    model_t.eval()

    # CIFAR-100 loaders (your helper returns train, held, test/val)
    _, _, test_loader = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    criterion = nn.CrossEntropyLoss()
    opt = SimpleNamespace(print_freq=args.print_freq)

    acc1, acc5, loss = validate(test_loader, model_t, criterion, opt)

    print(f"Loaded teacher checkpoint: {ckpt_path}")
    print(f"Teacher Top-1 Acc:  {acc1:.2f}%")
    print(f"Teacher Top-5 Acc:  {acc5:.2f}%")
    print(f"Teacher Loss:       {loss:.4f}")


if __name__ == "__main__":
    main()