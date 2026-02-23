import os
import argparse
import torch
import torch.nn as nn
from types import SimpleNamespace
from models import model_dict
from dataset.meta_cifar100 import get_cifar100_dataloaders
from helper.meta_loops import validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='resnet32x4', help='teacher model name')
    parser.add_argument('--save-folder', default='./save/student_model/S:resnet8_T:resnet32x4_cifar100_mlkd_a:None_1',
                        help='folder where teacher checkpoint is saved')
    parser.add_argument('--checkpoint', default=None, help='full path to a specific checkpoint file (.pth)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--print-freq', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    teacher_model_name = args.teacher
    save_folder = args.save_folder
    # allow user to pass a full checkpoint path; otherwise build default path
    teacher_ckpt = args.checkpoint if args.checkpoint else os.path.join(save_folder, f'{teacher_model_name}_teacher_last.pth')

    # helpful fallback: if default path missing, try to find .pth files in save_folder
    if not os.path.isfile(teacher_ckpt):
        if args.checkpoint:
            raise FileNotFoundError(f'Checkpoint not found: {teacher_ckpt}')
        # If user provided a save_folder but it doesn't exist, try searching likely locations.
        # 1) If save_folder is a directory, look for .pth inside it.
        if os.path.isdir(save_folder):
            candidates = [os.path.join(save_folder, f) for f in os.listdir(save_folder) if f.endswith('.pth')]
        else:
            candidates = []

        # 2) If none found and the default save root exists, search subfolders for the teacher name
        if not candidates:
            default_root = os.path.join('.', 'save', 'student_model')
            if os.path.isdir(default_root):
                for sub in os.listdir(default_root):
                    subpath = os.path.join(default_root, sub)
                    if os.path.isdir(subpath) and (f'T:{teacher_model_name}' in sub or teacher_model_name in sub):
                        for f in os.listdir(subpath):
                            if f.endswith('.pth'):
                                candidates.append(os.path.join(subpath, f))

        # 3) final fallback: walk the repo and find files that match the teacher filename
        if not candidates:
            for root, _, files in os.walk('.'):
                for f in files:
                    if f == f'{teacher_model_name}_teacher_last.pth' or f.endswith(f'_{teacher_model_name}_teacher_last.pth'):
                        candidates.append(os.path.join(root, f))
            if len(candidates) == 1:
                teacher_ckpt = candidates[0]
                print(f'Using found checkpoint: {teacher_ckpt}')
            elif len(candidates) > 1:
                raise FileNotFoundError(
                    f'Checkpoint not found at {os.path.join(save_folder, f"{teacher_model_name}_teacher_last.pth")}.'
                    f' Multiple .pth files exist in {save_folder}:\n' + '\n'.join(candidates)
                    + '\nPass --checkpoint to specify which to use.'
                )
            else:
                raise FileNotFoundError(f'Checkpoint not found: {teacher_ckpt}')
        else:
            raise FileNotFoundError(f'Checkpoint not found: {teacher_ckpt}')

    map_loc = None if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(teacher_ckpt, map_location=map_loc)

    model_t = model_dict[teacher_model_name](num_classes=100)
    model_t.load_state_dict(checkpoint['model'])
    model_t.eval()
    if torch.cuda.is_available():
        model_t.cuda()

    _, _, val_loader = get_cifar100_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()

    opt = SimpleNamespace(print_freq=args.print_freq)
    acc, acc_top5, loss = validate(val_loader, model_t, criterion, opt)
    print(f'Teacher model accuracy: {acc:.2f}%')
    print(f'Teacher model top-5 accuracy: {acc_top5:.2f}%')
    print(f'Teacher model loss: {loss:.4f}')


if __name__ == '__main__':
    main()
