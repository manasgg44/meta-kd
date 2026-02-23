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
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--print-freq', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    teacher_model_name = args.teacher
    save_folder = args.save_folder
    teacher_ckpt = os.path.join(save_folder, f'{teacher_model_name}_teacher_last.pth')

    if not os.path.isfile(teacher_ckpt):
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
