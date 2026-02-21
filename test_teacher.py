import os
import torch
import torch.nn as nn
from models import model_dict
from dataset.meta_cifar100 import get_cifar100_dataloaders
from helper.meta_loops import validate

def main():
    # Path to the saved teacher model
    teacher_model_name = 'resnet32x4'  # Change if needed
    save_folder = './save/student_model/S:resnet8_T:resnet32x4_cifar100_mlkd_a:None_1'  # Update if needed
    teacher_ckpt = os.path.join(save_folder, f'{teacher_model_name}_teacher_last.pth')

    # Load checkpoint
    checkpoint = torch.load(teacher_ckpt)
    model_t = model_dict[teacher_model_name](num_classes=100)
    model_t.load_state_dict(checkpoint['model'])
    model_t.eval()
    if torch.cuda.is_available():
        model_t.cuda()

    # Load CIFAR-100 test/validation data
    _, _, val_loader = get_cifar100_dataloaders(batch_size=64, num_workers=8, held_size=None, num_held_samples=None)
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    acc, acc_top5, loss = validate(val_loader, model_t, criterion, None)
    print(f'Teacher model accuracy: {acc:.2f}%')
    print(f'Teacher model top-5 accuracy: {acc_top5:.2f}%')
    print(f'Teacher model loss: {loss:.4f}')

if __name__ == '__main__':
    main()
