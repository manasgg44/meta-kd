from __future__ import print_function

import os
import argparse
import socket
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import model_dict
from dataset.meta_cifar100 import get_cifar100_dataloaders
from helper.util import adjust_learning_rate
from helper.meta_loops import train_distill as train, validate
from distiller_zoo.KD import CustomDistillKL
from distiller_zoo.MSE import MSEWithTemperature


def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--tb_freq", type=int, default=500)
    parser.add_argument("--save_freq", type=int, default=40)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--init_epochs", type=int, default=30)

    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--teacher_lr", type=float, default=0.05)
    parser.add_argument("--lr_decay_epochs", type=str, default="150,180,210")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--loss_type", type=str, choices=["mse", "kl"], required=True)

    parser.add_argument("--held_size", type=int, required=True)
    parser.add_argument("--num_held_samples", type=int, required=True)
    parser.add_argument("--num_meta_batches", type=int, default=1)
    parser.add_argument("--assume_s_step_size", type=float, default=0.05)

    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100"])

    parser.add_argument(
        "--model_s",
        type=str,
        default="resnet8",
        choices=[
            "resnet8", "resnet14", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110",
            "resnet8x4", "resnet32x4",
            "vgg8", "vgg11", "vgg13", "vgg16", "vgg19",
        ],
    )
    parser.add_argument(
        "--model_t",
        type=str,
        required=True,
        choices=[
            "resnet8", "resnet14", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110",
            "resnet8x4", "resnet32x4",
            "vgg8", "vgg11", "vgg13", "vgg16", "vgg19",
            "wrn_16_1", "wrn_16_2", "wrn_40_1", "wrn_40_2",
            "ResNet50", "MobileNetV2", "ShuffleV1", "ShuffleV2",
        ],
    )
    parser.add_argument("--path_t", type=str, default=None, required=True)

    parser.add_argument("--kd_T", type=float, default=4)
    parser.add_argument("--trial", type=str, default="1")
    parser.add_argument("-a", "--alpha", type=float, default=None)

    opt = parser.parse_args()

    if opt.model_s in ["MobileNetV2", "ShuffleV1", "ShuffleV2"]:
        opt.lr = 0.01

    if hostname.startswith("visiongpu"):
        opt.model_path = "/path/to/my/student_model"
        opt.tb_path = "/path/to/my/student_tensorboards"
    else:
        opt.model_path = "./save/student_model"
        opt.tb_path = "./save/student_tensorboards"

    iters = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = [int(x) for x in iters]

    opt.model_name = "S:{}_T:{}_{}_{}_a:{}_{}".format(
        opt.model_s, opt.model_t, opt.dataset, "mlkd", opt.alpha, opt.trial
    )

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def load_teacher(model_path, model_t, n_cls):
    print("==> loading teacher model")
    model = model_dict[model_t](num_classes=n_cls)

    if torch.cuda.is_available():
        ckpt = torch.load(model_path)
    else:
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    print("==> done")
    return model


def main():
    best_acc = 0.0
    opt = parse_option()

    if opt.dataset == "cifar100":
        train_loader, held_loader, val_loader = get_cifar100_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            held_size=opt.held_size,
            num_held_samples=opt.num_held_samples,
        )
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    model_t = load_teacher(opt.path_t, opt.model_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    criterion_cls = nn.CrossEntropyLoss()
    if opt.loss_type == "mse":
        criterion_kd = MSEWithTemperature(T=opt.kd_T)
    elif opt.loss_type == "kl":
        criterion_kd = CustomDistillKL(T=opt.kd_T)
    else:
        raise NotImplementedError()

    module_list = nn.ModuleList([model_s, model_t])
    criterion_list = nn.ModuleList([criterion_cls, criterion_kd])

    s_optimizer = optim.SGD(
        model_s.parameters(),
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )
    t_optimizer = optim.SGD(
        model_t.parameters(),
        lr=opt.teacher_lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    teacher_acc, teacher_acc5, teacher_loss = validate(val_loader, model_t, criterion_cls, opt)
    print("initial teacher accuracy: ", teacher_acc)
    print("initial teacher top5:     ", teacher_acc5)
    print("initial teacher loss:     ", teacher_loss)

    for param_group in t_optimizer.param_groups:
        param_group["lr"] = 0.0

    for epoch in range(1, opt.epochs + 1):
        if epoch == 150:
            for param_group in t_optimizer.param_groups:
                param_group["lr"] = opt.teacher_lr
            opt.assume_s_step_size *= opt.lr_decay_rate

        if epoch == 180:
            for param_group in t_optimizer.param_groups:
                param_group["lr"] *= opt.lr_decay_rate
            opt.assume_s_step_size *= opt.lr_decay_rate

        if epoch == 210:
            for param_group in t_optimizer.param_groups:
                param_group["lr"] *= opt.lr_decay_rate
            opt.assume_s_step_size *= opt.lr_decay_rate

        adjust_learning_rate(epoch, opt, s_optimizer)

        print("==> training...")
        time1 = time.time()

        train_acc, train_loss = train(
            epoch,
            train_loader,
            held_loader,
            module_list,
            criterion_list,
            s_optimizer,
            t_optimizer,
            opt,
        )

        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {"epoch": epoch, "model": model_s.state_dict(), "best_acc": best_acc}
            save_file = os.path.join(opt.save_folder, "{}_best.pth".format(opt.model_s))
            print("saving the best model!")
            torch.save(state, save_file)

        if epoch % opt.save_freq == 0:
            print("==> Saving...")
            state = {"epoch": epoch, "model": model_s.state_dict(), "accuracy": test_acc}
            save_file = os.path.join(opt.save_folder, "ckpt_epoch_{:d}.pth".format(epoch))
            torch.save(state, save_file)

    print("best student accuracy:", best_acc)

    final_teacher_acc, final_teacher_acc5, final_teacher_loss = validate(
        val_loader, model_t, criterion_cls, opt
    )
    print("final teacher accuracy: ", final_teacher_acc)
    print("final teacher top5:     ", final_teacher_acc5)
    print("final teacher loss:     ", final_teacher_loss)

    student_state = {"opt": opt, "model": model_s.state_dict()}
    student_save_file = os.path.join(opt.save_folder, "{}_last.pth".format(opt.model_s))
    torch.save(student_state, student_save_file)

    teacher_state = {"opt": opt, "model": model_t.state_dict()}
    teacher_save_file = os.path.join(opt.save_folder, "{}_teacher_last.pth".format(opt.model_t))
    torch.save(teacher_state, teacher_save_file)


if __name__ == "__main__":
    main()