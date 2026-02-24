from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from dataset.meta_cifar100 import get_cifar100_dataloaders
from helper.util import adjust_learning_rate

from distiller_zoo.KD import CustomDistillKL
from distiller_zoo.MSE import MSEWithTemperature

from helper.meta_loops_ta import train_distill as train, validate


def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('meta distill with TA')

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--tb_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=240)

    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--teacher_lr', type=float, default=0.05)
    parser.add_argument('--ta_lr', type=float, default=None)
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'kl'], required=True)

    parser.add_argument('--held_size', type=int, required=True)
    parser.add_argument('--num_held_samples', type=int, required=True)
    parser.add_argument('--num_meta_batches', type=int, default=1)
    parser.add_argument('--assume_s_step_size', type=float, default=0.05)

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'])

    parser.add_argument('--model_s', type=str, required=True,
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--model_t', type=str, required=True)
    parser.add_argument('--model_ta', type=str, default=None)

    parser.add_argument('--path_t', type=str, required=True)
    parser.add_argument('--path_ta', type=str, default=None)

    parser.add_argument('--kd_T', type=float, default=4.0)
    parser.add_argument('--trial', type=str, default='1')

    parser.add_argument('-a', '--alpha', type=float, required=True)
    parser.add_argument('-b', '--beta', type=float, default=0.5)

    parser.add_argument('--freeze_teacher_until', type=int, default=150)
    parser.add_argument('--meta_update_teacher', action='store_true')

    opt = parser.parse_args()

    if opt.ta_lr is None:
        opt.ta_lr = opt.teacher_lr
    if opt.model_ta is None:
        opt.model_ta = opt.model_t
    if opt.path_ta is None:
        opt.path_ta = opt.path_t

    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]

    opt.model_name = 'S:{}_T:{}_TA:{}_{}_{}_a:{}_b:{}_{}'.format(
        opt.model_s, opt.model_t, opt.model_ta, opt.dataset, 'mlkd_ta', opt.alpha, opt.beta, opt.trial
    )

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def load_model(arch, ckpt_path, n_cls):
    model = model_dict[arch](num_classes=n_cls)
    ckpt = torch.load(ckpt_path, map_location='cpu' if not torch.cuda.is_available() else None)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    return model


def main():
    opt = parse_option()
    best_acc = 0.0
    best_epoch = 0

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.dataset == 'cifar100':
        train_loader, held_loader, val_loader = get_cifar100_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            held_size=opt.held_size,
            num_held_samples=opt.num_held_samples,
        )
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    model_s = model_dict[opt.model_s](num_classes=n_cls)
    model_t = load_model(opt.model_t, opt.path_t, n_cls)
    model_ta = load_model(opt.model_ta, opt.path_ta, n_cls)

    module_list = nn.ModuleList([model_s, model_t, model_ta])

    criterion_cls = nn.CrossEntropyLoss()
    if opt.loss_type == 'mse':
        criterion_kd = MSEWithTemperature(T=opt.kd_T)
    else:
        criterion_kd = CustomDistillKL(T=opt.kd_T)

    criterion_list = nn.ModuleList([criterion_cls, criterion_kd])

    s_optimizer = optim.SGD(model_s.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    t_optimizer = optim.SGD(model_t.parameters(), lr=opt.teacher_lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    ta_optimizer = optim.SGD(model_ta.parameters(), lr=opt.ta_lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    t_acc0, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    ta_acc0, _, _ = validate(val_loader, model_ta, criterion_cls, opt)
    print('teacher before:', t_acc0)
    print('ta before     :', ta_acc0)

    for pg in t_optimizer.param_groups:
        pg['lr'] = 0.0

    for epoch in range(1, opt.epochs + 1):
        if epoch == opt.freeze_teacher_until:
            for pg in t_optimizer.param_groups:
                pg['lr'] = opt.teacher_lr
            opt.assume_s_step_size *= opt.lr_decay_rate

        if epoch in opt.lr_decay_epochs:
            for pg in t_optimizer.param_groups:
                pg['lr'] *= opt.lr_decay_rate
            for pg in ta_optimizer.param_groups:
                pg['lr'] *= opt.lr_decay_rate
            opt.assume_s_step_size *= opt.lr_decay_rate

        adjust_learning_rate(epoch, opt, s_optimizer)

        train_acc, train_loss = train(
            epoch, train_loader, held_loader,
            module_list, criterion_list,
            s_optimizer, t_optimizer, ta_optimizer,
            opt
        )

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        if test_acc > best_acc:
            best_acc = float(test_acc)
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model': model_s.state_dict(), 'best_acc': best_acc},
                       os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s)))

        if epoch % opt.save_freq == 0:
            torch.save({'epoch': epoch, 'model': model_s.state_dict(), 'accuracy': float(test_acc)},
                       os.path.join(opt.save_folder, 'ckpt_epoch_{}.pth'.format(epoch)))

    print('best student acc:', best_acc)
    print('best epoch:', best_epoch)

    t_acc1, t_top51, t_loss1 = validate(val_loader, model_t, criterion_cls, opt)
    ta_acc1, ta_top51, ta_loss1 = validate(val_loader, model_ta, criterion_cls, opt)
    print('teacher after: Acc@1 {:.3f} Acc@5 {:.3f} Loss {:.4f}'.format(t_acc1, t_top51, t_loss1))
    print('ta after    : Acc@1 {:.3f} Acc@5 {:.3f} Loss {:.4f}'.format(ta_acc1, ta_top51, ta_loss1))

    torch.save({'opt': opt, 'model': model_s.state_dict()},
               os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s)))
    torch.save({'opt': opt, 'model': model_t.state_dict(), 'acc1': float(t_acc1), 'acc5': float(t_top51), 'loss': float(t_loss1)},
               os.path.join(opt.save_folder, '{}_teacher_last.pth'.format(opt.model_t)))
    torch.save({'opt': opt, 'model': model_ta.state_dict(), 'acc1': float(ta_acc1), 'acc5': float(ta_top51), 'loss': float(ta_loss1)},
               os.path.join(opt.save_folder, '{}_ta_last.pth'.format(opt.model_ta)))


if __name__ == '__main__':
    main()