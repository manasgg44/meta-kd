from __future__ import print_function, division

import sys
import time
from collections import OrderedDict
from copy import deepcopy as cp

import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        v = float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_distill(epoch, train_loader, held_loader, module_list, criterion_list,
                 s_optimizer, t_optimizer, ta_optimizer, opt):
    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    s_model = module_list[0]
    t_model = module_list[-2]
    ta_model = module_list[-1]

    batch_time = AverageMeter()
    real_losses = AverageMeter()
    real_top1 = AverageMeter()
    real_top5 = AverageMeter()
    held_losses = AverageMeter()

    end = time.time()

    total_steps = len(train_loader)
    buffer = []
    round_counter = 0

    for d_idx, d_data in enumerate(train_loader):
        buffer.append((d_idx, d_data))

        if (d_idx + 1) % opt.num_meta_batches != 0 and (d_idx + 1) != total_steps:
            continue

        fast_weights = OrderedDict((n, p) for (n, p) in s_model.named_parameters())
        s_backup = cp(s_model.state_dict())
        s_opt_backup = cp(s_optimizer.state_dict())

        s_model.train()
        ta_model.eval()

        for idx, data in buffer:
            x, y = data
            x = x.float()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            out_s = s_model(x, params=None if idx == buffer[0][0] else fast_weights)
            out_ta = ta_model(x)

            loss_ce = criterion_cls(out_s, y)
            loss_kd = criterion_kd(out_s, out_ta)
            loss = opt.alpha * loss_kd + (1.0 - opt.alpha) * loss_ce

            grads = torch.autograd.grad(
                loss,
                s_model.parameters() if idx == buffer[0][0] else fast_weights.values(),
                create_graph=True,
                retain_graph=True
            )
            fast_weights = OrderedDict(
                (n, p - opt.assume_s_step_size * g) for ((n, p), g) in zip(fast_weights.items(), grads)
            )

        ta_model.train()
        t_model.train()

        s_prime_loss = None
        held_batches = 0

        for hx, hy in held_loader:
            hx = hx.float()
            if torch.cuda.is_available():
                hx = hx.cuda()
                hy = hy.cuda()

            out_s_prime = s_model(hx, params=fast_weights)
            step_loss = criterion_cls(out_s_prime, hy)

            s_prime_loss = step_loss if s_prime_loss is None else (s_prime_loss + step_loss)
            held_batches += 1

        s_prime_loss = s_prime_loss / max(1, held_batches)
        held_losses.update(s_prime_loss.item(), 1)

        ta_grads = torch.autograd.grad(s_prime_loss, ta_model.parameters(), retain_graph=opt.meta_update_teacher)
        for p, g in zip(ta_model.parameters(), ta_grads):
            p.grad = g

        ta_optimizer.step()
        for p in ta_model.parameters():
            p.grad = None

        if getattr(opt, 'meta_update_teacher', False):
            t_grads = torch.autograd.grad(s_prime_loss, t_model.parameters())
            for p, g in zip(t_model.parameters(), t_grads):
                p.grad = g

            t_optimizer.step()
            for p in t_model.parameters():
                p.grad = None

        for p in s_model.parameters():
            p.grad = None

        del fast_weights

        s_model.load_state_dict(s_backup)
        s_optimizer.load_state_dict(s_opt_backup)

        s_model.train()
        t_model.eval()
        ta_model.eval()

        for idx, data in buffer:
            x, y = data
            x = x.float()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            out_s = s_model(x)
            with torch.no_grad():
                out_t = t_model(x)
                out_ta = ta_model(x)

            loss_ce = criterion_cls(out_s, y)
            loss_kd_t = criterion_kd(out_s, out_t)
            loss_kd_ta = criterion_kd(out_s, out_ta)

            loss = (opt.alpha * opt.beta) * loss_kd_t + (opt.alpha * (1.0 - opt.beta)) * loss_kd_ta + (1.0 - opt.alpha) * loss_ce

            acc1, acc5 = accuracy(out_s, y, topk=(1, 5))
            real_losses.update(loss.item(), x.size(0))
            real_top1.update(acc1[0], x.size(0))
            real_top5.update(acc5[0], x.size(0))

            loss.backward()
            s_optimizer.step()
            s_optimizer.zero_grad()

        buffer = []
        round_counter += 1

        batch_time.update(time.time() - end)
        end = time.time()

        if round_counter % opt.print_freq == 0:
            print('Epoch: [{0}]\tTime {t.val:.3f} ({t.avg:.3f})\tLoss {l.val:.4f} ({l.avg:.4f})\t'
                  'Acc@1 {a1.val:.3f} ({a1.avg:.3f})\tAcc@5 {a5.val:.3f} ({a5.avg:.3f})\tHeld {h.avg:.4f}'.format(
                      epoch, t=batch_time, l=real_losses, a1=real_top1, a5=real_top5, h=held_losses))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=real_top1, top5=real_top5))
    return real_top1.avg, real_losses.avg


def validate(val_loader, model, criterion, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (x, y) in enumerate(val_loader):
            x = x.float()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            out = model(x)
            loss = criterion(out, y)

            acc1, acc5 = accuracy(out, y, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\tTime {t.val:.3f} ({t.avg:.3f})\tLoss {l.val:.4f} ({l.avg:.4f})\t'
                      'Acc@1 {a1.val:.3f} ({a1.avg:.3f})\tAcc@5 {a5.val:.3f} ({a5.avg:.3f})'.format(
                          idx, len(val_loader), t=batch_time, l=losses, a1=top1, a5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg