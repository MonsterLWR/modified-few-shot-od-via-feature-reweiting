import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from cfg import cfg
from numbers import Number
from random import random


def neg_filter(pred_boxes, target, neg_ratio=3, withids=False):
    """filter whole batches out?"""
    assert pred_boxes.size(0) == target.size(0)
    if neg_ratio == 'full':
        inds = list(range(pred_boxes.size(0)))
    elif isinstance(neg_ratio, Number):
        # flags indicate which base classes are in the target
        # maybe these code is wrong, flags should indicate anchors containing object
        flags = torch.sum(target, 1) != 0
        flags = flags.cpu().data.tolist()
        # ratio: expected neg samples / existing neg samples
        ratio = neg_ratio * sum(flags) * 1. / (len(flags) - sum(flags))
        if ratio >= 1:
            inds = list(range(pred_boxes.size(0)))
        else:
            # random() > ratio = ratio
            # sample a neg sample with probability ratio
            flags = [0 if f == 0 and random() > ratio else 1 for f in flags]
            inds = np.argwhere(flags).squeeze()
            pred_boxes, target = pred_boxes[inds], target[inds]
    else:
        raise NotImplementedError('neg_ratio not recognized')
    if withids:
        return pred_boxes, target, inds
    else:
        return pred_boxes, target


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale,
                  sil_thresh, seen, max_boxes=cfg.max_boxes):
    """mainly used for matching target boxes with pred_boxes,do not take cls into consideration"""
    # self.coord_scale = 1
    # self.noobject_scale = 1
    # self.object_scale = 5
    # self.class_scale = 1

    # num boxs here means the number of base classes selected in one batch
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors) // num_anchors
    # print('anchor_step: ', anchor_step)
    # note default value of conf_mask is noobject_scale
    # conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    conf_mask = torch.ones(nB, nA, nH, nW)
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask = torch.zeros(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW)

    total_anchors = nA * nH * nW
    nPixels = nH * nW
    for box in range(nB):
        # one box indicate the whole output for a base class in one batch
        # corresponding boxes for one base class, transpose for convenience in calculating iou
        cur_pred_boxes = pred_boxes[box * total_anchors:(box + 1) * total_anchors].t()
        cur_ious = torch.zeros(total_anchors)
        for t in range(max_boxes):
            # why not target[box][t * 5 + 0] == 0?
            if target[box][t * 5 + 1] == 0:
                break
            gx = target[box][t * 5 + 1] * nW
            gy = target[box][t * 5 + 2] * nH
            gw = target[box][t * 5 + 3] * nW
            gh = target[box][t * 5 + 4] * nH
            cur_gt_boxes = torch.tensor([gx, gy, gw, gh]).float().repeat(total_anchors, 1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        # Find anchors with iou > sil_thresh
        # no conf_loss for that one
        conf_mask[box][cur_ious.view(conf_mask[box].shape) > sil_thresh] = 0
    if seen < 12800:
        # anchor_step is always 2
        if anchor_step == 4:
            print("anchor step is four!")
            # tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1, nA, 1,
            #                                                                                                   1).repeat(
            #     nB, 1, nH, nW)
            # ty = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1, nA, 1,
            #                                                                                                   1).repeat(
            #     nB, 1, nH, nW)
        else:
            tx.fill_(0.5)
            ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        # coord loss for all anchors
        coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for box in range(nB):
        # for t in range(50):  # not cfg.max_boxes?
        for t in range(max_boxes):
            if target[box][t * 5 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            # min_dist = 10000
            gx = target[box][t * 5 + 1] * nW
            gy = target[box][t * 5 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[box][t * 5 + 3] * nW
            gh = target[box][t * 5 + 4] * nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):
                # find the best anchor
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = [0, 0, aw, ah]
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    print('anchor_step is 4')
                    # ax = anchors[anchor_step * n + 2]
                    # ay = anchors[anchor_step * n + 3]
                    # dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                # elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                #     print('anchor_step is 4')
                # best_iou = iou
                # best_n = n
                # min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[box * total_anchors + best_n * nPixels + gj * nW + gi]

            coord_mask[box][best_n][gj][gi] = 1
            cls_mask[box][best_n][gj][gi] = 1
            # conf_mask[box][best_n][gj][gi] = object_scale
            conf_mask[box][best_n][gj][gi] = 1
            tx[box][best_n][gj][gi] = target[box][t * 5 + 1] * nW - gi
            ty[box][best_n][gj][gi] = target[box][t * 5 + 2] * nH - gj
            tw[box][best_n][gj][gi] = math.log(gw / anchors[anchor_step * best_n])
            th[box][best_n][gj][gi] = math.log(gh / anchors[anchor_step * best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            tconf[box][best_n][gj][gi] = iou
            tcls[box][best_n][gj][gi] = target[box][t * 5]
            if iou > 0.5:
                # did not consider the result of cls
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


class RegionLossV2(nn.Module):
    def __init__(self, num_classes_per_anchor=None, anchors=None, num_anchors=None,
                 coord_scale=1, noobject_scale=1, object_scale=5, class_scale=1,
                 thresh=0.6, seen=0, show_time=False):
        super(RegionLossV2, self).__init__()
        self.num_classes_per_anchor = num_classes_per_anchor
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh
        self.seen = seen
        self.show_time = show_time

        self.anchor_step = len(anchors) // num_anchors

        # print('class_scale', self.class_scale)

    def forward(self, output, target):
        # output: batch_size*num_base_classes, num_anchor*(id+four_position+num_classes_per_anchor), h, w
        # Get all classification prediction
        # target: batch_size, num_base_classes, num_max_boxes*(id+four_position)
        batch_size = target.shape[0]
        base_cls_num = target.shape[1]
        num_anchor = self.num_anchors
        num_cls_per_anchor = self.num_classes_per_anchor
        output_height = output.shape[2]
        output_width = output.shape[3]

        cls = output.view(output.shape[0], num_anchor, 5 + num_cls_per_anchor, output_height, output_width)
        cls = cls.index_select(2,
                               torch.linspace(5, 5 + num_cls_per_anchor - 1,
                                              num_cls_per_anchor).long().cuda()).squeeze()
        # cls shape: batch_size * num_anchor * num_cls_per_anchor * output_height * output_width, base_cls_num
        cls = cls.view(batch_size, base_cls_num, -1).transpose(1, 2).contiguous().view(-1, base_cls_num)

        # target reshape to: batch_size*num_basec_classes, num_max_boxes*(id+four_position)
        target = target.view(-1, target.size(-1))
        output, target, inds = neg_filter(output, target, neg_ratio=cfg.neg_ratio, withids=True)
        # counts of boxes in each sample
        counts, _ = np.histogram(inds, bins=batch_size, range=(0, batch_size * base_cls_num))

        t0 = time.time()
        num_boxes = output.shape[0]

        output = output.view(num_boxes, num_anchor, (5 + num_cls_per_anchor), output_height, output_width)
        x = torch.sigmoid(output.index_select(2, torch.tensor([0]).cuda())
                          .view(num_boxes, num_anchor, output_height, output_width))
        y = torch.sigmoid(output.index_select(2, torch.tensor([1]).cuda())
                          .view(num_boxes, num_anchor, output_height, output_width))
        w = output.index_select(2, torch.tensor([2]).cuda()) \
            .view(num_boxes, num_anchor, output_height, output_width)
        h = output.index_select(2, torch.tensor([3]).cuda()) \
            .view(num_boxes, num_anchor, output_height, output_width)
        conf = torch.sigmoid(output.index_select(2, torch.tensor([4]).cuda())
                             .view(num_boxes, num_anchor, output_height, output_width))
        t1 = time.time()

        pred_boxes = torch.zeros(4, num_boxes * num_anchor * output_height * output_width).float().cuda()
        grid_x = torch.linspace(0, output_width - 1, output_width).repeat(output_height, 1) \
            .repeat(num_boxes * num_anchor, 1, 1).view(num_boxes * num_anchor * output_height * output_width).cuda()
        grid_y = torch.linspace(0, output_height - 1, output_height).repeat(output_width, 1).t() \
            .repeat(num_boxes * num_anchor, 1, 1).view(num_boxes * num_anchor * output_height * output_width).cuda()
        anchor_w = torch.tensor(self.anchors).view(num_anchor, self.anchor_step) \
            .index_select(1, torch.tensor([0])).cuda()
        anchor_h = torch.tensor(self.anchors).view(num_anchor, self.anchor_step) \
            .index_select(1, torch.tensor([1])).cuda()
        # it's ok to change parameters in the second repeat to (1,nH * nW)?
        anchor_w = anchor_w.repeat(num_boxes, 1).repeat(1, 1, output_height * output_width) \
            .view(num_boxes * num_anchor * output_height * output_width)
        anchor_h = anchor_h.repeat(num_boxes, 1).repeat(1, 1, output_height * output_width) \
            .view(num_boxes * num_anchor * output_height * output_width)
        pred_boxes[0] = x.data.view(grid_x.shape) + grid_x
        pred_boxes[1] = y.data.view(grid_y.shape) + grid_y
        pred_boxes[2] = torch.exp(w.data).view(anchor_w.shape) * anchor_w
        pred_boxes[3] = torch.exp(h.data).view(anchor_h.shape) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))
        t2 = time.time()

        # note using target.data here
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes,
                                                                                                    target.data,
                                                                                                    self.anchors,
                                                                                                    num_anchor,
                                                                                                    num_cls_per_anchor,
                                                                                                    output_height,
                                                                                                    output_width,
                                                                                                    self.noobject_scale,
                                                                                                    self.object_scale,
                                                                                                    self.thresh,
                                                                                                    self.seen)

        cls_num = torch.sum(cls_mask)
        idx_start = 0
        cls_mask_list = []
        tcls_list = []
        for i in range(len(counts)):
            # iterate over each sample
            if counts[i] == 0:
                cur_mask = torch.zeros(num_anchor, output_height, output_width)
                cur_tcls = torch.zeros(num_anchor, output_height, output_width)
            else:
                # what if tow gt collide in the same anchor
                cur_mask = torch.sum(cls_mask[idx_start:idx_start + counts[i]], dim=0)
                cur_tcls = torch.sum(tcls[idx_start:idx_start + counts[i]], dim=0)
            cls_mask_list.append(cur_mask)
            tcls_list.append(cur_tcls)
            idx_start += counts[i]
        cls_mask = torch.stack(cls_mask_list)
        tcls = torch.stack(tcls_list)

        colliding_num = cls_mask[cls_mask > 1].sum()
        # ignoring two gt collide in on anchor
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).float().sum().item())

        tx = tx.cuda()
        ty = ty.cuda()
        tw = tw.cuda()
        th = th.cuda()
        tconf = tconf.cuda()

        coord_mask = coord_mask.cuda()
        # conf_mask = conf_mask.cuda().sqrt()
        conf_mask = conf_mask.cuda()
        # each class use the same mask
        cls = cls[cls_mask.view(-1, 1).repeat(1, base_cls_num).cuda()].view(-1, base_cls_num)

        tcls = tcls[cls_mask].long().cuda()
        # ClassificationLoss = nn.CrossEntropyLoss()

        t3 = time.time()
        # loss_x = self.coord_scale * nn.MSELoss()(x * coord_mask, tx * coord_mask) / 2.0
        loss_x = self.coord_scale * nn.BCELoss()(x * coord_mask, tx * coord_mask)
        # loss_y = self.coord_scale * nn.MSELoss()(y * coord_mask, ty * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.BCELoss()(y * coord_mask, ty * coord_mask)
        loss_w = self.coord_scale * nn.MSELoss()(w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss()(h * coord_mask, th * coord_mask) / 2.0
        # loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask) / 2.0
        loss_conf = nn.BCELoss()(conf * conf_mask, tconf * conf_mask)
        loss_cls = self.class_scale * nn.CrossEntropyLoss()(cls, tcls)

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()

        if self.show_time:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))

        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
            self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
            loss_conf.item(), loss_cls.item(), loss.item()))

        return loss


if __name__ == '__main__':
    loss = RegionLossV2()
