import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger

from .base_mapper import BaseMapper, MAPPERS
from ..utils.memory_buffer import StreamTensorMemory


@MAPPERS.register_module()
class AMapDistiller(BaseMapper):
    '''AMap-style distillation on StreamMapNet.

    A frozen streaming teacher that sees frames in reverse time order (its BEV /
    query memory holds the future) teaches a single-frame student with
        - masked L2 on the basic BEV (teacher backbone output vs student backbone output)
        - masked L2 on the refined BEV (teacher GRU output vs adapter(student backbone output))
        - Bernoulli KL on class logits of Hungarian-matched student / teacher queries
    Only the student is used at test time; the adapter is training-only.

    Args:
        student_cfg (dict): single-frame StreamMapNet config
        teacher_config (str): path of the teacher's config file
        teacher_ckpt (str): teacher checkpoint, loaded in init_weights()
        batch_size (int): per-gpu batch size, overrides the teacher's memory size
        bev_channels (int): channels of the BEV features
        data_img_norm (dict): normalization the data pipeline applies (teacher's)
        student_img_norm (dict): normalization the student's image backbone expects;
            None if the same as data_img_norm
        distill_cfg (dict): loss weights and matching settings
    '''

    def __init__(self,
                 student_cfg,
                 teacher_config,
                 teacher_ckpt=None,
                 batch_size=1,
                 bev_channels=256,
                 data_img_norm=None,
                 student_img_norm=None,
                 distill_cfg=dict(),
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.student = MAPPERS.build(student_cfg)

        teacher_cfg = mmcv.Config.fromfile(teacher_config).model
        teacher_cfg.streaming_cfg.batch_size = batch_size
        teacher_cfg.head_cfg.streaming_cfg.batch_size = batch_size
        self.teacher = MAPPERS.build(teacher_cfg)
        self.teacher_ckpt = teacher_ckpt
        for p in self.teacher.parameters():
            p.requires_grad = False

        # training-only: predicts the teacher's refined (future-fused) BEV from the student's BEV
        self.refine_adapter = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, kernel_size=1),
        )

        self.data_img_norm = data_img_norm
        self.student_img_norm = student_img_norm
        if student_img_norm is not None:
            assert data_img_norm is not None
            self.register_buffer('data_mean', torch.tensor(data_img_norm['mean']).view(1, 1, 3, 1, 1))
            self.register_buffer('data_std', torch.tensor(data_img_norm['std']).view(1, 1, 3, 1, 1))
            self.register_buffer('student_mean', torch.tensor(student_img_norm['mean']).view(1, 1, 3, 1, 1))
            self.register_buffer('student_std', torch.tensor(student_img_norm['std']).view(1, 1, 3, 1, 1))

        self.basic_bev_weight = distill_cfg.get('basic_bev_weight', 1.0)
        self.refined_bev_weight = distill_cfg.get('refined_bev_weight', 1.0)
        self.query_weight = distill_cfg.get('query_weight', 1.0)
        self.use_bev_mask = distill_cfg.get('use_bev_mask', True)
        self.mask_thickness = distill_cfg.get('mask_thickness', 3)
        self.match_reg_weight = distill_cfg.get('match_reg_weight', 5.0)
        self.match_cls_weight = distill_cfg.get('match_cls_weight', 1.0)

        self.bev_h = self.student.bev_h
        self.bev_w = self.student.bev_w
        self.num_points = self.student.head.num_points

    def init_weights(self, pretrained=None):
        self.student.init_weights()
        if self.teacher_ckpt:
            load_checkpoint(self.teacher, self.teacher_ckpt, map_location='cpu',
                            strict=True, logger=get_root_logger())

    def train(self, mode=True):
        super().train(mode)
        # teacher modules always run in eval mode (no dropout / grid mask),
        # but its streaming memory follows the training batch layout
        self.teacher.eval()
        for module in (self.teacher, self.teacher.head):
            for v in module.__dict__.values():
                if isinstance(v, StreamTensorMemory):
                    v.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to_student_input(self, img):
        '''Re-normalize images from the data pipeline's normalization to the student's.
        img: [B, N, 3, H, W]
        '''
        if self.student_img_norm is None:
            return img
        img = img * self.data_std + self.data_mean
        if self.data_img_norm['to_rgb'] != self.student_img_norm['to_rgb']:
            img = img.flip(2)
        return (img - self.student_mean) / self.student_std

    @torch.no_grad()
    def teacher_forward(self, img, img_metas):
        t = self.teacher
        basic_bev, refined_bev = t.extract_bev(img, img_metas)
        bev_feats = t.neck(refined_bev)
        preds = t.head(bev_feats, img_metas=img_metas, return_loss=False)[-1]
        return basic_bev, refined_bev, preds

    def gt_bev_mask(self, gts, device):
        '''Rasterize gt lines (normalized to [0, 1]) to a [B, bev_h, bev_w] binary mask.
        Row 0 of the BEV grid is +y (see StreamMapNet.plane).
        '''
        lines_list = gts[0]['lines']
        masks = []
        scale = np.array([self.bev_w, self.bev_h], dtype=np.float32)
        for lines in lines_list:
            mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
            lines = lines.detach().cpu().numpy()
            if lines.ndim == 3: # permutations, take the original order
                lines = lines[:, 0]
            lines = lines.reshape(len(lines), -1, 2)
            for line in lines:
                pts = line.copy()
                pts[:, 1] = 1.0 - pts[:, 1]
                pts = np.round(pts * scale).astype(np.int32)
                cv2.polylines(mask, [pts], False, color=1, thickness=self.mask_thickness)
            masks.append(torch.from_numpy(mask))
        return torch.stack(masks).to(device).float()

    @staticmethod
    def masked_feat_loss(feat_s, feat_t, mask):
        dist = torch.norm(feat_s - feat_t, p=2, dim=1) # [B, H, W]
        return (dist * mask).sum() / mask.sum().clamp(min=1.0)

    def query_kd_loss(self, s_preds, t_preds):
        '''Hungarian-match student queries to teacher queries on (lines, class probs),
        then Bernoulli KL(teacher || student) on class logits of matched pairs,
        summed over queries and classes, averaged over the batch.
        '''
        s_lines, s_logits = s_preds['lines'], s_preds['scores']
        t_lines, t_logits = t_preds['lines'], t_preds['scores']
        bs = len(s_lines)
        loss = s_logits[0].new_zeros(())
        for i in range(bs):
            with torch.no_grad():
                sl, tl = s_lines[i].float(), t_lines[i].float()
                tl_flip = tl.view(-1, self.num_points, 2).flip(1).reshape(len(tl), -1)
                reg_cost = torch.minimum(torch.cdist(sl, tl, p=1), torch.cdist(sl, tl_flip, p=1))
                reg_cost = reg_cost / sl.shape[-1]
                cls_cost = torch.cdist(s_logits[i].float().sigmoid(), t_logits[i].float().sigmoid(), p=1)
                cost = self.match_reg_weight * reg_cost + self.match_cls_weight * cls_cost
                s_idx, t_idx = linear_sum_assignment(cost.cpu().numpy())
            zs = s_logits[i][s_idx]
            zt = t_logits[i][t_idx].detach()
            pt = zt.sigmoid()
            kl = pt * (F.logsigmoid(zt) - F.logsigmoid(zs)) + \
                (1 - pt) * (F.logsigmoid(-zt) - F.logsigmoid(-zs))
            loss = loss + kl.sum()
        return loss / bs

    def forward_train(self, img, vectors, points=None, img_metas=None, **kwargs):
        student = self.student
        gts, img, img_metas, valid_idx, points = student.batch_data(
            vectors, img, img_metas, img.device, points)

        t_basic, t_refined, t_preds = self.teacher_forward(img, img_metas)

        s_basic, _ = student.extract_bev(self.to_student_input(img), img_metas, points=points)
        bev_feats = student.neck(s_basic)
        preds_list, loss_dict, _, _ = student.head(
            bev_features=bev_feats,
            img_metas=img_metas,
            gts=gts,
            return_loss=True)

        if self.use_bev_mask:
            mask = self.gt_bev_mask(gts, img.device)
        else:
            mask = s_basic.new_ones((s_basic.shape[0], self.bev_h, self.bev_w))
        if self.basic_bev_weight > 0:
            loss_dict['kd_basic_bev'] = self.basic_bev_weight * \
                self.masked_feat_loss(s_basic, t_basic, mask)
        if self.refined_bev_weight > 0:
            loss_dict['kd_refined_bev'] = self.refined_bev_weight * \
                self.masked_feat_loss(self.refine_adapter(s_basic), t_refined, mask)
        if self.query_weight > 0:
            loss_dict['kd_query'] = self.query_weight * \
                self.query_kd_loss(preds_list[-1], t_preds)

        loss = 0
        for name, var in loss_dict.items():
            loss = loss + var

        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({'total': loss.item()})

        return loss, log_vars, img.size(0)

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, **kwargs):
        return self.student.forward_test(self.to_student_input(img), points=points, img_metas=img_metas)
