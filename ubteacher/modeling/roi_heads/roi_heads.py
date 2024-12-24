# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    Res5ROIHeads,
)
from ubteacher.modeling.roi_heads.fast_rcnn import (
    FastRCNNCrossEntropyLossOutputLayers,
    FastRCNNFocaltLossOutputLayers,
)

import numpy as np
from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F
from detectron2.layers import cat


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsPseudoLab(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            self.box_predictor = FastRCNNCrossEntropyLossOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1))
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            self.box_predictor = FastRCNNFocaltLossOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1))
        else:
            raise ValueError("Unknown ROI head loss.")
        
        # 维护类原型
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.queue_len = 128 # cfg.MODEL.ROI_HEADS.QUEUE_LEN
        if True:
            self.register_buffer("queue_s", torch.zeros(self.num_classes, self.queue_len, out_channels))
            self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
            self.register_buffer("queue_full", torch.zeros(self.num_classes, dtype=torch.long))
        
        if self.num_classes in [15, 20]:
            self.novel_index = [15, 16, 17, 18, 19]
        elif self.num_classes in [60, 80]:
            self.novel_index = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
        self.base_index = []
        for i in range(self.num_classes):
            if i not in self.novel_index:
                self.base_index.append(i)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, gt_class):
        keys_s = keys_s[:self.queue_len]
        batch_size = keys_s.shape[0]
        ptr = int(self.queue_ptr[gt_class])
        if ptr + batch_size <= self.queue_len:
            self.queue_s[gt_class, ptr:ptr + batch_size] = keys_s
        else:
            self.queue_s[gt_class, ptr:] = keys_s[:self.queue_len - ptr]
            self.queue_s[gt_class, :(ptr + batch_size) % self.queue_len] = keys_s[self.queue_len - ptr:]
            
        if ptr + batch_size >= self.queue_len:
            self.queue_full[gt_class] = 1
        ptr = (ptr + batch_size) % self.queue_len
        self.queue_ptr[gt_class] = ptr
        
    @torch.no_grad()
    def update_memory(self, features_s, gt_classes):
        fg_cases = (gt_classes >= 0) & (gt_classes < self.num_classes)
        features_fg_s = features_s[fg_cases]
        gt_classes_fg = gt_classes[fg_cases]
        
        if len(gt_classes_fg) == 0:
            return
        uniq_c = torch.unique(gt_classes_fg)
            
        for c in uniq_c:
            c = int(c)
            c_index = torch.nonzero(
                gt_classes_fg == c, as_tuple=False
            ).squeeze(1)
            features_c_s = features_fg_s[c_index]
            self._dequeue_and_enqueue(features_c_s, c)
    
    @torch.no_grad()
    def init_prototypes(self, features, targets):
        gt_boxes = [x.gt_boxes for x in targets]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], gt_boxes
        )
        feature_pooled_s = box_features.mean(dim=[2, 3])

        gt_classes = torch.cat([x.gt_classes for x in targets], dim=0)

        self.update_memory(feature_pooled_s.detach(), gt_classes)
    
    @torch.no_grad()
    def get_prototypes(self):
        prototypes_s = []
        for i in range(self.num_classes):
            if self.queue_full[i] or self.queue_ptr[i] == 0:
                prototypes_s.append(self.queue_s[i].mean(dim=0))
            else:
                prototypes_s.append(self.queue_s[i][:self.queue_ptr[i]].mean(dim=0))
                    
        prototypes_s = torch.stack(prototypes_s, dim=0)
        self.prototypes_s = prototypes_s
    
    def updata_prototypes(self):
        for i in range(self.num_classes):
            if self.queue_full[i] or self.queue_ptr[i] == 0:
                prototypes_s_avg = self.queue_s[i].mean(dim=0)
            else:
                prototypes_s_avg = self.queue_s[i][:self.queue_ptr[i]].mean(dim=0)
            
            alpha = 0.5
            # alpha = (F.cosine_similarity(self.prototypes_s[i], prototypes_s_avg, dim=0).item() + 1) / 2.0
            self.prototypes_s[i] = self.prototypes_s[i] * alpha + prototypes_s_avg * (1-alpha)

    @torch.no_grad()
    def predict_prototype(self, feature_pooled_s, gt_classes):

        predict_cosine_s = F.cosine_similarity(feature_pooled_s[:, None], self.prototypes_s[None, :], dim=-1)
        predict_classes_s = predict_cosine_s
        
        zeros = torch.zeros((len(predict_classes_s), 1), device=predict_classes_s.device)
        
        predict_classes_s = F.softmax(predict_classes_s*10, dim=-1)
        predict_classes_s = torch.cat([predict_classes_s, zeros], dim=1)
        
        return predict_classes_s
        
    def forward(self, images, features, proposals, targets=None, compute_loss=True, branch="", compute_val_loss=False):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training and compute_loss:  # apply if training loss
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets, branch=branch)
        elif compute_val_loss:  # apply if val loss
            assert targets
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        
        # 更新类原型
        if branch == "supervised_test":
            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            feature_pooled_s = box_features.mean(dim=[2, 3])
            self.update_memory(feature_pooled_s.detach(), gt_classes)

            self.updata_prototypes()
        
        if branch == "supervised_test":
            pred_class_logits, _ = predictions

            gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            # unknown_class_idx = num_classes + 1
            true_cases = (gt_classes == self.num_classes + 1)
            
            feature_pooled_s = box_features.mean(dim=[2, 3])

            gt_prototype_classes_s = self.predict_prototype(feature_pooled_s, gt_classes)
            gt_prototype_classes_s = gt_prototype_classes_s.detach()
        
            losses = self.box_predictor.losses(predictions, proposals, branch)
            if torch.all(true_cases.eq(False)):
                # no unknown class
                pass
            else:
                loss_kld = F.kl_div(F.log_softmax(pred_class_logits[true_cases], dim=1),
                    gt_prototype_classes_s[true_cases], reduction='batchmean')
                losses.update({'loss_kld': loss_kld * 0.1})
            losses['loss_cls'] = losses['loss_cls'] * 1.0
            losses['loss_box_reg'] = losses['loss_box_reg'] * 1.0
            return [], losses

        if (self.training and compute_loss) or compute_val_loss:
            del features
            losses = self.box_predictor.losses(predictions, proposals, branch)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
