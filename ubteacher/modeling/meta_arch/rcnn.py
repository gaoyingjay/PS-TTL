# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results


import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from ubteacher.modeling.meta_arch.gdl import decouple_layer, AffineLayer


@META_ARCH_REGISTRY.register()
class DefrcnTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.cfg = cfg
        self.affine_rpn = AffineLayer(num_channels=self.backbone.output_shape()['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self.backbone.output_shape()['res4'].channels, bias=True)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised_few" or branch == "supervised_test":
            # Region proposal network
            features_de_rpn = features
            if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
            
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features_de_rpn, gt_instances
            )
            
            # # roi_head lower branch
            features_de_rcnn = features
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
          
            _, detector_losses = self.roi_heads(
                images, features_de_rcnn, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            features_de_rpn = features
            if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
            
            proposals_rpn, _ = self.proposal_generator(
                images, features_de_rpn, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            features_de_rcnn = features
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
            
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features_de_rcnn,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            features_de_rpn = features
            if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
            
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features_de_rpn, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            features_de_rcnn = features
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
            
            _, detector_losses = self.roi_heads(
                images,
                features_de_rcnn,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
    
    def init_prototypes(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        # # roi_head lower branch
        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        
        self.roi_heads.init_prototypes(features_de_rcnn, gt_instances)

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                features_de_rpn = features
                if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
                    scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
                    features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
                proposals, _ = self.proposal_generator(images, features_de_rpn, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            features_de_rcnn = features
            if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
                scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
                features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
            results, _ = self.roi_heads(images, features_de_rcnn, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results
