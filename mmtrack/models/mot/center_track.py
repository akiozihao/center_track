from mmdet.core import bbox2result

from mmtrack.core import track2result
from ..builder import MODELS, build_tracker, build_detector
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class CenterTrack(BaseMultiObjectTracker):
    def __init__(self,
                 detector=None,
                 tracker=None,
                 pretrains=None):
        super(CenterTrack, self).__init__()
        if detector is not None:
            self.detector = build_detector(detector)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.init_weights(pretrains)
        # self.init_module('detector', pretrain.get('detector', False))  # todo

    def init_weights(self, pretrain):
        """Initialize the weights of the modules.

        Args:
            pretrained (dict): Path to pre-trained weights.
        """
        if pretrain is None:
            pretrain = dict()
        assert isinstance(pretrain, dict), '`pretrain` must be a dict.'
        if self.with_detector and pretrain.get('detector', False):
            self.init_module('detector', pretrain['detector'])

    def simple_test(self,
                    img,
                    img_metas,
                    rescale=False,
                    public_bboxes=None,  # todo check
                    **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.
            public_bboxes (list[Tensor], optional): Public bounding boxes from
                the benchmark. Defaults to None.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        if not self.training:
            self.detector.head.fp_disturb = .0
            self.detector.head.lost_disturb = .0
            self.detector.head.hm_disturb = .0

        frame_id = img_metas[0]['frame_id']
        if frame_id == 0:
            self.tracker.reset()
            self.ref_bboxes = None
            n, c, h, w = img.shape
            self.ref_hm = torch.zeros((n, 1, h, w), dtype=img.dtype, device=img.device)
            self.ref_img = img.clone()
            self.pre_labels = None
        else:
            self.ref_hm = self.detector._build_test_hm(self.ref_img, self.ref_bboxes)
        # todo check this
        batch_input_shape = tuple(img[0].size()[-2:])
        img_metas[0]['batch_input_shape'] = batch_input_shape
        x = self.detector.extract_feat(img, self.ref_img, self.ref_hm)
        center_heatmap_pred, wh_pred, offset_pred, tracking_pred, ltrb_amodal_pred = self.detector.bbox_head(x)
        outs = [center_heatmap_pred, wh_pred, offset_pred, tracking_pred]
        result_list = self.detector.bbox_head.get_bboxes(
            # todo Are outs always tensors?
            *[[tensor] for tensor in outs], img_metas=img_metas, rescale=rescale)
        # TODO: support batch inference
        det_bboxes = result_list[0][0]
        det_labels = result_list[0][1]
        gt_bboxes_with_motion = result_list[0][2]
        num_classes = self.detector.bbox_head.num_classes
        self.ref_img = img
        self.ref_bboxes = det_bboxes
        self.ref_labels = det_labels

        bboxes, labels, ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            bboxes=det_bboxes,
            bboxes_with_motion=gt_bboxes_with_motion,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)

        track_result = track2result(bboxes, labels, ids, num_classes)
        bbox_result = bbox2result(det_bboxes, det_labels, num_classes)
        return dict(bbox_results=bbox_result, track_results=track_result)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_instance_ids,
                      gt_match_indices,
                      ref_img_metas,
                      ref_img,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_match_indices,
                      ref_gt_instance_ids):
        return self.detector.forward_train(img,
                                           img_metas,
                                           gt_bboxes,
                                           gt_labels,
                                           gt_match_indices,
                                           ref_img_metas,
                                           ref_img,
                                           ref_gt_bboxes,
                                           ref_gt_labels,
                                           ref_gt_match_indices)
