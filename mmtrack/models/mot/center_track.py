from mmdet.core import bbox2result

from mmtrack.core import track2result
from ..builder import MODELS, build_tracker,build_detector
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
                    public_bboxes=None,
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
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)

        if hasattr(self.detector, 'bbox_head'):
            outs = self.detector.bbox_head(x)
            result_list = self.detector.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, rescale=rescale)
            # TODO: support batch inference
            det_bboxes = result_list[0][0]
            det_labels = result_list[0][1]
            ref_bboxes = result_list[0][2]
            num_classes = self.detector.bbox_head.num_classes
        else:
            raise TypeError('detector must has bbox_head.')

        bboxes, labels, ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            gt_bboxes=det_bboxes,
            ref_bboxes=ref_bboxes,
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
                                    gt_instance_ids,
                                    gt_match_indices,
                                    ref_img_metas,
                                    ref_img,
                                    ref_gt_bboxes,
                                    ref_gt_labels,
                                    ref_gt_match_indices,
                                    ref_gt_instance_ids)
