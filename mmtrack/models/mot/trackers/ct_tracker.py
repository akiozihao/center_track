from mmtrack.models import TRACKERS

from .base_tracker import BaseTracker
import torch


@TRACKERS.register_module()
class CTTracker(BaseTracker):
    def __init__(self, **kwargs):
        super(CTTracker, self).__init__(**kwargs)

    def track(self,
              img,
              img_metas,
              gt_bboxes,
              ref_bboxes,
              labels,
              frame_id,
              rescale,
              **kwargs):
        assert gt_bboxes.shape == ref_bboxes.shape
        if self.empty or ref_bboxes.size(0) == 0:
            num_new_tracks = ref_bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            ids = torch.full((ref_bboxes.size(0),), -1, dtype=torch.long)

            for idx, ref_bbox in enumerate(ref_bboxes):
                threshold = 0.0  # todo need to check
                det_center = self._xyxy2center(ref_bbox)
                ref_center = self._xyxy2center(ref_bbox)  # shape (1,2)
                assert det_center.ndim == 2 and det_center.shape[0] == 1
                assert ref_center.ndim == 2 and ref_center.shape[0] == 1
                closest_id = -1
                closest_distance = float('inf')
                for id in self.tracks.keys():
                    pre_gt_center = self.tracks[id].det_center[-1]
                    dis = self._cal_dist(pre_gt_center, ref_center)
                    if dis < threshold and dis < closest_distance:
                        closest_id = id
                        closest_distance = dis
                ids[idx] = closest_id

            new_track_inds = ids == -1

            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()

            self.update(
                ids=ids,
                gt_bboxes=gt_bboxes[:, :4],
                ref_bboxes=ref_bboxes[:, :4],
                scores=ref_bboxes[:, -1],
                labels=labels,
                frame_ids=frame_id)
        return gt_bboxes, labels, ids

    def _xyxy2center(self, bbox):  # shape ? (1,4) or (1,3)
        # try (1,3)
        ctx = bbox[0][0] + (bbox[0][2] - bbox[0][0]) / 2
        cty = bbox[0][1] + (bbox[0][3] - bbox[0][1]) / 2
        return ctx, cty
