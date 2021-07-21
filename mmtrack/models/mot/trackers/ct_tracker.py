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
              bboxes,
              bboxes_with_motion,
              labels,
              frame_id,
              rescale,
              **kwargs):
        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            det_ctx, det_cty = self._xyxy2center(bboxes_with_motion)
            threshold = 0.0  # todo check
            closest_id = -1
            closest_distance = float('inf')
            for gt_bboxes_id in range(det_ctx):
                for id, obj in self.tracks.items():
                    ref_ctx, ref_cty = self._xyxy2center(obj['bboxes'][-1])
                    dis = self._cal_dist(det_ctx, det_cty, ref_ctx, ref_cty)
                    if dis < threshold and dis < closest_distance:
                        closest_id = id
                        closest_distance = dis
                    ids[gt_bboxes_id] = closest_id

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()

            self.update(
                ids=ids,
                bboxes=bboxes,
                labels=labels,
                frame_ids=frame_id)
        return bboxes, labels, ids

    def _cal_dist(self, det_ctx, det_cty, ref_ctx, ref_cty):
        # todo check this  L2 or L1
        # use L2
        return torch.sqrt(torch.pow((det_ctx - ref_ctx), 2) + torch.pow((det_cty - ref_cty), 2))

    def _xyxy2center(self, bbox):  # todo shape ? (N,4)
        ctx = bbox[:, 0] + (bbox[:, 2] - bbox[:, 0]) / 2
        cty = bbox[:, 1] + (bbox[:, 3] - bbox[:, 1]) / 2
        return ctx, cty

