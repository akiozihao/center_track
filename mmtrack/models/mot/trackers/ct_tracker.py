import numpy as np

from mmtrack.models import TRACKERS

from .base_tracker import BaseTracker
import torch


@TRACKERS.register_module()
class CTTracker(BaseTracker):
    def __init__(self,
                 obj_score_thr=0.3,
                 **kwargs):
        super(CTTracker, self).__init__(**kwargs)
        self.pre_img = None
        self.pre_centers = None
        self.pre_center2id = {}

    def reset(self):
        super(CTTracker, self).reset()
        self.pre_img = None
        self.pre_centers = None

    def track(self,
              img,  # todo
              img_metas,  # todo
              bboxes,
              bboxes_with_motion,
              labels,
              frame_id,
              rescale,  # todo
              **kwargs):
        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
            self.pre_img = img
            self.pre_centers = self._xyxy2center(bboxes)
        else:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

            det_centers = self._xyxy2center(bboxes_with_motion)
            dist = torch.cdist(det_centers, self.pre_centers, 2)  # todo ximi cheack : diff with origin
            matched_indices = self._greedy_assignment(dist)
            for i in range(matched_indices.shape[0]):
                ids[matched_indices[i][0]] = matched_indices[i][1]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()
            self.pre_img = img
            self.pre_centers = det_centers

        self.update(
            ids=ids,
            bboxes=bboxes,
            labels=labels,
            frame_ids=frame_id)
        return bboxes, labels, ids

    def _xyxy2center(self, bbox):  # shape (N,5)
        ctx = bbox[:, 0] + (bbox[:, 2] - bbox[:, 0]) / 2
        cty = bbox[:, 1] + (bbox[:, 3] - bbox[:, 1]) / 2
        return torch.cat((ctx.reshape(-1, 1), cty.reshape(-1, 1)), 1)

    def _greedy_assignment(self, dist):
        dist = dist.cpu().numpy()
        matched_indices = []
        if dist.shape[1] == 0:
            return np.array(matched_indices, np.int32).reshape(-1, 2)
        for i in range(dist.shape[0]):
            j = dist[i].argmin()
            if dist[i][j] < 1e16:
                dist[:, j] = 1e18
                matched_indices.append([i, j])
        return np.array(matched_indices, np.int32).reshape(-1, 2)
