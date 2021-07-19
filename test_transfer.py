from mmdet.models.detectors.centertrack import CenterTrack

import torch
# m_backbone = DLA(34)
# m_neck = DLANeck(34)
# m_head = CenterTrackHead(64, 256, 1)
# model = CenterTrack(m_backbone, m_neck, m_head)

backbone = dict(
    type='DLA',
    arch=34)
neck = dict(
    type='DLANeck',
    arch=34)
bbox_head = dict(
    type='CenterTrackHead',
    num_classes=1,
    in_channel=64,
    feat_channel=256,  # todo check
    loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
    loss_wh=dict(type='L1Loss', loss_weight=0.1),
    loss_offset=dict(type='L1Loss', loss_weight=1.0),
    loss_tracking=dict(type='L1Loss', loss_weight=1.0))

model = CenterTrack(backbone, neck, bbox_head)

nsd = torch.load('../mmtracking/new_model.pth')['state_dict']

# for k, v in nsd.items():
#     print(k, v.shape)
model.load_state_dict(nsd)
