_base_ = [
    '../../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'MOTChallengeDataset'
img_norm_cfg = dict(
    mean=[104.01362, 114.034225, 119.916595], std=[73.60277, 69.89082, 70.91508], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCenterCropPad',
        share_params=True,
        crop_size=(544, 960),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None,
        bbox_clip_border=False),
    dict(
        type='SeqResize',
        img_scale=(544, 960),
        share_params=True,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(544, 960)],
        # scale_factor=1.0,
        flip=False,
        transforms=[
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                # test_pad_mode=['logical_or', 31],
                # test_pad_add_pix=1,
                test_pad_mode=['size_divisor', 32],
                bbox_clip_border=False),
            dict(type='Resize', keep_ratio=True, bbox_clip_border=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='VideoCollect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]
data_root = '../data/MOT17/'
# data_root = '/home/akio/data/MOT/MOT17-mini/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            visibility_thr=0.25,
            ann_file=data_root + 'annotations/half-train_cocoformat.json',
            img_prefix=data_root + 'train',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=2,
                filter_key_img=True,
                method='uniform'),
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/half-val_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=None,
        pipeline=test_pipeline))

model = dict(
    type='CenterTrack',
    pretrains=dict(
        detector='/home/akio/dev/mmtracking/new_model.pth'
        # detector='../new_model.pth'
    ),
    detector=dict(
        type='CTDetector',
        backbone=dict(
            type='DLA',
            arch=34),
        neck=dict(
            type='DLANeck',
            arch=34),
        bbox_head=dict(
            type='CenterTrackHead',
            num_classes=1,
            in_channel=64,
            feat_channel=256,
            loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
            loss_wh=dict(type='L1Loss', loss_weight=0.1),
            loss_offset=dict(type='L1Loss', loss_weight=1.0),
            loss_tracking=dict(type='L1Loss', loss_weight=1.0),
            loss_ltrb_amodal=dict(type='L1Loss', loss_weight=0.1)),
        test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100),
        train_cfg=dict(fp_disturb=0.1, lost_disturb=0.4, hm_disturb=0.05)
    ),
    tracker=dict(type='CTTracker')
)

# optimizer
optimizer = dict(_delete_=True, type='Adam', lr=1.25e-4)
# optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=1000,
    # warmup_ratio=1.0 / 1000,
    step=[60])  # the real step is [18*5, 24*5]

# runner = dict(type='EpochBasedRunner', max_epochs=28)  # the real epoch is 28*5=140

# runtime settings
total_epochs = 70
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# For distributed training
find_unused_parameters = True

# checkpoint
checkpoint_config = dict(_delete_=True, interval=10)
