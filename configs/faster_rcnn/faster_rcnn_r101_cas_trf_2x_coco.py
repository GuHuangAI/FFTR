_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
     '../_base_/default_runtime.py'
]
model = dict(
    pretrained='torchvision://resnet101', 
    backbone=dict(depth=101),
    neck=dict(
        type='Cascade_TRF3',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        d_models=[128, 256, 512],
        dim_feedforwards=[384, 768, 1536],
        n_head=8,
        patch_sizes=[[12,12], [24,24], [6,6], [12,12], [3,3], [6,6], [3,3], [3,3]],
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        out_type='add',
        overlap=False,
        cascade_num=1,
        pos_type='sin',
        relu_before_extra_convs=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        adapt_size1=(12,12),
        adapt_size2=(12,12)
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        reg_decoded_bbox=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            #nms=dict(type='nms', iou_threshold=0.7),
            nms=dict(type='soft_nms', iou_threshold=0.7, min_score=0.05),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            #nms=dict(type='nms', iou_threshold=0.5),
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    )
        )
        
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', 
    #    img_scale=(1333, 800), 
    #    keep_ratio=True),
    dict(type='AutoAugment',
          policies=[
              [
                  dict(type='Resize',
                       img_scale=[(640, 1333), (672, 1333), (704, 1333),
                                  (736, 1333), (768, 1333), (800, 1333)],
                       multiscale_mode='value',
                       keep_ratio=True)
              ],
              [
                  dict(type='Resize',
                       img_scale=[(500, 1333), (600, 1333), (700, 1333)],
                       multiscale_mode='value',
                       keep_ratio=True),
                  dict(type='RandomCrop',
                       crop_type='absolute_range',
                       crop_size=(448, 700),
                       allow_negative_crop=True),
                  dict(type='Resize',
                       img_scale=[(640, 1333), (672, 1333), (704, 1333), 
                                  (736, 1333), (768, 1333), (800, 1333)],
                       multiscale_mode='value',
                       override=True,
                       keep_ratio=True)
              ]
          ]),
          
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1333, 800),
        img_scale=[(1333, 640), (1333, 800), (1333, 960)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

optimizer = dict(
     type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001,
                paramwise_cfg=dict(custom_keys={'bias':dict(lr_mult=2., decay_mult=0.),
                                              'norm':dict(decay_mult=0.)}
                                  ))

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 23])
runner = dict(type='EpochBasedRunner', max_epochs=26)

# resume_from = '/home/huang/code/mmdetection/work_dirs/faster_rcnn_r101_cas_trf3_trick_2x_coco/epoch_23.pth'