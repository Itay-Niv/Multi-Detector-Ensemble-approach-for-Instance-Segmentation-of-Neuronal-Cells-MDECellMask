# SPECIFIY HERE YOUR DETAILS:
main_dir = "YOUR_MAIN_DIR"
exp_name = 'exp_0'  
wnb_username = 'wnb_username'
wnb_project_name = 'wnb_project_name-mmdet'

# Choose dataset:
# 'livecell'  => Pretrain on livecell dataset
#  or
# 'sartorius' => Finetune on competition dataset
livecell_or_sartorius = 'sartorius'

# fixed parameters:
batch_size = 2 
max_inst_per_img = 350 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #dict(mean=[128, 128, 128], std=[11.58, 11.58, 11.58], to_rgb=True)
n_epochs = 50
checkpoint_interval = 1 
init_lr = 0.0001 
steps_decrease = [8, 11]
seed=0
####################


# Define directories:
model_name = "det1_cascade_rcnn_resnext"
model_path = f'{main_dir}/models/det/{model_name}/'
data_dir =  f'{main_dir}data/'
livecell_data_dir = f'{data_dir}/LIVECell_dataset_2021/'   
ckpt_dir = model_path + 'ckpt/'
ckpt_path = ckpt_dir + exp_name 
log_dir = model_path + 'log/'
log_path = log_dir + exp_name                                               
work_dir = ckpt_path      



# pretrain_weights & resume from:
pretrain_weights = model_path + '/pretrained_weights/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
resume_from_path =  None #ckpt_path + '/epoch_40.pth'


if livecell_or_sartorius == 'livecell':                                         
    classes = ['a172','bt474','bv2','huh7','mcf7','shsy5y','skbr3','skov3']
    num_classes = 8
    img_scale = (520, 704)  #[(620, 839), (520, 704)]  #(620, 839)
if livecell_or_sartorius == 'sartorius':
    classes = ['shsy5y','cort','astro']
    num_classes = 3
    img_scale = [(1333, 800), (1690, 960)]

###############
# model settings
model = dict(
    type='CascadeRCNN',
        backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
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
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=max_inst_per_img)))
##################


# transformations and pipelines:

albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625,
         scale_limit=0.15, rotate_limit=15, p=0.4),
    dict(type='RandomBrightnessContrast', brightness_limit=0.2,
         contrast_limit=0.2, p=0.5),
    dict(
        type="OneOf",
        transforms=[
            dict(type="GaussianBlur", p=1.0, blur_limit=7),
            dict(type="MedianBlur", p=1.0, blur_limit=7),
        ],
        p=0.4,
    ),
]



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False), #DetOnly
    dict(type='Resize', img_scale= img_scale),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),  # DetOnly: removed:, gt_masks='masks'), 
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'), 
    dict(
	type='Collect',
	keys=['img', 'gt_bboxes', 'gt_labels']) # DetOnly: removed:, gt_masks), 
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale = img_scale, 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type="Normalize", **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


###################################################
if livecell_or_sartorius == 'livecell':
        data = dict(
            samples_per_gpu = batch_size,
            workers_per_gpu = 2,
            train=dict(
                type='CocoDataset',
                ann_file=[livecell_data_dir + 'ann_coco_livecell_train.json', livecell_data_dir + 'ann_coco_livecell_test.json'],
                img_prefix=livecell_data_dir + '/images',
                pipeline=train_pipeline,
                classes=classes,
                ),
            val=dict(
                type='CocoDataset',
                ann_file=livecell_data_dir + 'ann_coco_livecell_val.json',
                img_prefix=livecell_data_dir + '/images',
                pipeline=val_pipeline,
                classes=classes,
                ),
            test=dict(
                type='CocoDataset',
                ann_file=livecell_data_dir + 'ann_coco_livecell_val.json',
                img_prefix=livecell_data_dir + '/images',
                pipeline=val_pipeline,
                classes=classes,
                ),
                )
        
if livecell_or_sartorius == 'sartorius':
        data = dict(
            samples_per_gpu = batch_size,
            workers_per_gpu = 2,
            train=dict(
                type='CocoDataset',
                ann_file=data_dir + 'ann_coco_sartorius_train_95_5.json',
                img_prefix=data_dir,
                pipeline=train_pipeline,
                classes=classes,
                ),
            val=dict(
                type='CocoDataset',
                ann_file=data_dir + 'ann_coco_sartorius_val_95_5.json',
                img_prefix=data_dir,
                pipeline=val_pipeline,
                classes=classes,
                ),
            test=dict(
                type='CocoDataset',
                ann_file=data_dir + 'ann_coco_sartorius_test.json',
                img_prefix=data_dir,
                pipeline=val_pipeline,
                classes=classes,
                ),
                )





custom_hooks = [dict(type='NumClassCheckHook')]
fp16 = dict(loss_scale="dynamic") 

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project=wnb_project_name, 
                name=f'{model_name}-exp_name{exp_name}', 
                entity=wnb_username,
                config=dict(
                    work_dirs="${work_dir}",
                    #total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=True,
        ),
    ],
)

cudnn_benchmark = True

optimizer = dict(
    type='AdamW',
    lr= init_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)
        )
    )
)
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5
)
runner = dict(type='EpochBasedRunner', max_epochs=n_epochs)
checkpoint_config = dict(interval=checkpoint_interval, save_optimizer=False)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')

load_from = pretrain_weights    
resume_from = resume_from_path 


dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
gpu_ids = range(1)
meta = dict()
