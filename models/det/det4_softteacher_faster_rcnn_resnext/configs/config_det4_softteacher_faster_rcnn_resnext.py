# SPECIFIY HERE YOUR DETAILS:
main_dir = "YOUR_MAIN_DIR"
exp_name = 'exp_0'  
wnb_username = 'wnb_username'
wnb_project_name = 'wnb_project_name-mmdet'


# Parameters:
batch_size = 2 
max_inst_per_img = 350 
classes = ['shsy5y','cort','astro']
num_classes = 3
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
n_epochs = 35
n_iters_per_epoch = 1214 #0.95*607 = 576, 607*2=1214, 607*3=1821
sample_ratio=[1,1]
img_scale = [(1333, 800), (1690, 960)] 
init_lr = 0.001 
steps_decrease = [8, 11] 
seed=0
####################



# Define directories:
model_name = "det4_softteacher_faster_rcnn_resnext"
model_path = f'{main_dir}/models/det/{model_name}/'
data_dir =  f'{main_dir}data/'
ckpt_dir = model_path + 'ckpt/'
ckpt_path = ckpt_dir + exp_name 
log_dir = model_path + 'log/'
log_path = log_dir + exp_name                                                       
work_dir = ckpt_path                               


# pretrain_weights & resume from:
pretrain_weights = model_path + '/pretrained_weights/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth'  #None   
resume_from_path = None  #ckpt_path + '/iter_14568.pth' 


###############
# model settings
model = dict(
    type='FasterRCNN',
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
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
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
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
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
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
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
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=max_inst_per_img) #100
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
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
     dict(type='Resize', img_scale= img_scale, keep_ratio=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Albu',
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
	dict(type="ExtraAttrs", tag="sup"), # added for softteacher
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'), 
    dict(type='Collect',
	keys=['img', 'gt_bboxes', 'gt_labels'], # DetOnly: removed:, gt_masks), 
	meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag'))
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
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


strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale= img_scale, 
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],  # DetOnly: removed:'gt_masks'
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=img_scale, 
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",    
        keys=["img", "gt_bboxes", "gt_labels"], # DetOnly: removed:'gt_masks'
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),  #removed (unsupervised)
    
    # generate fake labels for data format compatibility
    dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]




data = dict(
    samples_per_gpu = batch_size,
    workers_per_gpu = 2,
    train=dict(
        type="SemiDataset",
        sup=dict(
            type="CocoDataset",
            ann_file= data_dir + 'ann_coco_sartorius_train_95_5.json',
            img_prefix=f"{data_dir}", 
            pipeline=train_pipeline,
			classes=classes,
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file= data_dir + 'ann_coco_semi.json',
            img_prefix=f"{data_dir}train_semi_supervised",
            pipeline=unsup_pipeline,
            filter_empty_gt=False,
			classes=classes,
        ),
    ),
    val=dict(
        type='CocoDataset',
        ann_file=data_dir + 'ann_coco_sartorius_val_95_5.json',
        img_prefix=f"{data_dir}", 
        pipeline=val_pipeline,
		classes=classes,
        ),
	test=dict(
        type='CocoDataset',
        ann_file=data_dir + 'ann_coco_sartorius_test.json',  
        img_prefix=f"{data_dir}", 
        pipeline=val_pipeline,
		classes=classes,
        ),
    sampler=dict(
        train=dict(
            type="SemiBalanceSampler",
            sample_ratio= sample_ratio,
            by_prob=False, #suggested in official softteacher's git forum to switch to False
            # at_least_one=True,
            epoch_length=n_iters_per_epoch, 
        )
    ),
)


semi_wrapper = dict(
    type="SoftTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=2.0, #4.0
    ),
    test_cfg=dict(inference_on="student"),
)


custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]


cudnn_benchmark = True


optimizer = dict(type="SGD", lr=init_lr, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
	policy='step',
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=0.001,
	step= [n_iters_per_epoch*steps_decrease[0], n_iters_per_epoch*steps_decrease[1]]) 
fp16 = dict(loss_scale="dynamic")
runner = dict(type="IterBasedRunner", max_iters=n_iters_per_epoch*n_epochs) 
checkpoint_config = dict(by_epoch=False, interval=n_iters_per_epoch*2, max_keep_ckpts=20) 
evaluation = dict(type="SubModulesDistEvalHook", interval=n_iters_per_epoch) 
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
            by_epoch=False,
        ),
    ],
)


load_from = pretrain_weights    
resume_from = resume_from_path 


dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
gpu_ids = range(1)
