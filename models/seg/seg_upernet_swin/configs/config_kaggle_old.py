# fixed parameters:
batch_size = 2 
#max_inst_per_img = 350 #300 # [200 , 250 , 300 , 350 , 400 , 500]
classes = ['shsy5y','cort','astro']
dataset_type = 'CocoDataset'
#img_norm_cfg = dict(mean=[128, 128, 128], std=[11.58, 11.58, 11.58], to_rgb=True)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (128, 128) 



###### hyperparameters to grid search #########
n_epochs = 10
#n_iters_per_epoch = 1214 #0.95*607 = 576, 607*2=1214, 607*3=1821
checkpoint_interval = 1 #n_epochs 
#img_scale = [(1333, 800), (1690, 960)]
#init_lr = 0.001 #0.001 #0.01 # [0.001 , 0.0025 , 0.005 , 0.01 , 0.015 , 0.2]                  ****
#lr_policy = 'step'  # ['CosineAnnealing','poly','step']
#steps_decrease = [8, 11] # [[16, 19], [8, 11]] - can think of more # ONLY FOR STEP POLICY:
####################


# EXP NAME:
exp_name = 'upernet_swin2'     # Specify here backbone+ any additional info about the experiment
model_name = "SegOnly"
colab_or_kaggle = 'colab'          # 'kaggle'  #'colab'                      
livecell_or_sartorius = 'sartorius'  # 'sartorius' #'livecell'
sagi_or_itay = 'sagi'               # 'itay'  # 'sagi'
title = sagi_or_itay + '_' + colab_or_kaggle + '_'+ livecell_or_sartorius + '_'+ exp_name
wnb_username = 'sagi_itay'
wnb_project_name = 'sartorius-mmdet'


# Config Modification
main_dir = "/content/drive/MyDrive/Deep_Learning_Itay_Sagi/Project/Sartorius_Cell_Instance_Segmentation/"
data_dir = f'{main_dir}data/'
ckpt_dir = f'{main_dir}{model_name}/ckpt/'
ckpt_path = ckpt_dir + title #model_name + '/' + title
#log_dir = f'{main_dir}{model_name}/log/'
#log_path = log_dir + model_name + '/' + title                           
data_root = f'{main_dir}{model_name}/ckpt/'                                   
work_dir = ckpt_path                               


# pretrain_weights & resume from:
pretrain_weights = main_dir + model_name + '/pretrained_weights/swin_base_patch4_window7_224_22k.pth'  #None 
resume_from_path = None #main_dir + model_name +'/ckpt/'+ title + '/iter_4856.pth' #None #   #None
seed=0

norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

################
# model settings
model = dict(
    type='CustomEncoderDecoder',
	#type='EncoderDecoder',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

dataset_type = 'GTBBoxDataset'

train_pipeline = [
    dict(type='BoxJitter', prob=0.5),
    dict(type='ROIAlign', output_size=crop_size),
    dict(type='FlipRotate'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='ROIAlign', output_size=crop_size),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='BBoxFormat'),
            dict(type='Collect', keys=['img', 'bbox'])
        ]
    )
]
helper_dataset = dict(
    type='CocoDataset',
    classes=classes,
    ann_file=data_dir + 'sartorius_coco_val_95_5.json',
    img_prefix=data_dir + 'train',
    pipeline=[],
)
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_dir=data_dir + 'train',
        ann_file=data_dir + 'sartorius_coco_train_95_5.json',
        helper_dataset=helper_dataset,
        pipeline=train_pipeline
    ),
    val=dict(
        type='PredBBoxDataset',
        score_thr=0.3,
        pred_file=main_dir + 'SoftTeacher_official/ckpt/itay_colab_sartorius_R101_faster_rcnn_sample11_niter1214_unsupw2/output/tested_val_95_5.pkl',
        img_dir=data_dir + 'train',
        ann_file=data_dir + 'sartorius_coco_val_95_5.json',
        helper_dataset=helper_dataset,
        pipeline=test_pipeline
    ),
    test=dict(
        type='PredBBoxDataset',
        mask_rerank=True,
        pred_file=main_dir + 'SoftTeacher_official/ckpt/itay_colab_sartorius_R101_faster_rcnn_sample11_niter1214_unsupw2/output/tested_val_95_5.pkl',
        img_dir=data_dir + 'train',  
        ann_file=data_dir + 'sartorius_coco_val_95_5.json', 
        helper_dataset=helper_dataset,
        pipeline=test_pipeline
    )
)

log_config = dict(
    interval=1000,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project=wnb_project_name, #"pre_release",
                name=f'{model_name}-exp_name{title}', #"${cfg_name}",
                entity=wnb_username,
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)


dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = pretrain_weights    
resume_from = resume_from_path 

fp16 = dict(loss_scale="dynamic")
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05 / 16,
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
evaluation = dict(interval=1, metric='dummy', pre_eval=True)