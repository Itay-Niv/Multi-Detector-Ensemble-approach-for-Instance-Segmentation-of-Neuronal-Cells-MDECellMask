# SPECIFIY HERE YOUR DETAILS:
main_dir =  "YOUR_MAIN_DIR"
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
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) #dict(mean=[128, 128, 128], std=[11.58, 11.58, 11.58], to_rgb=True)
n_epochs = 50
checkpoint_interval = 1
crop_size = (128, 128) 
seed=0
####################


# Define directories:
model_name = "seg_upernet_swin"
model_path = f'{main_dir}/models/seg/{model_name}/'
data_dir =  f'{main_dir}data/'
livecell_data_dir = f'{data_dir}/LIVECell_dataset_2021/'   
ckpt_dir = model_path + 'ckpt/'
ckpt_path = ckpt_dir + exp_name 
log_dir = model_path + 'log/'
log_path = log_dir + exp_name                                               
work_dir = ckpt_path      



# pretrain_weights & resume from:
pretrain_weights = model_path + '/pretrained_weights/swin_base_patch4_window7_224_22k.pth'
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
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

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
##################
pred_file = ''


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

if livecell_or_sartorius == 'livecell':
        helper_dataset = dict(
            type='CocoDataset',
            classes=classes,
            ann_file=livecell_data_dir + 'ann_coco_livecell_val.json',
            img_prefix=livecell_data_dir + '/images',
            pipeline=[],
        )
        data = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=2,
            train=dict(
                type='GTBBoxDataset',
                img_prefix=livecell_data_dir + '/images',
                ann_file=[livecell_data_dir + 'ann_coco_livecell_train.json', livecell_data_dir + 'ann_coco_livecell_test.json'],
                helper_dataset=helper_dataset,
                pipeline=train_pipeline
            ),
            val=dict(
                type='GTBBoxDataset',
                img_prefix=livecell_data_dir + '/images',
                ann_file=livecell_data_dir + 'ann_coco_livecell_val.json',
                helper_dataset=helper_dataset,
                pipeline=test_pipeline
            ),
            test=dict(
                type='PredBBoxDataset',
                mask_rerank=True,
                pred_file=pred_file,
                img_prefix=livecell_data_dir + '/images',
                ann_file=livecell_data_dir + 'ann_coco_livecell_val.json',
                helper_dataset=helper_dataset,
                pipeline=test_pipeline
            )
        )

if livecell_or_sartorius == 'sartorius':
        helper_dataset = dict(
            type='CocoDataset',
            classes=classes,
            ann_file=data_dir + 'ann_coco_sartorius_val_95_5.json',
            img_prefix=data_dir,
            pipeline=[],
        )
        data = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=2,
            train=dict(
                type='GTBBoxDataset',
                img_dir=data_dir,
                ann_file=data_dir + 'ann_coco_sartorius_train_95_5.json',
                helper_dataset=helper_dataset,
                pipeline=train_pipeline
            ),
            val=dict(
                type='GTBBoxDataset',
                img_dir=data_dir,
                ann_file=data_dir + 'ann_coco_sartorius_val_95_5.json',
                helper_dataset=helper_dataset,
                pipeline=test_pipeline
            ),
            test=dict(
                type='PredBBoxDataset',
                mask_rerank=True,
                pred_file=pred_file,
                img_dir=data_dir,  
                ann_file=data_dir + 'ann_coco_sartorius_test.json', 
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
gpu_ids = range(1)