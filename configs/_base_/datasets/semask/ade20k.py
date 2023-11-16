# dataset settings
# dataset_type = 'ADE20KDataset'
# data_root = '/home/datasets/ade20k/ADEChallengeData2016'
dataset_type = 'cbctdataset'
#data_root = '/rsch/jiaxiang/mmseg-copy/inbatch_tp_fn_fp/data/cbct_coco_init_all'
#data_root = '/mnt/ECE445/linjietong/SeMask-FPN/data/160CBCT_3'
#data_root = '/mnt/ECE445/linjietong/SeMask-FPN/data/Dental3'
#data_root = '/mnt/linjie/SeMask-FPN/data/MedicalTransformerCBCT160/'
#data_root = '/rsch/linjie/SeMask-FPN/data/data_460/'
data_root = '/mnt/ECE445/linjietong/SeMask-FPN/frame/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    #import pdb; pdb.set_trace()
    #dict(type='Resize', img_scale=(2048, 2048), ratio_range=[1]),
    dict(type='Resize', img_scale=(2048, 2048), ratio_range=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], cut_ratio=[0.125,0.25,0.375,0.5,0.625,0.75,0.875],cutmix=0 ,con=False, cutmix_orignal=False),
    
    #dict(type='Resize', img_scale=(2048, 2048), ratio_range=[1.0]),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1,ratio_range=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],mode='random'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True,cutmix=0,con=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
#import pdb; pdb.set_trace()
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/annotations',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/annotations',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        #data/data_460/images/validation1/
        #img_dir='images/validation1',
        #ann_dir='annotations/validation1',
        img_dir='test',
        ann_dir='test/annotations',
        pipeline=test_pipeline))
