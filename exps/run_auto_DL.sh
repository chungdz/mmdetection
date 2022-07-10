# setup env
export MMDET_DATASETS=/root/autodl-tmp/coco/
export OMP_NUM_THREADS=8
python -m exps.make_smaller_coco --image_counts=20000 --dpath=/root/autodl-tmp/coco

# Test on existing model and checkpoint
# refer for the website https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md
# weightes are downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn and save into checkpoints
python tools/test.py\
 configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
 /root/autodl-tmp/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
 --show \
 --work-dir=/root/autodl-tmp/results \
 --eval="bbox" \
# train on smaller files for better debug
python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=12 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=12 \
                    log_config.interval=100 \
                    runner.max_epochs=3 \
                    data.train.ann_file='/root/autodl-tmp/coco/annotations/instances_train2017.small.json' \
    --work-dir=/root/autodl-tmp/cps
# run all
python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=12 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=12 \
                    log_config.interval=300 \
                    runner.max_epochs=2 \
    --work-dir=/root/autodl-tmp/cps

# run swin
# Test on existing model and checkpoint
# refer for the website https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md
# weightes are downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn and save into checkpoints
python tools/test.py\
 configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
 /root/autodl-tmp/checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
 --show \
 --work-dir=results \
 --eval bbox segm

./tools/dist_test.sh\
 configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
 /root/autodl-tmp/checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
 6 \
 --work-dir=results \
 --eval bbox segm

# train on smaller files and smaller swin for better debug
# download https://github.com/SwinTransformer/storage/releases/download/v1.0.0/checkpoints/swin_small_patch4_window7_224.pth
# into checkpoints
python tools/train.py \
    configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=4 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=4 \
                    log_config.interval=100 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/root/autodl-tmp/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_small_patch4_window7_224.pth'\
                    evaluation.interval=1 \
    --work-dir=/root/autodl-tmp/cps

./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    6 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=18 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=50 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/root/autodl-tmp/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
    --work-dir=/root/autodl-tmp/cps


# train on smaller files and swin base for better debug
python tools/train.py \
    configs/swin/mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=4 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=4 \
                    log_config.interval=100 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/root/autodl-tmp/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
    --work-dir=/root/autodl-tmp/cps

# train on whole file
# --cfg-options resume_from
./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    6 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=18 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=1000 \
                    runner.max_epochs=20 \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
    --work-dir=/root/autodl-tmp/cps

./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    6 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=18 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=1000 \
                    runner.max_epochs=5 \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    load_from='/root/autodl-tmp/cps/second_swin_b_rcnn.pth' \
                    evaluation.interval=1 \
                    lr_config.step="[4,5]" \
                    lr_config.warmup_iters=1 \
    --work-dir=/root/autodl-tmp/cps

./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    6 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=18 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=1000 \
                    runner.max_epochs=5 \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_small_patch4_window7_224.pth'\
                    load_from='/root/autodl-tmp/cps/swin_small_mask_rcnn.0704.ep15.pth' \
                    evaluation.interval=1 \
                    lr_config.step="[4,5]" \
                    lr_config.warmup_iters=1 \
    --work-dir=/root/autodl-tmp/cps

./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-b-p4-w7_fp16_ms-crop-3x_coco.py \
    6 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=18 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=1000 \
                    runner.max_epochs=22 \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
                    lr_config.step="[19,22]" \
                    lr_config.warmup_iters=1 \
    --work-dir=/root/autodl-tmp/cps

## A40 4 on small dataset
./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    4 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=36 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=9 \
                    log_config.interval=50 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/root/autodl-tmp/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_small_patch4_window7_224.pth'\
                    evaluation.interval=1 \
                    optimizer.lr=0.00014 \
    --work-dir=/root/autodl-tmp/cps


## A40 4 GPU Swin small test
./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    4 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=36 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=9 \
                    log_config.interval=500 \
                    runner.max_epochs=15 \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_small_patch4_window7_224.pth'\
                    evaluation.interval=1 \
                    optimizer.lr=0.00014 \
                    lr_config.step="[11,15]" \
                    lr_config.warmup_iters=250 \
    --work-dir=/root/autodl-tmp/cps


    
# no fpn

./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-b-p4-w7_fp16_ms-crop-3x_coco.py \
    6 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=18 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=1000 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/root/autodl-tmp/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
                    lr_config.step="[19,22]" \
                    lr_config.warmup_iters=1 \
    --work-dir=/root/autodl-tmp/cps


