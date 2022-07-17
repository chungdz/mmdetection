# setup env
export MMDET_DATASETS=/mnt/coco/
export OMP_NUM_THREADS=8
python -m exps.make_smaller_coco --image_counts=5000

# Test on existing model and checkpoint
# refer for the website https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md
# weightes are downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn and save into checkpoints
python tools/test.py\
 configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
 checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
 --show \
 --work-dir=results \
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
                    data.train.ann_file='/mnt/coco/annotations/instances_train2017.small.json' \
    --work-dir=cps
# run all
python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=12 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=12 \
                    log_config.interval=300 \
                    runner.max_epochs=2 \
    --work-dir=cps

# run swin
# Test on existing model and checkpoint
# refer for the website https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md
# weightes are downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn and save into checkpoints
python tools/test.py\
 configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
 checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
 --show \
 --work-dir=results \
 --eval bbox segm

./tools/dist_test.sh\
 configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
 checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth \
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
                    data.train.ann_file='/mnt/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='checkpoints/swin_small_patch4_window7_224.pth'\
                    evaluation.interval=1 \
    --work-dir=cps

./tools/dist_train.sh \
    configs/swin/mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    6 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=18 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=50 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/mnt/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
    --work-dir=cps


# train on smaller files and swin base for better debug
python tools/train.py \
    configs/swin/mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=3 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=100 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/mnt/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
    --work-dir=cps

python tools/train.py \
    configs/swin/mask_rcnn_swin-b-p4-w7_fp16_ms-crop-3x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=3 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=100 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/mnt/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='/mnt/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
                    lr_config.step="[2,2]" \
                    lr_config.warmup_iters=10 \
    --work-dir=/mnt/cps_detect

python tools/train.py \
    configs/swin/mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco_gem.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=3 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=3 \
                    log_config.interval=5 \
                    runner.max_epochs=2 \
                    data.train.ann_file='/mnt/coco/annotations/instances_train2017.small.json' \
                    model.backbone.init_cfg.checkpoint='/mnt/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
                    lr_config.step="[2,2]" \
                    lr_config.warmup_iters=10 \
    --work-dir=/mnt/cps_detect

