HF_ENV_NAME=py38-202111 hfai bash tools/dist_train.sh \
    configs/swin/cascade_mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    8 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=16 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=2 \
                    log_config.interval=1000 \
                    runner.max_epochs=22 \
                    evaluation.interval=1 \
                    lr_config.step="[20,22]" \
                    lr_config.warmup_iters=1 \
    --work-dir=/root/autodl-tmp/cps \
    --auto-resume \
    -- --nodes 1 --environments MMDET_DATASETS=/public_dataset/1/COCO/ --environments OMP_NUM_THREADS=8 


data.train.ann_file='/root/autodl-tmp/coco/annotations/instances_train2017.small.json' \
model.backbone.init_cfg.checkpoint='/root/autodl-tmp/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    