hfai workspace push --force

HF_ENV_NAME=py38-202111 hfai bash tools/dist_train_hfai.sh \
    /weka-jd/prod/public/permanent/group_huyunfan/huyunfan/workspaces/detect/mmdetection/configs/swin/cascade_mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    8 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=16 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=2 \
                    log_config.interval=1000 \
                    runner.max_epochs=36 \
                    model.backbone.init_cfg.checkpoint='/ceph-jd/pub/jupyter/huyunfan/notebooks/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
                    lr_config.step="[20,36]" \
                    lr_config.warmup_iters=1 \
    --work-dir=/ceph-jd/pub/jupyter/huyunfan/notebooks/cps/ \
    --auto-resume \
    -- --nodes 1 --environments MMDET_DATASETS=/public_dataset/1/COCO/ --environments OMP_NUM_THREADS=8 

HF_ENV_NAME=py38-202111 hfai bash tools/dist_train_hfai.sh \
    /weka-jd/prod/public/permanent/group_huyunfan/huyunfan/workspaces/detect/mmdetection/configs/swin/cascade_mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco_gem.py \
    8 \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=16 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=2 \
                    log_config.interval=1000 \
                    runner.max_epochs=36 \
                    model.backbone.init_cfg.checkpoint='/ceph-jd/pub/jupyter/huyunfan/notebooks/checkpoints/swin_base_patch4_window7_224_22k.pth'\
                    evaluation.interval=1 \
                    lr_config.step="[20,36]" \
                    lr_config.warmup_iters=1 \
    --work-dir=/ceph-jd/pub/jupyter/huyunfan/notebooks/cps/ \
    --auto-resume \
    -- --nodes 1 --environments MMDET_DATASETS=/public_dataset/1/COCO/ --environments OMP_NUM_THREADS=8 



                    