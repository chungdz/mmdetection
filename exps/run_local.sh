# setup env
export MMDET_DATASETS=/mnt/e/coco/
export OMP_NUM_THREADS=8
python -m exps.make_smaller_coco --image_counts=20000 --dpath=/mnt/e/coco

python tools/train.py \
    configs/swin/cascade_mask_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=12 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=2 \
                    log_config.interval=100 \
                    runner.max_epochs=3 \
                    data.train.ann_file='/mnt/e/coco/annotations/instances_train2017.small.json' \
    --work-dir=/mnt/e/coco/cps
