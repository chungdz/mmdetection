# setup env
export MMDET_DATASETS=/mnt/
export OMP_NUM_THREADS=8

# Test on existing model and checkpoint
# refer for the website https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md
# weightes are downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn and save into checkpoints
python tools/test.py\
 configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
 checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
 --show \
 --work-dir=results \
 --eval="bbox" \

python tools/train.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --auto-scale-lr \
    --cfg-options auto_scale_lr.base_batch_size=12 \
                    data.workers_per_gpu=8 \
                    data.samples_per_gpu=12 \
                    log_config.interval=30000 \
                    runner.max_epoch=3 \
    --work-dir=/mnt/cps/coco/


    



