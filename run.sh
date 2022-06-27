# setup env
export MMDET_DATASETS=/mnt/

# Test on existing model and checkpoint
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --show
