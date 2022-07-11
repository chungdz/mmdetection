cp -r -v /root/autodl-pub/COCO2017/ /root/autodl-tmp/coco/

cd /root/autodl-tmp/coco/
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

rm train2017.zip
rm val2017.zip
rm test2017.zip
rm annotations_trainval2017.zip


pip install mmcv-full
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"
