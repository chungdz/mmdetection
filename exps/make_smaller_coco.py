from email.mime import image
import json
import os
from tqdm import tqdm

image_counts = 20000
dpath = '/mnt/coco'

input_file = os.path.join(dpath, 'annotations', 'instances_train2017.json')
out_file = os.path.join(dpath, 'annotations', 'instances_train2017.small.json')

train_instance = json.load(open(input_file, "r"))
outd = {}
outd['info'] = train_instance['info']
outd['licenses'] = train_instance['licenses']
outd['categories'] = train_instance['categories']

outd['images'] = train_instance['images'][:image_counts]
iidset = set()
for iinfo in tqdm(outd['images']):
    iidset.add(iinfo['id'])

outd['annotations'] = []
for ainfo in tqdm(train_instance['annotations']):
    if ainfo['id'] in iidset:
        outd['annotations'].append(ainfo)

json.dump(outd, open(out_file, "w"))
