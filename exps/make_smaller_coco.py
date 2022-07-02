from email.mime import image
import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Make small dataset')
parser.add_argument('--image_counts', type=int,default=20000)
parser.add_argument('--dpath', type=str, default='/mnt/coco')

args = parser.parse_args()
image_counts = args.image_counts
dpath = args.dpath

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
