# 2.voc2coco

import json
import os
import os.path as osp
import sys
import shutil
from PIL import Image,ImageDraw
import numpy as np

categories_list = []
labels_list = ['fish','car']
label_to_num = {'fish':1, 'car':2}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def getbbox(self, points):
    polygons = points
    mask = self.polygons_to_mask([self.height, self.width], polygons)
    return self.mask2box(mask)

def images_labelme(data, num):
    image = {}
    image['height'] = data['imageHeight']
    image['width'] = data['imageWidth']
    image['id'] = num + 1
    image['file_name'] = data['imagePath'].split('/')[-1]
    return image

def images_cityscape(num, img_file, w, h):
    image = {}
    image['height'] = h
    image['width'] = w
    image['id'] = num + 1
    image['file_name'] = img_file
    return image


def categories(label, labels_list):
    category = {}
    category['supercategory'] = 'component'
    category['id'] = len(labels_list) + 1
    category['name'] = label
    return category

def categories_custom(labels_list):
    categories = []
    count = 0
    for c in labels_list:
        category = {}
        category['supercategory'] = 'component'
        category['id'] = count + 1
        category['name'] = labels_list[count]
        count += 1
        categories.append(category)
    return categories

categories_list = categories_custom(labels_list)

def annotations_rectangle(points, label, image_num, object_num, label_to_num):
    annotation = {}
    seg_points = np.asarray(points).copy()
    annotation['segmentation'] = [list(seg_points.flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_num + 1
    annotation['bbox'] = list(
        map(float, [
            points[0][0], points[0][1], points[1][0] - points[0][0], points[1][
                1] - points[0][1]
        ]))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = object_num + 1
    return annotation


def annotations_polygon(height, width, points, label, image_num, object_num,
                        label_to_num):
    annotation = {}
    annotation['segmentation'] = [list(np.asarray(points).flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_num + 1
    annotation['bbox'] = list(map(float, get_bbox(height, width, points)))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = object_num + 1
    return annotation


def get_bbox(height, width, points):
    polygons = points
    mask = np.zeros([height, width], dtype=np.uint8)
    mask = Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    left_top_r = np.min(rows)
    left_top_c = np.min(clos)
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)
    return [
        left_top_c, left_top_r, right_bottom_c - left_top_c,
                                right_bottom_r - left_top_r
    ]

def deal_json_custom(label_info,image_num=-1,object_num=-1):
    data_coco = {}
    images_list = []
    annotations_list = []
    # image_num = -1
    # object_num = -1
    # image_num = 16864-1
    # object_num = 35046-1

    image_ids = []

    with open(label_info) as f:
        for line in f.readlines():
            line = eval(line.strip('\n'))
            imgpath = line['imgpath']
            image_id = imgpath.split('\\')[-1]
            image = Image.open(imgpath)
            mask_img = Image.open(line['maskpath'])
            h,w = image.height,image.width
            mh,mw = mask_img.height,mask_img.width
            point = line['point']
            Fxy = line['Fxy']
            xmin,ymin,xmax,ymax = point[0], point[1], point[0]+int(mw*Fxy[0]), point[1]+int(mh*Fxy[1])
            # xmin,ymin,xmax,ymax = line['bbox']
            label = line['maskname']
            if image_id not in image_ids:
                image_ids.append(image_id)
                image_num = image_num + 1
                images_list.append(images_cityscape(image_num, image_id, w, h))

            if label not in labels_list:
                print(image_id,label)
            else:
                object_num = object_num + 1
                annotations_list.append(
                    annotations_rectangle([[xmin,ymin],[xmax,ymax]], label, image_num,
                                            object_num, label_to_num))

    data_coco['images'] = images_list
    data_coco['categories'] = categories_list
    data_coco['annotations'] = annotations_list
    return data_coco

if __name__ == '__main__':
    import os.path as osp
    import glob
    import os
    import shutil


    label_info = r'labels-info.txt'
    train_data_coco = deal_json_custom(
        label_info)
    train_json_path = 'instance_train.json'
    json.dump(
        train_data_coco,
        open(train_json_path, 'w'),
        indent=4,
        cls=MyEncoder)

    # val_data_coco = deal_json('data/cocome' + '/val', train_img, val_list)
    # val_json_path = osp.join('data/cocome' + '/annotations', 'instance_val.json')
    # json.dump(val_data_coco, open(val_json_path, 'w'), indent=4, cls=MyEncoder)
