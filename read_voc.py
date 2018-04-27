'Main/car_train'
''' Modified from https://github.com/mprat/pascal-voc-python'''
import os
import pandas as pd
from bs4 import BeautifulSoup
import voc_utils
from more_itertools import unique_everseen
import pathlib
import sys

def load_data_full():
    # category name is from above, dataset is either "train" or
    # "val" or "train_val"
    def imgs_from_category(cat_name, dataset):
        filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'],
            dtype={'filename': object})
        return df

    def imgs_from_category_as_list(cat_name, dataset):
        df = imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]

        return df['filename'].values

    def annotation_file_from_img(img_name):
        return os.path.join(ann_dir, img_name) + '.xml'

    # annotation operations
    def load_annotation(img_filename):
        xml = ""
        #print "good here1"
        #print(annotation_file_from_img(img_filename))
        #print "good here2"
        with open(annotation_file_from_img(img_filename)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        return BeautifulSoup(xml)

    def get_all_obj_and_box(objname, img_set):
        img_list = imgs_from_category_as_list(objname, img_set)

        for img in img_list:
            annotation = load_annotation(img)

    # image operations
    def load_img(img_filename):
        return io.load_image(os.path.join(img_dir, img_filename + '.jpg'))

    # Loading train data
    def load_train_data(category):
        to_find = category
        #train_filename = '/Users/mprat/personal/VOCdevkit/VOC2012/csvs/train_' + category + '.csv'
        #if os.path.isfile(train_filename):
        #    return pd.read_csv(train_filename)
        #else:
        train_img_list = imgs_from_category_as_list(to_find, 'train')
        data = []
        for item in train_img_list:
            anno = load_annotation(item)
            objs = anno.findAll('object')
            for obj in objs:
                obj_names = obj.findChildren('name')
                for name_tag in obj_names:
                    if str(name_tag.contents[0]) == 'bicycle':
                        fname = anno.findChild('filename').contents[0]
                        img_size = anno.findChild('size')
                        #print "img_size", img_size
                        im_width = int(img_size.findChildren('width')[0].contents[0])
                        im_height = int(img_size.findChildren('height')[0].contents[0])
                        im_depth = int(img_size.findChildren('depth')[0].contents[0])
                        bbox = obj.findChildren('bndbox')[0]
                        xmin = int(bbox.findChildren('xmin')[0].contents[0])
                        ymin = int(bbox.findChildren('ymin')[0].contents[0])
                        xmax = int(bbox.findChildren('xmax')[0].contents[0])
                        ymax = int(bbox.findChildren('ymax')[0].contents[0])
                        data.append([fname, im_width, im_height, im_depth, xmin, ymin, xmax, ymax])
        df = pd.DataFrame(data, columns=['fname', 'im_width', 'im_height', 'im_depth',
                                         'xmin', 'ymin', 'xmax', 'ymax'])
        #print data
        #df.to_csv(train_filename, index=False)
        return df

    def not_rep_char(words, img_dir):
        word_list = list([])
        ban_list = list([])
        for w in words:
            if w not in ban_list:
                word_list.append(str(os.path.join(img_dir, w)))
                ban_list.append(w)
        #print ban_list
        return word_list

    """###############################PROGRAM STARTS ######################################"""

    root_dir = '/home/hanz/Documents/2018/datasets/VOCdevkit/VOC2007/'
    pth = pathlib.Path(root_dir)
    assert ((pathlib.Path(root_dir)).is_dir()), "ERROR: Please input a correct root_dir"
    print "path found"
    img_dir = os.path.join(root_dir, 'JPEGImages')
    ann_dir = os.path.join(root_dir, 'Annotations')
    set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
    assert ((pathlib.Path(img_dir)).is_dir()), "ERROR: file JPEGImages doesn't exist"
    assert ((pathlib.Path(ann_dir)).is_dir()), "ERROR: file Annotations doesn't exist"
    assert ((pathlib.Path(set_dir)).is_dir()), "ERROR: file ImageSets doesn't exist"
    print("Good: JPEGImages, Annotations, and ImageSets/Main exist")

    image_sets = []
    for file in os.listdir(set_dir):
        if file.endswith(".txt"):
            image_sets.append(str(file).strip().split('.')[0].split('_')[0])
    #print image_sets, "img_Set" ###Return list of txt files per class

    train_img_list = imgs_from_category_as_list('bicycle', 'train')
    #print train_img_list
    a = load_annotation(train_img_list[0])

    df = load_train_data('bicycle')
    #print df  ###RETURN panda data of file name with the bbxs ['fname', 'im_width', 'im_height', 'im_depth',
    #                                 'xmin', 'ymin', 'xmax', 'ymax']
    JPEG_images = not_rep_char(df['fname'], img_dir)
    #print JPEG_images ###RETURN list of the desired images

    #for cat in image_sets:
    #    if cat != 'train' and cat != 'val' and cat != 'trainval' and cat != 'test':
    #        load_train_data(cat)
    return JPEG_images, df

load_data_full()
#print list(unique_everseen(list(voc_utils.img_dir + df['fname'])))
#print(voc_utils.img_dir)
#print df
