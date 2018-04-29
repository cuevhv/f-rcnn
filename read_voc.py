'Main/car_train'
''' LOAD VOC FILE - NO LIBRARIES, NO INSTALL, NO TRICKS '''
import os
import pandas as pd
from bs4 import BeautifulSoup
import pathlib
import sys
import numpy as np
import cv2

def load_data_full(class_name, shw_example=True):
    # category name is from above, dataset is either "train" or
    # "val" or "train_val"
    def imgs_from_category(cat_name, dataset):
        if cat_name == '':
            filename = os.path.join(set_dir, cat_name + dataset + ".txt")
        else:
            filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'],
            dtype={'filename': object})
        #print df
        return df

    def imgs_from_category_as_list(cat_name, dataset):
        df = imgs_from_category(cat_name, dataset)
        if cat_name != '':
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
                    if str(name_tag.contents[0]) == to_find or to_find == '':
                        fname = anno.findChild('filename').contents[0]
                        img_size = anno.findChild('size')
                        #print "img_size", img_size
                        im_width = int(img_size.findChildren('width')[0].contents[0])
                        im_height = int(img_size.findChildren('height')[0].contents[0])
                        im_depth = int(img_size.findChildren('depth')[0].contents[0])
                        bbox = obj.findChildren('bndbox')[0]
                        img_class = str(obj.findChildren('name')[0].contents[0])
                        xmin = int(bbox.findChildren('xmin')[0].contents[0])
                        ymin = int(bbox.findChildren('ymin')[0].contents[0])
                        xmax = int(bbox.findChildren('xmax')[0].contents[0])
                        ymax = int(bbox.findChildren('ymax')[0].contents[0])
                        data.append([fname, im_width, im_height, im_depth, xmin, ymin, xmax, ymax, img_class])
        df = pd.DataFrame(data, columns=['fname', 'im_width', 'im_height', 'im_depth',
                                         'xmin', 'ymin', 'xmax', 'ymax', 'img_class'])
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

    def finding_VOC_path():
        for root, dirs, files in os.walk('/home'):
            for name in dirs:
                if (name.endswith('VOC2007')):
                    #print os.path.join(root, 'VOC2007')
                    for rt, drs, fls in os.walk(os.path.join(root, 'VOC2007')):
                        #print drs
                        count = 0
                        for nam in drs:
                            if (nam.endswith('Annotations')):
                                count += 1
                            if (nam.endswith('ImageSets')):
                                count += 1
                            if (nam.endswith('JPEGImages')):
                                count += 1
                        if count == 3:
                            print os.path.join(root, 'VOC2007')
                            return os.path.join(root, 'VOC2007/')
                            print count
        return 'NO_PATH'
    """###############################PROGRAM STARTS ######################################"""
    string_path = finding_VOC_path()
    root_dir = string_path
    #root_dir = '/home/hanz/Documents/2018/datasets/VOCdevkit/VOC2007/'
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
    image_sets = set(image_sets)
    print image_sets, "img_Set" ###Return list of txt files per class

    type_data = class_name

    train_img_list = imgs_from_category_as_list(type_data, 'train')

    a = load_annotation(train_img_list[0])
    df = load_train_data(type_data)
    #print df  ###RETURN panda data of file name with the bbxs ['fname', 'im_width', 'im_height', 'im_depth',
    #                                 'xmin', 'ymin', 'xmax', 'ymax']
    JPEG_images = not_rep_char(df['fname'], img_dir)
    #print JPEG_images ###RETURN list of the desired images
    Annotation_images = not_rep_char(df['fname'], ann_dir)
    return JPEG_images, Annotation_images, df

    #for cat in image_sets:
    #    if cat != 'train' and cat != 'val' and cat != 'trainval' and cat != 'test':
    #        load_train_data(cat)

    ###### Example #######
"""    class_size = len(JPEG_images)
    print 'class_size', class_size
    img_n = 1#np.random.randint(class_size) #What image is going to be shown
    print "There are %d of " %(class_size)+type_data+"Showing image %d" %(img_n)

    print JPEG_images[img_n], Annotation_images[img_n]
    bbxs_index = df.index[df['fname'] == JPEG_images[img_n].split('/')[-1]].tolist()
    bbxs_sizes = []
    for bbx in bbxs_index:
        bbxs_sizes.append([df['xmin'][bbx], df['ymin'][bbx], df['xmax'][bbx], df['ymax'][bbx]])
    print "Bounding boxes sizes in this iamge: \n", bbxs_sizes

    #for indx in df['fname']:
    #    if indx == JPEG_images[img_n].split('/')[-1]:
    #        print indx
    #shw_example = False
    if shw_example == True:
        img = cv2.imread(JPEG_images[img_n])
        for b in bbxs_sizes:
            xmin = b[0]
            ymin = b[1]
            xmax = b[2]
            ymax = b[3]
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),
                          (np.random.randint(255),np.random.randint(255),np.random.randint(255)),3)
        cv2.imshow(JPEG_images[img_n], img)

        cv2.waitKey(0)
        print "Image it was closed"
        cv2.destroyAllWindows()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return JPEG_images, Annotation_images, df, bbxs_sizes"""

def print_images(JPEG_images, Annotation_images, df, type_data, shw_example = True):
    class_size = len(JPEG_images)
    img_n = 1#np.random.randint(class_size) #What image is going to be shown
    print "There are %d of " %(class_size)+type_data+" Showing image %d" %(img_n)

    print JPEG_images[img_n], Annotation_images[img_n]
    bbxs_index = df.index[df['fname'] == JPEG_images[img_n].split('/')[-1]].tolist()
    bbxs_sizes = []
    for bbx in bbxs_index:
        bbxs_sizes.append([df['xmin'][bbx], df['ymin'][bbx], df['xmax'][bbx], df['ymax'][bbx]])
    print "Bounding boxes sizes in this iamge: \n", bbxs_sizes

    #for indx in df['fname']:
    #    if indx == JPEG_images[img_n].split('/')[-1]:
    #        print indx
    if shw_example == True:
        img = cv2.imread(JPEG_images[img_n])
        for b in bbxs_sizes:
            xmin = b[0]
            ymin = b[1]
            xmax = b[2]
            ymax = b[3]
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),
                          (np.random.randint(255),np.random.randint(255),np.random.randint(255)),3)
        cv2.imshow(JPEG_images[img_n], img)

        cv2.waitKey(0)
        print "Image it was closed"
        cv2.destroyAllWindows()

def getting_all_bbx(Annotation_images):
    class_size = len(Annotation_images)
    print "There are %d of " %(class_size)+type_data
    bbxs_sizes = []
    for img_n in range(0, class_size-1):
        #print img_n
        #print "asdfsfsdf", Annotation_images[img_n]
        bbxs_index = df.index[df['fname'] == Annotation_images[img_n].split('/')[-1]].tolist()
        bbxs_sizes.append([])
        for bbx in bbxs_index:
            bbxs_sizes[img_n].append([df['xmin'][bbx], df['ymin'][bbx], df['xmax'][bbx], df['ymax'][bbx]])
    #print "Bounding boxes sizes in this iamge: \n", bbxs_sizes
    return bbxs_sizes

def bb_intersection_over_union(boxA, boxB):
    #boxA and B [xmin, ymin, xmax, ymax]
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

type_data = 'person'
shw_example = False
JPEG_images, Annotation_images, df = load_data_full(type_data, shw_example)
bbxs_sizes = getting_all_bbx(Annotation_images)
print_images(JPEG_images, Annotation_images, df, type_data, shw_example)
print bbxs_sizes[1]
print bb_intersection_over_union(bbxs_sizes[1][0], bbxs_sizes[1][1])
print bb_intersection_over_union([1, 1, 3, 3], [1, 1, 3, 3])
