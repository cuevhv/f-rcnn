# f-rcnn
faster rcnn

Steps
  - Download the repo.
  - Download the Pascal VOC dataset 2007:
  
      wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
      wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
      wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

      tar xvf VOCdevkit_08-Jun-2007.tar 
      tar xvf VOCtrainval_06-Nov-2007.tar
      tar xvf VOCtest_06-Nov-2007.tar
  - run mine_faster_rcnn.py
  
NOTE: read_voc.py will read the files of the PASCAL VOC dataset and return the file path of the images and labels of a given class.
