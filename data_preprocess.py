# data_preprocess.py
import os
import re
import glob
import config

def get_bboxes(classes, xmin_list, xmax_list, ymin_list, ymax_list):
    output = []
    for c, xmin, xmax, ymin, ymax in zip(classes, xmin_list, xmax_list, ymin_list, ymax_list):
        output.extend([c, xmin, xmax, ymin, ymax])

    return output

def get_voc_classidx(classes):
    class_map = config.class_map

    for i in range(len(classes)):
        classes[i] = class_map[classes[i]]

    return classes

def xml2info(xml_path, img_dir):
    with open(xml_path, 'r') as f:
        data = f.read()
    if 'part' in data:
        return 
    img_name = re.findall(r'<filename>(.*?)</filename>', data)
    img_path = os.path.join(img_dir, img_name[0])
    img_width = re.findall(r'<width>(.*?)</width>', data)
    img_height = re.findall(r'<height>(.*?)</height>', data)
    classes = re.findall(r'<name>(.*?)</name>', data)[1:]
    classes = get_voc_classidx(classes)
    xmin_list = re.findall(r'<xmin>(.*?)</xmin>', data)
    xmax_list = re.findall(r'<xmax>(.*?)</xmax>', data)
    ymin_list = re.findall(r'<ymin>(.*?)</ymin>', data)
    ymax_list = re.findall(r'<ymax>(.*?)</ymax>', data)
    bboxes = get_bboxes(classes, xmin_list, xmax_list, ymin_list, ymax_list)
    ret = [img_path] + img_width + img_height + bboxes

    return ret

def datainfo2txt(xml_dir, img_dir, output_path):
    with open(output_path, 'w') as f:
        for xml_file in glob.glob(xml_dir):
            ret = xml2info(xml_file, img_dir) 
            if not ret:
                continue
            f.write(' '.join(ret) + '\n')

if __name__ == '__main__':
    print('data preprocess')
    xml2info('./data/VOCtrainval/Annotations/000005.xml', './data/VOCtrainval/JPEGImages')
    datainfo2txt('./data/VOCtrainval/Annotations/*', './data/VOCtrainval/JPEGImages/', './data/train.txt')
    datainfo2txt('./data/VOCtest/Annotations/*', './data/VOCtest/JPEGImages/', './data/test.txt')

