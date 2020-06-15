# For voc dataset setting
width = 608
height = 608
class_num = 20
anchor_num = 3
anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
stride1 = 32
stride2 = 16 
stride3 = 8
class_map = {
        'aeroplane': '1', 'bicycle': '2', 'bird': '3', 'boat': '4', 'bottle': '5', 
        'bus': '6', 'car': '7', 'cat': '8', 'chair': '9', 'cow': '10', 'diningtable': '11', 
        'dog': '12', 'horse': '13', 'motorbike': '14', 'person': '15', 'pottedplant': '16',
        'sheep': '17', 'sofa': '18', 'train': '19', 'tvmonitor': '20'}
index2class = {int(index): class_name for class_name, index in class_map.items()}

# For loss function setting
ignore_thresh = 0.5
prob_thresh = 0.25
score_thresh = 0.25
iou_normalizer = 0.07