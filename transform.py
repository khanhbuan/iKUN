import os
import json
from utils import *
from collections import defaultdict

# bước 1: liệt kê các object_id trong các video
# bước 2: mỗi object_id lại ứng với danh sách các frame
# bước 3: mỗi cặp (id, frame) gắn với 1 danh sách các expression
# bước 4: mỗi bộ (video, id, frame) sẽ gắn liền với 1 bounding box

labels = {}
target_expressions = defaultdict(list)


for video in RESOLUTION.keys():
    object = {}
    dirs = os.listdir(os.path.join('./plugins/Refer-KITTI/expression', video))
    for dir in dirs:
        data = json.load(open(os.path.join(os.path.join('./plugins/Refer-KITTI/expression', video, dir))))
        for frame, id_list in data["label"].items():
            frame = int(frame)
            for id in id_list:
                if id not in object.keys():
                    object[id] = {}
                object = {key: object[key] for key in sorted(object)}
                
                if frame not in object[id].keys():
                    object[id][frame] = {}
                object[id] = {key: object[id][key] for key in sorted(object[id])}
                
                if "expression_new" not in object[id][frame].keys():
                    object[id][frame]["expression_new"] = []

                expression = expression_conversion(dir.split(".")[0])
                # expression = dir.split(".")[0]
                if expression not in object[id][frame]["expression_new"]:
                    object[id][frame]["expression_new"].append(expression)
                
                object[id][frame]["expression_new"].sort()
                
                if "bbox" not in object[id][frame].keys():
                    object[id][frame]["bbox"] = []

                path = os.path.join("./plugins/Refer-KITTI/KITTI/labels_with_ids/image_02", video, f"{frame:06}"+".txt")
                with open(path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip().split(" ")
                        if line[1] == str(id):
                            object[id][frame]["bbox"] = [float(x) for x in line[2:6]]
                            break
                
                if "category" not in object[id][frame].keys():
                    object[id][frame]["category"] = []

                for word in WORDS["category"]:
                    if word in expression:
                        object[id][frame]["category"] = [word]
                        break
                
    labels[video] = object

with open('Refer-KITTI_labels.json', 'w', encoding='utf-8') as file:
    json.dump(labels, file, ensure_ascii=False, indent=4)