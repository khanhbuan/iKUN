import os
import clip
import torch
import json
from utils import * 
import torch.nn.functional as F

train = {}
test = {}

root = "./plugins/Refer-KITTI/expression"

for video in VIDEOS["train"]:
    dir = os.path.join(root, video)
    files = os.listdir(dir)
    for file in files:
        file = file.split(".")[0]
        expression = expression_conversion(file)
        if expression not in train.keys():
            train[expression] = {}
        if "probability" not in train[expression].keys():
            train[expression]['probability'] = 0
        train[expression]['probability'] = train[expression]['probability'] + 1

total = 0
for expression in train.keys():
    total += train[expression]['probability']

model, _ = clip.load("./plugins/CLIP/ViT-B-32.pt", device="cpu")
for expression in train.keys():
    texts = clip.tokenize([expression])
    with torch.no_grad():
        feat = model.encode_text(texts)
        feat = F.normalize(feat, p=2)
        feat = feat.detach().cpu().tolist()[0]
    train[expression]["probability"] = (train[expression]['probability'] * 1.0) / total
    train[expression]["feature"] = feat

data = {"train": train}

for video in VIDEOS["test"]:
    dir = os.path.join(root, video)
    files = os.listdir(dir)
    for file in files:
        file = file.split(".")[0]
        expression = expression_conversion(file)
        if expression not in test.keys():
            test[expression] = {}
            texts = clip.tokenize([expression])
            with torch.no_grad():
                feat = model.encode_text(texts)
                feat = F.normalize(feat, p=2)
                feat = feat.detach().cpu().tolist()[0]
            test[expression]["feature"] = feat

data["test"] = test

with open('./plugins/textual_features.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)