import os
from random import shuffle
import shutil

# 300명의 데이터셋만 추출
directory = []
org_dir = 'train'
for root, dirs, files in os.walk(org_dir):
    directory.append(root)
shuffle(directory)
directory = directory[:300]

for di in directory:
    for root, dirs, files in os.walk(di):
        path = 'dataset/' + root[6:]
        os.mkdir(path)
        for file in files:
            src = root + '/'
            shutil.copy(src + file, path + '/' + file)


