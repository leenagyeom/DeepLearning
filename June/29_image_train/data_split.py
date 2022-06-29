import os
import random
import shutil
import numpy as np

# 데이터 추출해서 옮길 폴더 저장
os.makedirs("./test_images", exist_ok=True)
os.makedirs("./test_annotations", exist_ok=True)

print(len(os.listdir("./annotations")))
print(len(os.listdir("./images")))

random.seed(6146)
idx = random.sample(range(853), 170)

for img in np.array(sorted(os.listdir("./images")))[idx]:
    print("img info", img)
    shutil.move("./images/"+img,"./test_images/"+img)

for anno in np.array(sorted(os.listdir("./annotations")))[idx]:
    print("annotation info", anno)
    shutil.move("./annotations/"+anno, "./test_annotations/"+anno)

print("info file size\n")
print(len(os.listdir("./images")))
print(len(os.listdir("./annotations")))
print(len(os.listdir("./test_images")))
print(len(os.listdir("./test_annotations")))