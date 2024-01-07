import shutil
import os
import glob
from tqdm import tqdm

def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

# path = "/data/Kaggle/FaceClust/Datasets/Celeba/Anno/identity_CelebA.txt"
# f = open(path, "r")
# arr = [i.strip() for i in f.readlines()]
# f.close()

# identity_d = {}
# for a in tqdm(arr):
#     img_name, identity = a.split(" ")
#     if (identity in identity_d):
#         identity_d[identity].append(img_name)
#     else:
#         identity_d[identity] = [img_name]

# imgp = "/data/Kaggle/FaceClust/Datasets/Celeba/Img/img_align_celeba/"
# save_path = "/data/Kaggle/FaceClust/Datasets/CelebaTrain/"
# for identity in tqdm(identity_d):
#     MakeDir(save_path + identity + "/")
#     img_names = identity_d[identity]
#     if (len(img_names) > 3):
#         for imgname in img_names:
#             shutil.copy(imgp + imgname, save_path + identity + "/" + imgname)


imgp = "/data/Kaggle/FaceClust/Datasets/lfw-112X96/"
save_path = "/data/Kaggle/FaceClust/Datasets/lfw-112X96_test/"
MakeDir(save_path)
identity_d = os.listdir(imgp)
for identity in tqdm(identity_d):
    img_names = os.listdir(imgp + identity + "/")
    if (len(img_names) > 5):
        MakeDir(save_path + identity + "/")
        for imgname in img_names:
            shutil.copy(imgp + identity + "/" + imgname, save_path + identity + "/" + imgname)
