import os
import glob

if __name__=="__main__":

    rpath = "/data/Kaggle/FaceClust/Datasets/"
    dataset_l = [
        "105_classes_pins_dataset_test",
        "105_classes_pins_dataset_train",
        "CASIA",
        "CelebaTrain",
        "DigiFace"
    ]

    for dl in dataset_l:
        p = rpath + dl + "/"
        c = os.listdir(p)
        num_identities = len(c)
        num_images = 0
        for identity in c:
            p1 = p + identity + "/"
            images = os.listdir(p1)
            num_images += len(images)
        print (dl)
        print (num_identities)
        print (num_images)
        print ("-------------------")


