import os
import torch
import numpy as np
from backbones.mobilefacenet import MobileFaceNet
from backbones.mobilefacenetv2 import MobileFaceNetv2

from torchvision import transforms
from tqdm import tqdm
import yaml
import argparse
import cv2
import struct
from thop import profile
from clustering import ClusterInfomap
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import openvino as ov

def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

class DotDict:
    def __init__(self, data):
        self.__dict__.update(data)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


def model_size(file_path):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return size_mb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_flops(model, input):
    flops, params = profile(model, inputs=(input,))
    return flops, params

def read_data(path, image_size, infer_type):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print ("==== Reading Data =====")
    images = []
    labels = []
    orig_images = []
    classes = os.listdir(path)
    for idx, class_name in tqdm(enumerate(classes)):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path)
                orig_images.append(cv2.resize(image, (256, 256)))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if (infer_type == 1):
                    image = transform(image)
                elif (infer_type == 2):
                    image = cv2.resize(image, (image_size, image_size))
                    image = np.transpose(image, (2, 0, 1))
                    image = ((image / 255.0) - 0.5) / 0.5
                images.append(image)
                labels.append(idx)
    
    return images, labels, orig_images

def inference(images, labels, backbone, infer_type):
    print ("===== Extracting Embeddings =======")
    if (infer_type == 2):
        output_layer_ir = backbone.output(0)

    embeddings_list = []
    label_info = []
    fpsl = []
    with torch.no_grad():
        for i in tqdm(range(len(images))):
            data = images[i]
            st = time.time()
            img = data[None,:,:,:]
            if (infer_type == 1):
                net_out: torch.Tensor = backbone(img)
                embeddings = net_out.detach().numpy()
            elif (infer_type == 2):
                embeddings = backbone([img])[output_layer_ir]

            et = time.time()
            fpsl.append(1.0/(et-st))
            embeddings_list.append(embeddings)
            label_info.append(labels[i])
    embeddings = np.concatenate(embeddings_list, axis=0)

    return embeddings, label_info, fpsl

def save_data(embeddings, labels, savedir):
    print ("===== Saving Data ======")
    file_path = savedir + 'embeddings.bin'
    with open(file_path, 'wb') as file:
        for emb in embeddings:
            binary_data = struct.pack('f' * len(emb), *emb)
            file.write(binary_data)

    file_path = savedir + 'embeddings.tsv'
    with open(file_path, 'w') as file:
        for row in embeddings:
            row_str = '\t'.join(map(str, row)) + '\n'
            file.write(row_str)

    file_path = savedir + 'label.meta'
    with open(file_path, 'w') as file:
        for label in labels:
            file.write(f"{label}\n")

    embeddings_tensor = torch.tensor(embeddings)
    torch.save(embeddings_tensor, savedir + 'embeddings.pt')

    labels_tensor = torch.tensor(labels)
    torch.save(labels_tensor, savedir + 'labels.pt')

class CustomLogFile:
    def __init__(self, root):
        self.fp = root + "logs.txt"
        self.f = open(self.fp, "w")
    
    def write(self, text):
        self.f.write(text + "\n")
    
    def sep(self):
        self.f.write("\n")
        self.f.write("-" * 80)
        self.f.write("\n")
    
    def done(self):
        self.f.close()

def cluster_metrics(pred_labels, true_labels):
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    pf1 = fowlkes_mallows_score(true_labels, pred_labels)

    # print(f"Normalized Mutual Information (NMI): {nmi}")
    # print(f"Adjusted Rand Index (ARI): {ari}")
    return nmi, ari, pf1

def save_cluster_images(pred_labels, images, root):
    cluster = {}
    for idx in range(len(pred_labels)):
        label = pred_labels[idx]
        if (label not in cluster):
            cluster[label] = []
        cluster[label].append(idx)
    
    sp = root + "/cluster_results/"
    MakeDir(sp)
    for label in tqdm(cluster):
        p = sp + str(label) + "/"
        MakeDir(p)
        for idx in cluster[label]:
            img = images[idx]
            cv2.imwrite(p + str(idx) + ".jpg", img)
    print ("Done")

if __name__ == '__main__':

    cluster_method = ClusterInfomap(
        k = 30,
        min_sim=0.8
    )

    parser = argparse.ArgumentParser(description='FaceClusteringEvaluation')
    parser.add_argument('--model_version', help='model_version', default=None)
    parser.add_argument('--dataset_name', help='dataset_name', default="custom")
    parser.add_argument('--dataset_folder', help='dataset_path', type=str, default=None)
    args = parser.parse_args()

    model_version = args.model_version
    dataset_name = args.dataset_name
    dataset_folder = args.dataset_folder

    if (model_version is not None and dataset_folder is not None):
        config_path = "./config/evaluation/"+str(model_version)+".yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            config = DotDict(config)
            image_size = config.img_size
            backbone_type = config.backbone_type
            backbone_pth = config.backbone_weights
            MakeDir(config.save_dir)
            save_dir = config.save_dir + dataset_name + "/"
            MakeDir(save_dir)

            logger = CustomLogFile(root = save_dir)
            logger.write("Model Infomation")
            logger.sep()
            logger.write("Backbone Type : " + str(backbone_type))
            logger.write("Backbone Path : " + str(backbone_pth))

            if (backbone_type == "mobilefacenet"):
                backbone=MobileFaceNet()
                backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device("cpu")))
                backbone.eval()
                inference_type = 1

            elif (backbone_type == "mobilefacenetv2"):
                backbone=MobileFaceNetv2()
                backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device("cpu")))
                backbone.eval()
                inference_type = 1

            elif ("openvino" in backbone_type):
                core = ov.Core()
                model_ir = core.read_model(model=backbone_pth)
                backbone = core.compile_model(model=model_ir, device_name="CPU")
                inference_type = 2


            ms = model_size(backbone_pth)
            logger.write("Model Size in mb : " + str(ms))

            if (inference_type == 1):
                fakeinp = torch.randn(1, 3, image_size, image_size) 
                flops, params = calc_flops(backbone, fakeinp)
                logger.write("FLOPS : " + str(flops))
                logger.write("Params : " + str(params))


            images, labels, vis_images = read_data(dataset_folder, image_size, inference_type)
            embeddings, labels, fpsl = inference(images, labels, backbone, inference_type)

            np.savez(save_dir + "images_file.npz", images=vis_images)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            fps_mean = np.mean(fpsl[10:])
            logger.write("FPS : " + str(fps_mean))

            save_data(embeddings, labels, save_dir)

            cluster_labels = cluster_method.do_clustering(embeddings)
            with open(save_dir + "cluster_labels.txt", 'w') as of:
                for label in cluster_labels:
                    of.write(str(label) + '\n')
            print (cluster_labels)

            nmi, ari, pf = cluster_metrics(
                pred_labels=cluster_labels,
                true_labels=labels
            )
            logger.sep()
            logger.write("Cluster Method: Infomap")
            logger.write("NMI : " + str(nmi))
            logger.write("ARI : " + str(ari))
            logger.write("Pairwise F1 Score : " + str(pf))

            print ("=== Saving cluster results =====")
            save_cluster_images(cluster_labels, vis_images, save_dir)

    else:
        print ("Please specify arguments!!")