import argparse
import logging
import os

import torch
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from backbones.mobilefacenet import MobileFaceNet
from backbones.mobilefacenetv2 import MobileFaceNetv2

from losses import *

from utils import AverageMeter, test_model_pred_quality
from torch.utils.data import DataLoader
from dataset import FaceDataset, EvaluationFaceDataset
import numpy as np
from tqdm import tqdm
import wandb

import os
import yaml
import argparse

torch.backends.cudnn.benchmark = True

USE_WANDB = False

class DotDict:
    def __init__(self, data):
        self.__dict__.update(data)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

# Evaluation
class Evaluation(object):
    def __init__(self, frequent, eval_loader, device = "cuda:0"):
        self.frequent: int = frequent
        self.highest_acc: float = 0.0
        self.eval_loader = eval_loader
        self.device = device

    def ver_test(self, backbone, global_step):
        embeddings_list = []
        label_info = []
        with torch.no_grad():
            for _, (img, label) in tqdm(enumerate(self.eval_loader)):
                global_step += 1
                img = img.to(self.device)
                label = label.to(self.device)
                net_out = backbone(img)
                embeddings = net_out.detach().cpu().numpy()
                embeddings_list.append(embeddings)
                label_info.append(label.detach().cpu().numpy())

        embeddings = np.concatenate(embeddings_list, axis=0)
        labels = np.concatenate(label_info, axis=0)
        roc_percentile = test_model_pred_quality(embeddings, labels)
        print ("ROC : " + str(roc_percentile))
        return roc_percentile

    def __call__(self, num_update, backbone):
        backbone.eval()
        roc_perc = self.ver_test(backbone, num_update)
        backbone.train()
        return roc_perc

def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

def main(args):

    config_path = args.config_file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        cfg = DotDict(config)

    device = "cuda:0"
    torch.cuda.set_device(device)
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    
    print ("===== Initializing Training and Evaluation Dataset ========")
    trainset = FaceDataset(root_dir=cfg.rec, folder_l = cfg.train_l)
    evalset = EvaluationFaceDataset(root_dir=cfg.rec, folder_l = cfg.eval_l)

    train_loader = DataLoader(dataset=trainset, batch_size=cfg.batch_size, shuffle = True, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(dataset=evalset, batch_size=cfg.batch_size, shuffle = False, pin_memory=True, drop_last=True)

    print ("======== Loading Model ========")
    print (cfg.network)
    if cfg.network == "mobilefacenet":
        backbone = MobileFaceNet().to(device)
    elif cfg.network == "mobilefacenetv2":
        backbone = MobileFaceNetv2().to(device)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    if args.resume:
        try:
            print ("=== resuming stuff===")
            backbone_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(device)))
            print("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            print("load backbone resume init, failed!")

    backbone.train()

    # get header
    print ("===== Loading Header Training ======")
    header = ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(device)



    if args.resume:
        try:
            print ("==== Resuming stuff =====")
            header_pth = os.path.join(cfg.output, str(cfg.global_step) + "header.pth")
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(device)))
            print("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            print("header resume init, failed!")
            exit()
    
    header.train()

    print ("==== Initializing Optimizers =====")
    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

    print ("====== Initializing Optimizers =======")
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_backbone, lr_lambda=lr_step_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_header, lr_lambda=lr_step_func)        

    print ("===== Initializing Loss ======")
    criterion = CrossEntropyLoss()

    start_epoch = 0
    total_step = int((len(trainset) / cfg.batch_size) * cfg.num_epoch)
    print("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        print("resume from estimated epoch {}".format(cur_epoch))
        print("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    global_step = cfg.global_step
    evaluation = Evaluation(cfg.eval_step, eval_loader)

    # print ("==== Doing prior evaluation ======")
    # roc_eval = evaluation(global_step, backbone)

    if (USE_WANDB):
        wandb.log({
            "roc_eval": roc_eval[1]
        })

    print ("===== Starting Training Now.... Enjoy Coffee....=====")
    loss = AverageMeter()
    for epoch in range(start_epoch, cfg.num_epoch):
        for local_idx, (img, label) in enumerate(train_loader):
            global_step += 1
            
            img = img.to(device)
            label = label.to(device)

            features = F.normalize(backbone(img)) # Getting features
            thetas = header(features, label) # Getting thetas
            loss_v = criterion(thetas, label)
            loss_v.backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            
            if (local_idx % 100 == 0):
                print ("Loss : " + str(loss.avg))
                if (USE_WANDB):
                    wandb.log({
                        "loss": loss.avg
                    })

        if (global_step%cfg.eval_step == 0):
            roc_eval = evaluation(global_step, backbone)
            if (USE_WANDB):
                wandb.log({
                    "epoch_loss": loss.avg,
                    "roc_eval": roc_eval[1]
                })

        scheduler_backbone.step()
        scheduler_header.step()

        output_path = cfg.output
        MakeDir(output_path)
        if global_step > 100:
            torch.save(backbone.state_dict(), os.path.join(output_path, str(global_step)+ "backbone.pth"))
        if global_step > 100 and header is not None:
            torch.save(header.state_dict(), os.path.join(output_path, str(global_step)+ "header.pth"))

    # dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Params')
    parser.add_argument('--project', type=str, default="FaceFeatureExtractor", help="resume training")
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    parser.add_argument('--config_file', help='model_version', default=None)

    args_ = parser.parse_args()
    if (USE_WANDB):
        wandb.init(project=args_.project)
    main(args_)
