from PIL import Image
import torchvision.transforms as T
import torch
from torchvision.models import resnet50
import os
from os.path import join, isdir, isfile
import gzip
import numpy as np
from torch import nn
from tqdm import tqdm



def main():
    resize_dim = (320, 512)
    resize = T.Resize(resize_dim)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    device = 'cuda:0'
    src_path = "/data/add_disk1/zbyang/datasets/coco_search18/images/"
    target_path = "./ISVN/img_features/"

    fix_clusters = np.load('/data/add_disk1/zbyang/datasets/coco_search18/clusters.npy', allow_pickle=True).item()
    task_img_pairs = list(fix_clusters.keys())
    task_img_pairs = list(filter(lambda x: x.split('-')[1] != 'freeview', task_img_pairs))

    model_ft = resnet50(pretrained=True)
    model = nn.Sequential(*list(model_ft.children())[:-2]).to(device)

    model.eval()
    for pair in tqdm(task_img_pairs):
        task, img_name = pair.split('-')[-2:]
        task = task.replace(' ', '_')
        if not (os.path.exists(join(target_path, task)) and os.path.isdir(join(target_path, task))):
            os.mkdir(join(target_path, task))
        
        PIL_image = Image.open(join(src_path, task, img_name + '.jpg')).convert('RGB')
        tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(tensor_image).squeeze().detach().cpu()
        with gzip.GzipFile(join(target_path, task, img_name + 'npy.gz'), "w") as w:
            np.save(w, features.numpy(), allow_pickle=True)
            w.close()

if __name__ == '__main__':
    main()