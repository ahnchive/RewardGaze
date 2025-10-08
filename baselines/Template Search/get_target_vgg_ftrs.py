from PIL import Image
import torchvision.transforms as T
import torch
from torchvision.models import resnet50
import os
from os.path import join, isdir, isfile
import gzip
import numpy as np
from torch import nn


def main():
    resize_dim = (32, 32)
    resize = T.Resize(resize_dim)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    device = 'cuda:0'
    src_path = "./target_images/"
    target_path = "./target_features/"
    
    model_ft = resnet50(pretrained=True)
    model = nn.Sequential(*list(model_ft.children())[:-1]).to(device)
    model.eval()
    #counter = 0
    files =  [i for i in os.listdir(src_path) if isfile(join(src_path, i)) and i.endswith('.jpg')]
    for f in files:
        #counter = counter + 1
        #print(counter, end='\r')
        PIL_image = Image.open(join(src_path, f)).convert('RGB')
        tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(tensor_image).squeeze().detach().cpu()
        print(features.size())
        
        with gzip.GzipFile(join(target_path,  f.replace('jpg', 'npy.gz')), "w") as w:
            np.save(w, features.numpy(), allow_pickle=True)
            w.close()
    #print(counter)

if __name__ == '__main__':
    main()