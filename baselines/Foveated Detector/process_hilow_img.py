import numpy as np
from PIL import Image
import glob, os
from shutil import copyfile
import matplotlib.pyplot as plt
import random

def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    return mask.astype(np.uint8)

def img2hilow(img_h_path, img_l_path, x, y, r, h, w):
    # x, y: fixated location, r: mask radius, h,w: image size
    img = Image.open(img_h_path)
    lr_img = Image.open(img_l_path)
    img = np.array(img)
    h, w, c = img.shape
    mask = foveal2mask(x, y, r, h, w)
    mask = np.dstack([mask] * c)
    hi_img = img * mask
    low_img = lr_img * (1 - mask)
    hl_img = hi_img + low_img
    return hl_img  #np.array
            


targets = ['bottle', 'chair', 'cup', 'fork', 'laptop', 'microwave',
                   'sink', 'stop sign', 'toilet', 'tv', 'bowl', 'car',
                   'clock', 'keyboard', 'knife', 'mouse', 'oven', 'potted plant']


annos= list(np.load('./annos/annos_trainval_320x512.npy', allow_pickle=True))
root_dir = './search/'
list_dir = root_dir + 'images/320x512/present/'
src_dir = root_dir + 'imagesLR/320x512/present/'
dst_dir = root_dir + 'imagesHiLow/320x512/present/'
tgt_dir_from = root_dir + 'target/'
tgt_dir_save = root_dir + 'targetHiLow/'
fdm_dir_from = root_dir + 'density/'
fdm_dir_save = root_dir + 'densityHiLow/'

r = 24
for target in targets:
    img_list = glob.glob(list_dir + target + '/*.jpg')
    img_dir = src_dir + target + '/'
    save_dir = dst_dir + target + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_path in img_list:

        img_name = os.path.split(img_path)[-1]
        img_id = int(img_name.split('.')[0])
        lr_img_name = 'blur_' + img_name
        print(target, img_name)

        img = Image.open(img_dir + img_name)
        lr_img = Image.open(img_dir + lr_img_name)
        img = np.array(img)
        h, w, c = img.shape

        img_anno = list(filter(lambda x: x['image_id']== img_id, annos))
        x = w/2.0
        y = h/2.0
        mask = foveal2mask(x, y, r, h, w)
        mask = np.dstack([mask] * c)
        hi_img = img * mask
        low_img = lr_img * (1 - mask)
        hl_img = hi_img + low_img
        hl_img = Image.fromarray(hl_img)
        hl_img.save(save_dir + img_name)
        if os.path.isfile(tgt_dir_from + 'train/' + target + '/{}'.format(img_name)):
            if not os.path.exists(tgt_dir_save + 'train/' + target):
                os.makedirs(tgt_dir_save + 'train/' + target)
            copyfile(tgt_dir_from + 'train/' + target + '/{}'.format(img_name),
                     tgt_dir_save + 'train/' + target + '/{}'.format(img_name))
        if os.path.isfile(tgt_dir_from + 'valid/' + target + '/{}'.format(img_name)):
            if not os.path.exists(tgt_dir_save + 'valid/' + target):
                os.makedirs(tgt_dir_save + 'valid/' + target)
            copyfile(tgt_dir_from + 'valid/' + target + '/{}'.format(img_name),
                     tgt_dir_save + 'valid/' + target + '/{}'.format(img_name))
        if os.path.isfile(fdm_dir_from + 'train/' + target + '/{}'.format(img_name)):
            if not os.path.exists(fdm_dir_save + 'train/' + target):
                os.makedirs(fdm_dir_save + 'train/' + target)
            copyfile(fdm_dir_from + 'train/' + target + '/{}'.format(img_name),
                     fdm_dir_save + 'train/' + target + '/{}'.format(img_name))
        if os.path.isfile(fdm_dir_from + 'valid/' + target + '/{}'.format(img_name)):
            if not os.path.exists(fdm_dir_save + 'valid/' + target):
                os.makedirs(fdm_dir_save + 'valid/' + target)
            copyfile(fdm_dir_from + 'valid/' + target + '/{}'.format(img_name),
                     fdm_dir_save + 'valid/' + target + '/{}'.format(img_name))

        try:
            ind = random.sample(range(len(img_anno)), 5)
            for i in ind:
                x = img_anno[i]['bbox'][0] + img_anno[i]['bbox'][2]/2.0
                y = img_anno[i]['bbox'][1] + img_anno[i]['bbox'][3]/2.0
                mask = foveal2mask(x, y, r, h, w)
                mask = np.dstack([mask] * c)
                hi_img = img * mask
                low_img = lr_img * (1 - mask)
                hl_img = hi_img + low_img
                hl_img = Image.fromarray(hl_img)
                hl_img.save(save_dir + str(i) + '_' + img_name)
                if os.path.isfile(tgt_dir_from + 'train/' + target + '/{}'.format(img_name)):
                    copyfile(tgt_dir_from + 'train/' + target + '/{}'.format(img_name),
                             tgt_dir_save + 'train/' + target + '/{}_{}'.format(str(i), img_name))
                if os.path.isfile(tgt_dir_from + 'valid/' + target + '/{}'.format(img_name)):
                    copyfile(tgt_dir_from + 'valid/' + target + '/{}'.format(img_name),
                             tgt_dir_save + 'valid/' + target + '/{}_{}'.format(str(i), img_name))
                if os.path.isfile(fdm_dir_from + 'train/' + target + '/{}'.format(img_name)):
                    copyfile(fdm_dir_from + 'train/' + target + '/{}'.format(img_name),
                             fdm_dir_save + 'train/' + target + '/{}_{}'.format(str(i), img_name))
                if os.path.isfile(fdm_dir_from + 'valid/' + target + '/{}'.format(img_name)):
                    copyfile(fdm_dir_from + 'valid/' + target + '/{}'.format(img_name),
                             fdm_dir_save + 'valid/' + target + '/{}_{}'.format(str(i), img_name))


        except:
            for i, anno in enumerate(img_anno):
                x = anno['bbox'][0] + img_anno[i]['bbox'][2]/2.0
                y = anno['bbox'][1] + img_anno[i]['bbox'][3]/2.0
                mask = foveal2mask(x, y, r, h, w)
                mask = np.dstack([mask] * c)
                hi_img = img* mask
                low_img = lr_img * (1-mask)
                hl_img = hi_img + low_img
                hl_img = Image.fromarray(hl_img)
                hl_img.save(save_dir + str(i) + '_' + img_name)
                if os.path.isfile(tgt_dir_from + 'train/' + target + '/{}'.format(img_name)):
                    copyfile(tgt_dir_from + 'train/' + target + '/{}'.format(img_name),
                             tgt_dir_save + 'train/' + target + '/{}_{}'.format(str(i), img_name))
                if os.path.isfile(tgt_dir_from + 'valid/' + target + '/{}'.format(img_name)):
                    copyfile(tgt_dir_from + 'valid/' + target + '/{}'.format(img_name),
                             tgt_dir_save + 'valid/' + target + '/{}_{}'.format(str(i), img_name))
                if os.path.isfile(fdm_dir_from + 'train/' + target + '/{}'.format(img_name)):
                    copyfile(fdm_dir_from + 'train/' + target + '/{}'.format(img_name),
                             fdm_dir_save + 'train/' + target + '/{}_{}'.format(str(i), img_name))
                if os.path.isfile(fdm_dir_from + 'valid/' + target + '/{}'.format(img_name)):
                    copyfile(fdm_dir_from + 'valid/' + target + '/{}'.format(img_name),
                             fdm_dir_save + 'valid/' + target + '/{}_{}'.format(str(i), img_name))
