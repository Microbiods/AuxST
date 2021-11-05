# use for raw WSI image to color normalization

import glob
import staintools
import PIL
import tqdm
import os
import shutil
import cv2

target = staintools.read_image('data/STBC/template.jpg')
stain_norm = staintools.StainNormalizer(method='vahadane')
stain_norm.fit(target)

images = glob.glob("data/STBC/*/*/*.jpg")

for img in tqdm.tqdm((images)):  # color normalization or not, save both before CN or after CN

    ### here can be ignored 
    raw_path = img.replace(img.split('/')[-1],'').replace('STBC', 'image_raw')
    if os.path.exists(raw_path):
        shutil.copy(img, img.replace('STBC', 'image_raw'))
    else:
        os.makedirs(raw_path)
        shutil.copy(img, img.replace('STBC', 'image_raw'))
    ###

    X = staintools.read_image(img)  # return: RGB uint8 image.
    X = stain_norm.transform(X)
    X = PIL.Image.fromarray(X.astype('uint8')).convert('RGB')

    path = img.replace(img.split('/')[-1],'').replace('STBC', 'image_stained')
    # path = img.split('.')[0] + '_norm.' + img.split('.')[1]

    if os.path.exists(path):
        X.save(img.replace('STBC', 'image_stained'))
    else:
        os.makedirs(path)
        X.save(img.replace('STBC', 'image_stained'))


path = sorted(glob.glob("data/image_stained/*/*/*.jpg")) 

for p in path:
    img_name= p.split(".")[0] + '_mask.jpg'
    img = cv2.imread(p, 0)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(img_name, th3)