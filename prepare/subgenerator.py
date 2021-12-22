import numpy as np
import cv2
import glob
import pathlib
import PIL
import pickle
import collections
import torchvision
import torch
import openslide
import sys
sys.path.append('./')
import utils



# ### get patches with different cropped sizes


# dataset = sorted(glob.glob("training/counts/*/*/*.npz"))  # the npz collection for each count file

# patients = sorted(utils.util.get_spatial_patients().keys())

# test_patients = ["BC23450", "BC23903"]
# train_patients = [p for p in patients if p not in test_patients]

# # print("Train patients: ",  train_patients)
# # print("Test patients: ", test_patients)
# # print()  

# # print(dataset)


# dataset = [d for d in dataset if ((d.split("/")[-2] in train_patients) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in train_patients))]
    

# for win in [128, 150, 299, 384, 512]:

#     print("Saving and cropping patches with window size : " + str(win))

#     train_dataset = utils.dataloader.SubGenerator(train_patients, window = win, 
#                             img_cached = 'training/images/',
#                             transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, 
#                             num_workers=0, shuffle=True)

#     for (i, (he, npz)) in enumerate(train_loader):
#             # calculate the white ratio and delete the noise images
#         print("Saving training filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(train_dataset)) + '...')


#     test_dataset = utils.dataloader.SubGenerator(test_patients, window = win, 
#                             img_cached = 'test/images/',
#                             transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]

#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
#                             num_workers=0, shuffle=True)

#     # print(len(test_dataset))

#     for (i, (he, npz)) in enumerate(test_loader): 
#         print("Saving test filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(test_dataset)) + '...')








### get patches with different resolutions


dataset = sorted(glob.glob("training/counts/*/*/*.npz"))  # the npz collection for each count file

patients = sorted(utils.util.get_spatial_patients().keys())

test_patients = ["BC23450", "BC23903"]
train_patients = [p for p in patients if p not in test_patients]

# print("Train patients: ",  train_patients)
# print("Test patients: ", test_patients)
# print()  

# print(dataset)


dataset = [d for d in dataset if ((d.split("/")[-2] in train_patients) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in train_patients))]
    

for re in [128, 256, 299, 384, 512]:

    print("Saving and cropping patches with resolution : " + str(re))

    train_dataset = utils.dataloader.SubGenerator(train_patients, window = 299, resolution =re,
                            img_cached = 'training/images/',
                            transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, 
                            num_workers=0, shuffle=True)

    for (i, (he, npz)) in enumerate(train_loader):
            # calculate the white ratio and delete the noise images
        print("Saving training filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(train_dataset)) + '...')


    test_dataset = utils.dataloader.SubGenerator(test_patients, window = 299, resolution =re,
                            img_cached = 'test/images/',
                            transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
                            num_workers=0, shuffle=True)

    # print(len(test_dataset))

    for (i, (he, npz)) in enumerate(test_loader): 
        print("Saving test filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(test_dataset)) + '...')























    # print(torch.sum(1 * (wr <= ratio))) # if less than 0.2, return true 1, else return false 0
    # cnt = cnt + torch.sum(1 * (wr <= ratio))



# test_dataset = utils.dataloader.Generator(test_patients,
#                         test_mode = 1,
#                         count_cached = 'test/counts/',
#                         img_cached = 'test/images/',
#                         transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]

# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
#                         num_workers=0, shuffle=True)


# for (i, (he, npz)) in enumerate(train_loader):  # here may conduct all the cutting work but only for training loader
#     # calculate the white ratio and delete the noise images
#     print("Saving filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(train_dataset)) + '...')
#     # print(torch.sum(1 * (wr <= ratio))) # if less than 0.2, return true 1, else return false 0
#     # cnt = cnt + torch.sum(1 * (wr <= ratio))

# for (i, (he, npz)) in enumerate(test_loader): 
#     print("Saving filtered images and counts: " + str((i + 1) * 32) + "/" + str(len(test_dataset)) + '...')















# print(ratio, cnt)  # cnt, the image numbers should be saved
# white_ratio = np.count_nonzero(X_mask * 255) / float(X_mask.size)
# print(X_mask.size,  white_ratio)


    ### synchronization



# def is_image_mostly_white_pixels(path, white_threshold, white_pixel_count_threshold, saved_img_paths, discarded_img_paths):
#     saved_folder = "saved/"
#     discarded_folder = "discarded/"
#     for i in path:
#         img_name= i.split("/")[-1]
#         rgb = PIL.Image.open(i)
#         image = PIL.Image.open(i).convert('L')
#         pixels = image.getdata()  # get the pixels as a flattened sequence
#         number_of_white_pixels = 0
#         for pixel in pixels:
#             if pixel > white_threshold:
#                 number_of_white_pixels += 1
#         n = len(pixels)
#         if (number_of_white_pixels / float(n)) > white_pixel_count_threshold:  # the ratio of white background
#             discarded_img_paths.append(i)
#             rgb.save(discarded_folder + img_name)
#         else:
#             saved_img_paths.append(i)
#             rgb.save(saved_folder + img_name)



# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def calculate_white_ratio(path):
    
#     img = cv2.imread(path,0)
#     blur = cv2.GaussianBlur(img,(5,5),0)
#     ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     number_of_white_pixels = 0
#     for pixel in th3.flatten():
#         if pixel == 255:
#             number_of_white_pixels += 1
#     n = len(th3.flatten())
#     white_ratio = number_of_white_pixels / float(n)
#     return white_ratio

# def kmeans_cluster(path): 

#     org = cv2.imread(path)
#     lab_image = cv2.cvtColor(org, cv2.COLOR_BGR2LAB)
#     flatten_img = lab_image.reshape(-1, 3)
#     ab_img = flatten_img[:, 1:]/255
#     kmeans = cluster.MiniBatchKMeans(n_clusters = 3, random_state=5489, n_init=1000)
#     img_clusters = kmeans.fit_predict(ab_img)
#     img_clusters[img_clusters == 1] = 0
#     img_clusters[img_clusters == 2] = 1
#     white_ratio = np.count_nonzero(img_clusters) / float(img_clusters.size)
    
#     return white_ratio



# path = sorted(glob.glob("data/image_filtered/*/*/*/*.jpg")) 

# saved_img_paths = []
# discarded_img_paths = []

# saved_folder = "saved/"
# discarded_folder = "discarded/"


# pathlib.Path(saved_folder).mkdir(parents=True, exist_ok=True)
# pathlib.Path(discarded_folder).mkdir(parents=True, exist_ok=True)


# for i in path:

#     img_name= i.split("/")[-1]
#     rgb = PIL.Image.open(i)
#     white_ratio = kmeans_cluster(i)

#     if white_ratio >= 0.2:
#         discarded_img_paths.append(i)
#         rgb.save(discarded_folder + img_name)
#     else:
#         saved_img_paths.append(i)
#         rgb.save(saved_folder + img_name)

# print(len(saved_img_paths), len(discarded_img_paths))






# # is_image_mostly_white_pixels(path, 220, 0.2, saved_img_paths, discarded_img_paths)




# quantileRGB = 80


# ### count mean RGB values ###
# mean_value_list = list()
# for i in path:
#     #print(i)
#     I = cv2.imread(i)
#     #print(I.shape)
#     total_value = np.sum(I[:,:,0]) + np.sum(I[:,:,1]) + np.sum(I[:,:,2])
#     total_value = total_value / (I.shape[0] * I.shape[1]) / 3
#     #print(total_value)
#     mean_value_list.append(total_value)

# mean_RGB = [round(f, 4) for f in mean_value_list]
# # print("cluster_pos_df: "+str(cluster_pos_df.shape))
# # print(cluster_pos_df.head())

# ## Define threshold
# c_array = np.percentile(mean_value_list, q=[quantileRGB])

# white_th = 255 if quantileRGB == 100 else c_array[0]

# print("white_th: "+str(white_th))

# pixel_th_white = I.shape[0] * I.shape[1] * 0.5
# print("pixel_th_white: "+str(pixel_th_white))

# saved_img_paths = []
# discarded_img_paths = []

# pathlib.Path(saved_folder).mkdir(parents=True, exist_ok=True)
# pathlib.Path(discarded_folder).mkdir(parents=True, exist_ok=True)
# ### load Image
# for i in path:
#     #print(i)
#     I = cv2.imread(i)
#     img_name= i.split("/")[-1]
#     #print(I.shape)
#     ### color threshold (white)
#     count_white = sum(np.logical_and.reduce((I[:,:,0] > white_th, I[:,:,1] > white_th, I[:,:,2] > white_th)))

#     if sum(count_white) > pixel_th_white:
#         discarded_img_paths.append(i)
#         cv2.imwrite(discarded_folder + img_name, I)
#     else:
#         saved_img_paths.append(i)
#         cv2.imwrite(saved_folder + img_name, I)

# print(len(path))      
# print(len(discarded_img_paths))





