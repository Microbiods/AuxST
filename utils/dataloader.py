import torch
import glob
import utils
import pathlib
import PIL
import pickle
import openslide
import collections
import shutil
import torchvision
import numpy as np
PIL.Image.MAX_IMAGE_PIXELS = 1000000000 


# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class Generator(torch.utils.data.Dataset):  # for baseline 

    def __init__(self,
                 patient=None,
                 test_mode = None,
                 window = 224,  
                 count_root='data/count_filtered/',
                 img_root='data/image_stained/',
                 count_cached = None,
                 img_cached = None,
                 transform=None,
                 ):
        
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))  # the npz collection for each count file
        # print(len(self.dataset)) 30625

        if patient is not None:
            # Can specify (patient, section) or only patient (take all sections)
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]
    
        self.test_mode = test_mode
        self.window = window

        self.count_root = count_root
        self.img_root = img_root

        self.count_cached = count_cached
        self.img_cached = img_cached

        self.transform = transform

         # read subtype: HER2_non_luminal
        with open(self.count_root + "subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)
     
        self.slide = collections.defaultdict(dict)
        self.slide_mask = collections.defaultdict(dict)
        # read every tif (can be parallelized)
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            #  HER2_non_luminal/BC23810/BC23810_D2.tif
            self.slide[patient][section] = openslide.open_slide("{}{}/{}/{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))
            self.slide_mask[patient][section]  = openslide.open_slide("{}{}/{}/{}_{}_mask.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))
            # get the slide for each patient/section (slide is a collection)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = np.load(self.dataset[index])
        count   = npz["count"]
        pixel   = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"]  # 11*17

        slide = self.slide[patient][section]
        slide_mask = self.slide_mask[patient][section]
        X = slide.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))  # return PIL.Image (Return an RGBA Image containing the contents of the specified region)
        X = X.convert("RGB") # may need create folder?
        X_mask = slide_mask.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
        X_mask = X_mask.convert("1")  # binary image
        
        he = X
        he_mask = X_mask
        X = self.transform(X)
        X_mask = self.transform(X_mask)

        # print(X_mask * 255)

        cached_count = "{}{}/{}/{}_{}_{}.npz".format(self.count_cached, self.subtype[patient], patient, section, coord[0], coord[1])
        cached_image = "{}{}/{}/{}/{}_{}_{}.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, section, coord[0], coord[1])
        # cached_image_mask = "{}{}/{}/{}/{}_{}_{}_mask.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, section, coord[0], coord[1])

        pathlib.Path(cached_count.strip(cached_count.split('/')[-1])).mkdir(parents=True, exist_ok=True)
        pathlib.Path(cached_image.strip(cached_image.split('/')[-1])).mkdir(parents=True, exist_ok=True)

        if self.test_mode == None:  # not test data

            white_ratio = torch.count_nonzero(X_mask * 255) / float(torch.numel(X_mask))  # the ratio of white pixels (255)
        
            if white_ratio < 0.5:  #  save the images with wr <= 0.5

                # print(type(npz))

                shutil.copy(self.dataset[index],cached_count)
                he.save(cached_image)

                # he_mask.save(cached_image_mask)

        else: # for test data

            shutil.copy(self.dataset[index],cached_count)
            he.save(cached_image)
        
        return X, count






class SubGenerator(torch.utils.data.Dataset):  # for ablation 

    def __init__(self,
                 patient=None,
                 window = 224, 
                 resolution = 224, 
                 count_root='data/count_filtered/',
                 img_root='data/image_stained/',
                 img_cached = None,
                 transform=None,
                 ):
        
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))  # the npz collection for each count file
        # print(len(self.dataset)) 30625

        if patient is not None:
            # Can specify (patient, section) or only patient (take all sections)
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]
    
       
        self.window = window
        self.resolution = resolution

        self.count_root = count_root  # subtype path
        self.img_root = img_root  # training path

        self.img_cached = img_cached  # saved path

        self.transform = transform

         # read subtype: HER2_non_luminal
        with open(self.count_root + "subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)
     
        self.slide = collections.defaultdict(dict)
     
        # read every tif (can be parallelized)
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            #  HER2_non_luminal/BC23810/BC23810_D2.tif
            self.slide[patient][section] = openslide.open_slide("{}{}/{}/{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))
            # get the slide for each patient/section (slide is a collection)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = np.load(self.dataset[index])
        count   = npz["count"]
        pixel   = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"]  # 11*17

        slide = self.slide[patient][section]
        X = slide.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))  # return PIL.Image (Return an RGBA Image containing the contents of the specified region)
        X = X.convert("RGB") # may need create folder?
      

        he = X

        ### resize to specific resolutions

        X = self.transform(X)

        # cached_image = "{}{}/{}/{}/{}_{}_{}.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, section, coord[0], coord[1])
        # pathlib.Path(cached_image.strip(cached_image.split('/')[-1])).mkdir(parents=True, exist_ok=True)


        ### only for get different resolutions

        if self.resolution != 224:
            he = torchvision.transforms.Resize((self.resolution, self.resolution))(he)
        cached_image = "{}{}/{}/{}/{}/{}_{}_{}.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, self.resolution, section, coord[0], coord[1])
        pathlib.Path(cached_image.strip(cached_image.split('/')[-1])).mkdir(parents=True, exist_ok=True)
    



        he.save(cached_image)

        return X, count






class Spatial(torch.utils.data.Dataset):

    def __init__(self,
                 patient=None,
                 window = 224,  
                 resolution = 224,
                 count_root=None,
                 img_root=None,
                #  count_root='training/counts/',
                #  img_root='training/images/',
                 gene_filter = 250,
                 aux_ratio = 0.1,
                 transform=None,
                 normalization=None,
                 ):
        
        # data/STBC_filter/ HER2_luminal /BC23287 /  C2_10_25 .npz
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))  # the npz collection for each count file
        # print(len(self.dataset)) 30625
        if patient is not None:
            # Can specify (patient, section) or only patient (take all sections)
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]
    
        self.transform = transform
        self.window = window
        self.resolution = resolution
        self.count_root = count_root
        self.img_root = img_root
        self.gene_filter = gene_filter

        self.aux_ratio = aux_ratio

        self.normalization = normalization
      
        # read subtype: HER2_non_luminal
        with open("data/count_filtered/subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)

        # read gene names  
        with open("data/count_filtered/gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)

        # read mean expression (for all, not training set)
        self.mean_expression = np.load('data/count_filtered/mean_expression.npy')

        self.gene_names = list(map(lambda x: utils.ensembl.symbol[x], self.ensg_names))

        # save the top gene names and indexes to get the counts
        keep_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:self.gene_filter]))[1])
        
        self.keep_bool = np.array([i in keep_gene for i in range(len(self.gene_names))])   # set keeped gene index to be true else to be false
        
        ### here not the first 250 or the last 5000+ but the 250 given true or false 

        self.ensg_keep = [n for (n, f) in zip(self.ensg_names, self.keep_bool) if f]
        self.gene_keep = [n for (n, f) in zip(self.gene_names, self.keep_bool) if f]

        ### self.aux_ratio cannot be zero, or it will return out of index
        if self.aux_ratio != 0:
            self.aux_nums = int((len(self.gene_names) - self.gene_filter) * self.aux_ratio)   # from 0 to 1, 0 equals to no aux prediction, 1 means use all for aux
            aux_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][self.gene_filter:self.gene_filter + self.aux_nums]))[1])
            self.aux_bool = np.array([i in aux_gene for i in range(len(self.gene_names))]) 
            self.ensg_aux = [n for (n, f) in zip(self.ensg_names, self.aux_bool) if f]
            self.gene_aux = [n for (n, f) in zip(self.gene_names, self.aux_bool) if f]

            # print(aux_nums,len(aux_gene),len(self.aux_bool),len(self.ensg_aux),len(self.gene_aux))
           
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = np.load(self.dataset[index])
        count   = npz["count"]
        pixel   = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"]       # 11*17
      
        # # could save tif

        if self.resolution != 224:
    
            cached_image = "{}{}/{}/{}/{}/{}_{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, self.window, self.resolution, section, coord[0], coord[1])
            X = PIL.Image.open(cached_image)  # return RGB. Image type

        else:
            
            cached_image = "{}{}/{}/{}/{}_{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, self.window, section, coord[0], coord[1])
            X = PIL.Image.open(cached_image)  # return RGB. Image type

        # TODO: for differnet window size

        if self.transform is not None:
            X = self.transform(X)



        # # ### TODO: to be revised for window size debugging
        if X.shape[1] != 224:
            X = torchvision.transforms.Resize((224, 224))(X)



        # print(X.shape)

        # Z = np.sum(count)  # the sum of the selected genes  (in an image)
        # n = count.shape[0] # how many genes to predict

        # print(count.shape)

        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        keep_count = count[self.keep_bool]
        y = torch.as_tensor(keep_count, dtype=torch.float)
        y = torch.log(1 + y)

        # y = torch.log((1 + y) / (n + Z))
        # y = torch.log( y / Z * 10000 + 1)
        
        ### TODO: different normalization

        if self.normalization is not None:
            y = (y - self.normalization[0]) / self.normalization[1]

        if self.aux_ratio != 0: ### return aux

            aux_count = count[self.aux_bool]
            aux = torch.as_tensor(aux_count, dtype=torch.float)
            aux = torch.log(1 + aux)

            return X, y, aux, coord, index, patient, section, pixel
        
        else:

            return X, y, coord, index, patient, section, pixel
       