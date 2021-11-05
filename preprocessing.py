import os
import numpy as np
import pickle
import logging
import pathlib
import time
import glob
import collections
import tqdm
import argparse
import PIL
import skimage
import shutil

PIL.Image.MAX_IMAGE_PIXELS = 1000000000    

# generate the count file for single spot, will generate 2 folders with the names count_raw/ and count_filtered/, generally, we just need the latter one (after flitering).

def spatial(args):

    window = 512  # only to check if patch is off of boundary
    # if bigger than the size in the next program, use the smallest and then resize
 
    logger = logging.getLogger(__name__)

    pathlib.Path(args.process).mkdir(parents=True, exist_ok=True)

    # return data, subtype (LA, LB, HER2, TNBC)
    raw, subtype = load_raw(args.root)
    # {'BC24220': 'HER2_luminal', 'BC23901': 'HER2_luminal', 'BC23450': 'HER2_luminal', 'BC23287': 'HER2_luminal', 'BC23944': 'HER2_luminal', 'BC23903': 'HER2_non_luminal', 'BC23810': 'HER2_non_luminal', 'BC24105': 'HER2_non_luminal', 'BC24044': 'HER2_non_luminal', 'BC23567': 'HER2_non_luminal', 'BC23269': 'Luminal_A', 'BC23268': 'Luminal_A', 'BC23272': 'Luminal_A', 'BC24223': 'Luminal_A', 'BC23377': 'TNBC', 'BC23288': 'TNBC', 'BC23209': 'TNBC', 'BC23803': 'TNBC', 'BC23506': 'Luminal_B', 'BC23508': 'Luminal_B', 'BC23277': 'Luminal_B', 'BC23270': 'Luminal_B', 'BC23895': 'Luminal_B'}
    #data/count_raw/
    with open(args.process + "subtype.pkl", "wb") as f:   # 1. save subtype
        pickle.dump(subtype, f)

    t = time.time()
    t0 = time.time()

    section_header = None
    gene_names = set()

    for patient in raw:
        for section in raw[patient]:
            section_header = raw[patient][section]["count"].columns.values[0]  # Unnamed: 0
            # TODO: here we can change to intersection
            gene_names = gene_names.union(set(raw[patient][section]["count"].columns.values[1:]))
    gene_names = list(gene_names) # without header
    gene_names.sort()  # sort by name
    with open(args.process + "gene.pkl", "wb") as f:     # 2. save all sorted gene names 
        pickle.dump(gene_names, f)
    gene_noheader = gene_names
    gene_names = [section_header] + gene_names

    print("Finding list of genes: " + str(time.time() - t0))

    for (i, patient) in enumerate(raw):
        print("Processing " + str(i + 1) + " / " + str(len(raw)) + ": " + patient)
        pathlib.Path("{}{}/{}".format(args.process, subtype[patient], patient)).mkdir(parents=True, exist_ok=True)
        for section in raw[patient]:
            print("Processing " + patient + " " + section + "...")
            # In the original data, genes with no expression in a section are dropped from the table.
            # This adds the columns back in so that comparisons across the sections can be done.
            t0 = time.time()
            missing = list(set(gene_names) - set(raw[patient][section]["count"].keys())) # missing names without header
            c = raw[patient][section]["count"].values[:, 1:].astype(float)  # without header
            pad = np.zeros((c.shape[0], len(missing)))
            c = np.concatenate((c, pad), axis=1)
            names = np.concatenate((raw[patient][section]["count"].keys().values[1:], np.array(missing)))  # raw name + missing name
            
            c = c[:, np.argsort(names)]  # return index and sort (no header)
            print("Adding zeros and ordering columns: " + str(time.time() - t0))

            t0 = time.time()
            count = {}      # return complete count with no header (with gene name, index and coord id)
            for (j, row) in raw[patient][section]["count"].iterrows():  # 0, 3*34
                count[row.values[0]] = c[j, :]   
            print("Extracting counts: " + str(time.time() - t0))

            t0 = time.time()
            image = skimage.io.imread(raw[patient][section]["image"])

            # height * width * channel
            print("Loading image: " + str(time.time() - t0))

            for (_, row) in raw[patient][section]["spot"].iterrows():

                x = round(float(row[0].split(',')[1]))   # coord: float (4622, 4621)
                y = round(float(row[0].split(',')[2]))

                spot_x = row[0].split(",")[0].split('x')[0] # spot id 11x17  str
                spot_y = row[0].split(",")[0].split('x')[1] 

                # resize to avoid the errors, negative dimensions are not allowed therefore here we make sure all the selected images are not exceeded 
                # data preprocessing #1
                # We predict the gene expression by images, therefore we must need images
                if (x + (-window // 2))>= 0 and (x + (window // 2)) <= image.shape[1] and (y + (-window // 2))>= 0 and (y + (window // 2)) <= image.shape[0]:
                    
                    if (spot_x + "x" + spot_y) in list(count.keys()) :
                    # data preprocessing #2
                    # make sure the sopt in the coord file have the GE data in count file
                        # data preprocessing #3
                        # make sure the selected spots with at least 1000 total read counts    
                        if np.sum(count[spot_x + "x" + spot_y]) >= 1000: 
                            # save for each spot
                            filename = "{}{}/{}/{}_{}_{}.npz".format(args.process, subtype[patient], patient, section,
                                                                    spot_x, spot_y)
                            np.savez_compressed(filename, 
                                                count=count[spot_x + "x" + spot_y],
                                                pixel=np.array([x, y]),
                                                patient=np.array([patient]),
                                                section=np.array([section]),
                                                index=np.array([int(spot_x), int(spot_y)]))
                        else:
                            logger.warning("Total counts of Patch " + str(spot_x) + "x" + str(spot_y) + " in " + patient + " " + section + " less than 1000")
                    else:
                        logger.warning("Patch " + str(spot_x) + "x" + str(spot_y) + " not found in " + patient + " " + section)
                else:
                    logger.warning("Detected " + patient +  " " + section + " " + str(spot_x) + "x" + str(spot_y) + " too close to edge.")

            print("Saving patches: " + str(time.time() - t0))

    print("Preprocessing took " + str(time.time() - t) + " seconds")

    # save the mean gene expression
    
    logging.info("Computing statistics of dataset")   # only compute the previous keeped genes
    gene = []           # data/count_raw/
    for filename in tqdm.tqdm(glob.glob("{}*/*/*_*_*.npz".format(args.process))):
        npz = np.load(filename)
        count = npz["count"]   # (26933,)
        gene.append(np.expand_dims(count, 1))
    gene = np.concatenate(gene, 1)  #（26933, 30625）# in total 

    print( "There are {} genes and {} spots left before filtering.".format(gene.shape[0], gene.shape[1]))

    np.save(args.process + "mean_expression.npy", np.mean(gene, 1))

    # data preprocessing #4 
    # delete all gene with 0 mean expression across spots
    # keep the gene expressed more than 10% (30625 * 10%) of the spots across samples (filtered out genes that are expressed in less than 10% of the array spots across all the samples )
    # fullfilled both conditions at the same time (actually the second includes the first)
    # second: sum more than (30625 * 10%) and for each, count more than 1
    filter = np.where(np.sum(np.where(gene > 0, 1, 0), 1) >= 0.1 * gene.shape[1], True, False)
    # update all the files
    # print(gene[filter].shape)  (5943, 30625)

    pathlib.Path(args.filter).mkdir(parents=True, exist_ok=True)
    with open(args.filter + "subtype.pkl", "wb") as f:   # 1. subtype
        pickle.dump(subtype, f)
    with open(args.filter + "gene.pkl", "wb") as f:     # 2. gene names
        gene_names = list(np.array(gene_noheader)[filter])
        pickle.dump(gene_names, f)

    # 3. counts                              
    for filename in tqdm.tqdm(glob.glob("{}*/*/*_*_*.npz".format(args.process))):
                                        # count_filter / Luminal_B/  BC23895/  D1_11_20.npz
        new_path = "{}{}/{}/".format(args.filter, filename.split('/')[2], filename.split('/')[3])
        pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
        npz = np.load(filename)
        new_filename = filename.replace(args.process, args.filter)
        np.savez_compressed(new_filename, 
                            count = npz["count"][filter],                                
                            pixel= npz["pixel"],
                            patient=npz["patient"],
                            section=npz["section"],
                            index= npz["index"])

    # 4. mean
    logging.info("Computing statistics of dataset")   # only compute the saved genes
    gene = []
    for filename in tqdm.tqdm(glob.glob("{}*/*/*_*_*.npz".format(args.filter))):
        npz = np.load(filename)
        count = npz["count"]
        gene.append(np.expand_dims(count, 1))

    gene = np.concatenate(gene, 1)  
    print( "There are {} genes and {} spots left after filtering.".format(gene.shape[0], gene.shape[1]))
    np.save(args.filter + "mean_expression.npy", np.mean(gene, 1))
    
    # delete the temp (processed folder), can be ignored
    # shutil.rmtree(args.process)    

def newer_than(file1, file2):
    """
    Returns True if file1 is newer than file2.
    A typical use case is if file2 is generated using file1.
    For example:

    if newer_than(file1, file2):
        # update file2 based on file1
    """
    return os.path.isfile(file1) and (not os.path.isfile(file2) or os.path.getctime(file1) > os.path.getctime(file2))

def load_section(root: str, patient: str, section: str, subtype: str):
    """
    Loads data for one section of a patient.
    """
    import pandas
    import gzip
    # data/STBC/ HER2_luminal/BC23450/BC23450_D2
    file_root = root + subtype + "/" + patient + "/" + patient + "_" + section
    # image = skimage.io.imread(file_root + ".jpg")
    image = file_root + ".jpg"
    ##### save counts and spot coords to pkl
    with gzip.open(file_root + ".tsv.gz", "rb") as f:
        count = pandas.read_csv(f, sep="\t")
    spot = pandas.read_csv(file_root + ".spots.gz", sep="\t")
    return {"image": image, "count": count, "spot": spot}

def load_raw(root: str):
    """
    Loads data for all patients.
    """
    # Wildcard search for patients/sections
    # data/STBC/ HER2_luminal/BC23450/BC23450_D2.jpg
    images = glob.glob(root + "*/*/*_*.jpg")
    #   file_root = root + subtype + "/" + patient + "/" + patient + "_" + section
    # Dict mapping patient ID (str) to a list of all sections available for the patient (List[str])  sections: C/D
    patient = collections.defaultdict(list)
    for (p, s) in map(lambda x: x.split("/")[-1][:-4].split("_"), images):
        patient[p].append(s)

    # Dict mapping patient ID (str) to subtype (str)
    subtype = {} #  HER2_luminal, BC23450
    for (st, p) in map(lambda x: (x.split("/")[2], x.split("/")[3]), images):
            subtype[p] = st

    print("Loading raw data...")
    t = time.time()
    data = {}

    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            data[p] = {}
            for s in patient[p]:
                data[p][s] = load_section(root, p, s, subtype[p])
                pbar.update()
    print("Loading raw data took " + str(time.time() - t) + " seconds.")

    return data, subtype


parser = argparse.ArgumentParser(description='Generate the necessary files.')

parser.add_argument('--root',  type=str, default='data/STBC/',
                    help='Path for the raw dataset')     
parser.add_argument('--process',  type=str, default='data/count_raw/',
                    help='Path for the generated files')
parser.add_argument('--filter',  type=str, default='data/count_filtered/',
                    help='Path for the filtered dataset')

args = parser.parse_args()

spatial(args)

# use jpg not tif
'''
There are some standards:
1. choose the gene with the mean expression more than 0
2. filtered out genes that are expressed in less than 10% of the all array spots across all the samples
3. selected spots with at least ten total read counts, delete the spot near the edge
'''