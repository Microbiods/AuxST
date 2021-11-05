import torch
import time
import numpy as np
import pathlib

def get_mean_and_std(loader, args):

    # cached = "data/normalize/"
    # # print(cached_image)

    # if pathlib.Path(cached).exists():
    #     mean = np.load(cached + 'img_mean.npy')
    #     std = np.load(cached + 'img_std.npy')
    #     mean_count = np.load(cached + 'count_mean.npy')
    #     std_count = np.load(cached + 'count_std.npy')

    # else:  
        
    t = time.time()
    mean = 0.
    std = 0.
    nb_samples = 0
    epoch_count = []

    for (i, (X, y, *_)) in enumerate(loader):  # here may conduct all the cutting work but only for training loader


        if args.debug and i==3:
            break

        batch_samples = X.size(0)  # [32, 3, 224, 224]
        X = X.view(batch_samples, X.size(1), -1)  #[32, 3, 50176]
        mean += X.mean(2).sum(0)   # [32, 3] -> [3]
        std += X.std(2).sum(0)
        nb_samples += batch_samples
        epoch_count.append(y)

    mean /= nb_samples
    std /= nb_samples
    epoch_count = torch.cat(epoch_count, dim=0)
    # print(epoch_count.shape)  # [28315, 250]
    mean_count = epoch_count.mean(0)
    std_count = epoch_count.std(0)
    
    # print("Computing mean and std of gene expressions, estimating mean (" + str(mean) + ") and std (" + str(std) )
    print("Computing mean and std of gene expressions, estimating mean (" + str(mean) + ") and std (" + str(std) + " took {:.4f}s" .format(time.time() - t))
    print()
    ###
        # pathlib.Path(cached).mkdir(parents=True, exist_ok=True)
        # np.save(cached + 'img_mean.npy', mean)
        # np.save(cached + 'img_std.npy', std)
        # np.save(cached + 'count_mean.npy', mean_count)
        # np.save(cached + 'count_std.npy', std_count)

    return mean.tolist(), std.tolist(), mean_count, std_count





