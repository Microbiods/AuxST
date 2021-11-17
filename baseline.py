import torch
import random
import argparse
import math
import torchvision
import utils
import pathlib
import numpy as np
import utils.cross_validate as CV
from model import trainer, pipeline


# from pytorch_pretrained_vit import ViT
# import pathlib
# import os
# from torch.utils.data.sampler import SubsetRandomSampler
# import model.metric as metric 


# main program
def run_spatial(args=None):

        ### Seed ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### Split patients into train/test ###
        patients = sorted(utils.util.get_spatial_patients().keys())
        test_patients = ["BC23450", "BC23903"]
        train_patients = [p for p in patients if p not in test_patients]

        print("Train patients: ",  train_patients)
        print("Test patients: ", test_patients)
        print("Parameters: ", args)
        print()
        
        ###   cross-validation to get the best epoch

        best_loss_epoch, best_aMAE_epoch, best_aRMSE_epoch, best_aCC_epoch = CV.get_cv_resluts(train_patients, args.cv_fold, args, device)

        # print("Best loss epoch is: ", best_loss_epoch)
        # print("Best aMAE epoch is: ", best_aMAE_epoch) 
        # print("Best aRMSE epoch is: ", best_aRMSE_epoch) 
        # print("Best aCC epoch is: ", best_aCC_epoch)
        # print()
        best_epoch = math.ceil(np.mean(np.array(([best_loss_epoch, best_aMAE_epoch, best_aRMSE_epoch, best_aCC_epoch]))))
        print("Best CV epoch: ", best_epoch)
        print()

        # best_epoch = 50

        ###     cross-validation to get the best epoch

        print("### START MAIN PROGRAM:")
        print()
        print("Train patients: ",  train_patients)
        print("Test patients: ", test_patients)

        ### main network
        model, train_loader, test_loader, optim, lr_scheduler, criterion = pipeline.setup(train_patients, test_patients, args, device)

        if best_epoch <= 3:  # for debug
            best_epoch = 3

        for epoch in range(best_epoch):

            if args.debug and epoch==3:
                break

            print("Epoch #" + str(epoch + 1) + ":")
            train_loss = trainer.fit(model, train_loader, optim, criterion, args, device)
            lr_scheduler.step()
            # here test will save should change the file name
            test_loss = trainer.test(model, test_loader, criterion, device, args, best_epoch)
        
        ### TODO: best_epoch = 0, skip the for loop and direct save model?
        # if args.debug:
        #     pass
        # else:
        torch.save(model, args.pred_root + '/model.pkl') 
        print()                
            # 
        # # model.save (the best model should be the current epoch - patience)
        # # relative


            # lr_scheduler(val_loss)
            # early_stopping(val_loss)
            # if early_stopping.early_stop: # when break, the model is at the final epoch 
            #     break
        

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Process the paths.')

    parser.add_argument('--seed', type=int, default=0, help='seed for reproduction')

    parser.add_argument("--cv_fold", type=int, default=5, help="cv fold for cross-validation") 
                                    
    parser.add_argument("--batch", type=int, default=32, help="training batch size")  

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")

    parser.add_argument("--cv_epochs", type=int, default=50, help="number of cross-validation epochs")

    parser.add_argument("--workers", type=int, default=8, help="number of workers for dataloader")

    parser.add_argument("--window", type=int, default=224, help="window size") # try 128 150 224 299 512 (smaller, normal, and bigger)

    parser.add_argument("--model", type=str, default= 'densenet121', help="choose different model")   # alexnet, vgg16, resnet101, densenet121, inception_v3, efficientnet_b7

    parser.add_argument("--pretrained", action="store_false", help="use ImageNet pretrained model?")

    parser.add_argument("--finetuning", type=str, default= None, help="use ImageNet pretrained model with fine tuning fcs?")

    parser.add_argument("--gene_filter",  default=250, type =int,
                        help="specific predicted main genes (defalt use all the rest for aux tasks)")

    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

    parser.add_argument("--pred_root", type=str, default="output/", help="root for prediction outputs")

    parser.add_argument("--debug", action="store_true", help="debug")

    args = parser.parse_args()

    # set different log name 

    # pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)

    run_spatial(args)

