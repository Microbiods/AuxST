 ### Get mean and std

import utils
import torchvision
import torch
import numpy as np
from model import trainer, pipeline



def get_cv_resluts(patients, cv_folds, args, device):

    fold = [patients[f::cv_folds] for f in range(cv_folds)]
    
    total_loss = []
    total_aMAE = []
    total_aRMSE = []
    total_aCC = []
    print("### START CROSS VALIDATION:")
    print()
    for f in range(cv_folds):
        print("Fold ##{}".format(f))
        print('=' * 10)
        train = [fold[i] for i in range(cv_folds) if i != f]
        train = [i for j in train for i in j]
        test = fold[f]
        print("Train patients: ", train)
        print("Test patients: ", test)
        test_loss, test_aMAE, test_aRMSE, test_aCC = train_cv_folds(train, test, args, device)

        total_loss.append(test_loss)
        total_aMAE.append(test_aMAE)
        total_aRMSE.append(test_aRMSE)
        total_aCC.append(test_aCC)
        print()

    # print(total_loss)
    # print()
    # print(total_aMAE) 
    # print()
    # print(total_aRMSE)
    # print() 
    # print(total_aCC)
    # print()

    best_loss_epoch = np.argmin(np.vstack(total_loss).mean(0))
    best_aMAE_epoch = np.argmin(np.vstack(total_aMAE).mean(0))
    best_aRMSE_epoch  = np.argmin(np.vstack(total_aRMSE).mean(0))
    best_aCC_epoch = np.argmax(np.vstack(total_aCC).mean(0))


    # print(best_loss_epoch, best_aMAE_epoch, best_aRMSE_epoch, best_aCC_epoch)

    return best_loss_epoch, best_aMAE_epoch, best_aRMSE_epoch, best_aCC_epoch





def train_cv_folds(train_patients, test_patients, args, device):
    
    total_loss = []
    total_aMAE = []
    total_aRMSE = []
    total_aCC = []

    model, train_loader, test_loader, optim, lr_scheduler, criterion = pipeline.setup(train_patients, test_patients, args, device, cv = True)

    for epoch in range(args.cv_epochs):
        if args.debug and epoch==3:
                break
        print("Epoch #" + str(epoch + 1) + ":")
        train_loss, train_aMAE, train_aRMSE, train_aCC = trainer.fit(model, train_loader, optim, criterion, args, device)
        lr_scheduler.step()
        test_loss, test_aMAE, test_aRMSE, test_aCC = trainer.validate(model, test_loader, criterion, args, device)
        total_loss.append(test_loss)
        total_aMAE.append(test_aMAE)
        total_aRMSE.append(test_aRMSE)
        total_aCC.append(test_aCC)

        torch.cuda.empty_cache()

    return total_loss, total_aMAE, total_aRMSE, total_aCC