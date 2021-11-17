import torch
import os
import numpy as np
import pathlib
from model import metric 
from sklearn import metrics

### Training Loop ###  
def fit(model, train_loader, optim, criterion, args, device):
    print('-' * 10)
    print('Training:')
    model.train()
    total_loss = 0 #
    epoch_count = []
    epoch_preds = []
    for (i, (X, y, c, ind, pat, s, pix)) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)

        epoch_count.append(y.cpu().detach().numpy())
        epoch_preds.append(pred.cpu().detach().numpy())

        optim.zero_grad()
        loss = criterion(pred, y)  # batch-gene average
        total_loss += loss.cpu().detach().numpy()

        # aMAE = metrics.mean_absolute_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        # aRMSE = metrics.mean_squared_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), squared=False)
        # aCC = metric.average_correlation_coefficient(pred.cpu().detach().numpy(), y.cpu().detach().numpy())
        # # aR2 = metrics.r2_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        # print("{:3d} / {:3d}: Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(i + 1, len(train_loader), loss, aMAE, aRMSE, aCC)) # loop each batch, print averaged batch-gene loss
        
        if args.debug and i==3:
            break

        loss.backward()
        optim.step()   # for each batch, calculate grads and update the params

    total_loss /= len(train_loader)
    epoch_count = np.concatenate(epoch_count)
    epoch_preds = np.concatenate(epoch_preds)

    total_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
    total_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
    total_aCC = metric.average_correlation_coefficient(epoch_preds, epoch_count)
    ## total_aR2 = metrics.r2_score(epoch_count, epoch_preds)

    print("Train Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(total_loss, total_aMAE, total_aRMSE, total_aCC))
    
    return total_loss, total_aMAE, total_aRMSE, total_aCC

      
def validate(model, val_loader, criterion, args, device):
    print('Validate:')
    model.eval()
    total_loss = 0 #
    epoch_count = []
    epoch_preds = []
    with torch.no_grad():
        for (i, (X, y, c, ind, pat, s, pix)) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            epoch_count.append(y.cpu().detach().numpy())
            epoch_preds.append(pred.cpu().detach().numpy())
            loss = criterion(pred, y)  # batch-gene average
            total_loss += loss.cpu().detach().numpy()

            # aMAE = metrics.mean_absolute_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
            # aRMSE = metrics.mean_squared_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), squared=False)
            # aCC = metric.average_correlation_coefficient(pred.cpu().detach().numpy(), y.cpu().detach().numpy())
            # print("{:3d} / {:3d}: Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(i + 1, len(val_loader), loss, aMAE, aRMSE, aCC))

            if args.debug and i==3:
                break

        total_loss /= len(val_loader)
        epoch_count = np.concatenate(epoch_count)
        epoch_preds = np.concatenate(epoch_preds)

        total_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
        total_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
        total_aCC = metric.average_correlation_coefficient(epoch_preds, epoch_count)

        print("Val Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(total_loss, total_aMAE, total_aRMSE, total_aCC))

    return total_loss, total_aMAE, total_aRMSE, total_aCC

     
def test(model, test_loader, criterion, device, args, epoch):
    print('Test:')
    model.eval()
    total_loss = 0 #
    epoch_count = []
    epoch_preds = []
    
    patient = []
    section = []
    coord = []
    pixel = []

    with torch.no_grad():
        for (i, (X, y, c, ind, pat, s, pix)) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            epoch_count.append(y.cpu().detach().numpy())
            epoch_preds.append(pred.cpu().detach().numpy())

            loss = criterion(pred, y)  # batch-gene average
            total_loss += loss.cpu().detach().numpy()
                
            # aMAE = metrics.mean_absolute_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
            # aRMSE = metrics.mean_squared_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy(), squared=False)
            # aCC = metric.average_correlation_coefficient(pred.cpu().detach().numpy(), y.cpu().detach().numpy())
            # print("{:3d} / {:3d}: Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(i + 1, len(test_loader), loss, aMAE, aRMSE, aCC))
            
            if args.debug and i==3:
                break

            patient += pat
            section += s
            coord.append(c.detach().numpy())
            pixel.append(pix.detach().numpy())

        total_loss /= len(test_loader)
        epoch_count = np.concatenate(epoch_count)
        epoch_preds = np.concatenate(epoch_preds)

        total_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
        total_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
        total_aCC = metric.average_correlation_coefficient(epoch_preds, epoch_count)

        print("Test Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(total_loss, total_aMAE, total_aRMSE, total_aCC))
        
        coord = np.concatenate(coord)
        pixel = np.concatenate(pixel)

        if args.pred_root:
            pathlib.Path(os.path.dirname(args.pred_root + '/')).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.pred_root + "/epoch_" + str(epoch),  #stopped epoch + 1 - patience 
                                task="gene",
                                counts=epoch_count,
                                predictions=epoch_preds,
                                coord=coord,
                                patient=patient,
                                section=section,
                                pixel=pixel,
                                ensg_names=test_loader.dataset.ensg_keep,
                                gene_names=test_loader.dataset.gene_keep,
                                )

    return total_loss, total_aMAE, total_aRMSE, total_aCC

