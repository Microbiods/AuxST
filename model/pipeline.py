import torchvision
import torch
import utils
from model import net


def setup(train_patients, test_patients, args, device, cv = False):
    
    ### Get mean and std
    train_dataset = utils.dataloader.Spatial(train_patients, 
                        count_root='training/counts/',
                        img_root='training/images/',
                        window=args.window, resolution = args.resolution,
                        gene_filter=args.gene_filter,
                            transform=torchvision.transforms.ToTensor()) # range [0, 255] -> [0.0,1.0]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                            num_workers=args.workers, shuffle=True)

    mean, std, count_mean, count_std = \
                        utils.normalize.get_mean_and_std(train_loader, args)


    ### Train transform
    train_transform = torchvision.transforms.Compose([
                    # torchvision.transforms.Resize((224, 224)), # can resize if the input is not 224
                    torchvision.transforms.RandomHorizontalFlip(), # default p = 0.5
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=mean, std=std)])                 
    ### Train data loader
    train_dataset = utils.dataloader.Spatial(train_patients,
                        count_root='training/counts/',
                        img_root='training/images/', 
                        window=args.window, resolution = args.resolution,
                        gene_filter=args.gene_filter,
                            transform=train_transform,normalization = [count_mean, count_std]) # range [0, 255] -> [0.0,1.0]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, 
                            num_workers=args.workers, shuffle=True)
    


    ### Val / test transform
    val_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=mean, std=std)])
    ### Test data loader

    if cv:

        test_dataset = utils.dataloader.Spatial(test_patients,
                                count_root='training/counts/',
                                img_root='training/images/',
                                window=args.window, resolution = args.resolution,
                                gene_filter=args.gene_filter, 
                                transform = val_transform, normalization = [count_mean, count_std])

    else: 

        test_dataset = utils.dataloader.Spatial(test_patients,
                                count_root='test/counts/',
                                img_root='test/images/',
                                window=args.window, resolution = args.resolution,
                                gene_filter=args.gene_filter, 
                                transform = val_transform, normalization = [count_mean, count_std])


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, 
                                num_workers=args.workers, shuffle=True)


    ### Model setup

    architecture = net.set_models(args.model, args.pretrained) # get the architecture and return imageNet weights?

    # set pretrained or fine-tuning, default is to initialize all the layers, only train fc? or based on all layers?

    model = net.set_out_features(architecture, args.gene_filter)  # get the model with the best params

    # default is random initialized

    # print(args.finetuning)
    # print(args.finetuning == 'ftfc')

    # print(args.model)
    # print(args.model == 'efficientnet_b4')

    if args.model == 'efficientnet_b4':

        if args.finetuning == 'ftfc':  # fine tuning only classifier


            for param in model.parameters():
                param.requires_grad = False
        
            for cnt, child in enumerate(model.children()):
                # print(cnt, child)
                if cnt >= 7:
                    for param in child.parameters():
                        param.requires_grad = True


        elif args.finetuning == 'ftconv':  # fine tuning half of the MBConvBlocks

            for param in model.parameters():
                param.requires_grad = False

            for param in model._blocks[15:].parameters():
                param.requires_grad = True

            for cnt, child in enumerate(model.children()):
                # print(cnt, child)
                if cnt >= 3:
                    for param in child.parameters():
                        param.requires_grad = True


        elif args.finetuning == 'ftall':  # fine tuning all the layers
            for param in model.parameters():
                param.requires_grad = True


    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    model = torch.nn.DataParallel(model)
    model.to(device)


    criterion = torch.nn.MSELoss()
    # optim = torch.optim.__dict__['Adam'](model.parameters(), lr=3e-4, weight_decay = 1e-6)  # here need to be revised
    
    optim = torch.optim.__dict__['SGD'](model.parameters(), lr = args.learning_rate,
    #                             momentum=0.9)  # here need to be revised
                                momentum=0.9, weight_decay = 1e-6)  # here need to be revised

    # optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
    # lr_scheduler = utils.util.LRScheduler(optim)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=5)
    
    # # early_stopping = utils.util.EarlyStopping()


    return model, train_loader, test_loader, optim, lr_scheduler, criterion



