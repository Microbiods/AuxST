import glob
import collections
import torch

def get_spatial_patients():
    """
    Returns a dict of patients to sections.
    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/image_raw/*/*/*_*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient

def patient_or_section(name):
        if "_" in name:
            return tuple(name.split("_"))
        return name

def get_sections(patients, testpatients):
    train_patients = []
    test_patients = []
    for (i, p) in enumerate(patients):
        for s in patients[p]:
            if p in testpatients :
                test_patients.append((p, s))
            else:
                train_patients.append((p, s))

    print('Train patients: ',train_patients)
    print('Test patients: ', test_patients)

    return train_patients, test_patients

def cv_split(patients, cv_folds):
    fold = [patients[f::cv_folds] for f in range(cv_folds)]
    for f in range(cv_folds):
        print("Fold #{}".format(f))
        train = [fold[i] for i in range(5) if i != f]
        train = [i for j in train for i in j]
        test = fold[f]
    return train, test



class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):   # only if there is one better than others in patience epoch, stop, model.checkpoint can save the best
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
