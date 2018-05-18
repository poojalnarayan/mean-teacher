import torchvision.transforms as transforms

from . import data
from .utils import export
from torch.utils.data import Dataset
from .processILPdata.family_data import Data as FamilyData

# import data
# from utils import export
# from torch.utils.data import Dataset
# from processILPdata.family_data import Data as FamilyData
@export
def fb237():

    return {
        'train_transformation': None,
        'eval_transformation': None,
        'datadir': 'data-local/neuralilp/fb15k-237',
        'num_classes': 1000 #todo: what should be filled here??
    }

@export
def wn18():

    return {
        'train_transformation': None,
        'eval_transformation': None,
        'datadir': 'data-local/neuralilp/wn-18',
        'num_classes': 1000 #todo: what should be filled here??
    }

@export
def kinship():

    return {
        'train_transformation': None,
        'eval_transformation': None,
        'datadir': 'data-local/neuralilp/kinship',
        'num_classes': 1000 #todo: what should be filled here??
    }

@export
def umls():

    return {
        'train_transformation': None,
        'eval_transformation': None,
        'datadir': 'data-local/neuralilp/umls',
        'num_classes': 1000 #todo: what should be filled here??
    }

@export
def family():

    return {
        'train_transformation': None,
        'eval_transformation': None,
        'datadir': 'data-local/neuralilp/family/',
        'num_classes': 1000 #todo: what should be filled here??
    }

class ILP_dataset(Dataset):
    def __init__(self, datadir, dataset_type):
        ## TODO: parameterize, currently hard-coded
        # datadir = 'data-local/neuralilp/family/'
        # NOTE: removing these params
        # seed = 33
        # type_check = False
        # no_extra_facts = False
        # domain_size = 128
        # Note: setting 'share_db = true'

        # dataset_type = 'train'  # or 'test' or 'valid'
        self.dataset_type = dataset_type
        self.family_data = FamilyData(datadir, dataset_type)

    def __getitem__(self, idx):
        # TODO: Can we clean this up ?
        if self.dataset_type == 'train':
            query, head, tail = self.family_data.train[idx]
        elif self.dataset_type == 'test':
            query, head, tail = self.family_data.test[idx]
        elif self.dataset_type == 'valid':
            query, head, tail = self.family_data.valid[idx]
        else:
            assert False, "Wrong dataset type .. " + self.dataset_type

        ## todo: need to augment with reverse... ??
        ## todo: Also need to pass the matrix_db per batch (filtered out by removing facts in the current mini-batch) check: @NuralLP:data.py - lines 336-339 .. not sure how ??
        return idx, query, head, tail

    def __len__(self):
        # TODO: Can we clean this up ?
        if self.dataset_type == 'train':
            return len(self.family_data.train)
        elif self.dataset_type == 'test':
            return len(self.family_data.test)
        elif self.dataset_type == 'valid':
            return len(self.family_data.valid)
        else:
            assert False, "Wrong dataset type .. " + self.dataset_type

    def get_labels(self):
        if self.dataset_type == 'train':
            return [datum[1] for datum in self.family_data.train]
        elif self.dataset_type == 'test':
            return [datum[1] for datum in self.family_data.test]
        elif self.dataset_type == 'valid':
            return [datum[1] for datum in self.family_data.valid]
        else:
            assert False, "Wrong dataset type .. " + self.dataset_type

    def relabel_datum(self, idx, label):
        if self.dataset_type == 'train':
            self.family_data.train[idx] = (self.family_data.train[idx][0], label, self.family_data.train[idx][2])
        else:
            assert False, "Cannot relabel dataset of type .. " + self.dataset_type


@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }
