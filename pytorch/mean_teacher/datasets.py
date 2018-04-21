import torchvision.transforms as transforms

from . import data
from .utils import export
from torch.utils.data import Dataset
from .processILPdata.family_data import Data as FamilyData

@export
def family():

    return {
        'train_transformation': None,
        'eval_transformation': None,
        'datadir': 'data-local/neuralilp/family/',
        'num_classes': 1000 #todo: what should be filled here??
    }

class ILP_dataset(Dataset):
    def __init__(self):
        ## TODO: parameterize, currently hard-coded
        datadir = 'data-local/neuralilp/family/'
        # NOTE: removing these params
        # seed = 33
        # type_check = False
        # no_extra_facts = False
        # domain_size = 128
        # Note: setting 'share_db = true'

        type = 'train' # or 'test' or 'valid'
        self.family_data = FamilyData(datadir, type)

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
