import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

from . import data
from .utils import export

from .processNLPdata.processNECdata import *

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

@export
def conll():

    num_words_to_dropout = 1
    dropout = data.RandomPatternDropout(num_words_to_dropout, CoNLLDataset.OOV_ID)
    # todo: Add the replace (from wordnet) noise functionality

    return {
        'train_transformation': data.TransformTwiceNEC(dropout),
        'eval_transformation': None,
        'datadir': 'data-local/nec/conll',
        'num_classes': 4
    }

##### USING Torchtext ... now reverting to using custom code
# def simple_tokenizer(datapoint):
#     fields = datapoint.split("__")
#     return fields
######################################################################


class CoNLLDataset(Dataset):

    PAD = "@PADDING"
    OOV = "</s>"
    ENTITY = "@ENTITY"
    OOV_ID = 0
    ENTITY_ID = -1

    def __init__(self, dir, transform=None):
        entity_vocab_file = dir + "/entity_vocabulary.emboot.filtered.txt"
        context_vocab_file = dir + "/pattern_vocabulary_emboot.filtered.txt"
        dataset_file = dir + "/training_data_with_labels_emboot.filtered.txt"

        categories = sorted(list(['PER', 'ORG', 'LOC', 'MISC']))
        self.entity_vocab = Vocabulary.from_file(entity_vocab_file)
        self.context_vocab = Vocabulary.from_file(context_vocab_file)
        self.mentions, self.contexts, self.labels_str = Datautils.read_data(dataset_file, self.entity_vocab, self.context_vocab)
        self.word_vocab, self.max_entity_len, self.max_pattern_len, self.max_num_patterns = self.build_word_vocabulary()
        CoNLLDataset.OOV_ID = self.word_vocab.get_id(CoNLLDataset.OOV)
        CoNLLDataset.ENTITY_ID = self.word_vocab.get_id(CoNLLDataset.ENTITY)
        self.lbl = [categories.index(l) for l in self.labels_str]

        self.transform = transform

    def build_word_vocabulary(self):
        word_vocab = Vocabulary()

        max_entity_len = 0
        max_pattern_len = 0
        max_num_patterns = 0

        max_entity = ""
        max_pattern = ""

        for mentionId in self.mentions:
            words = [w for w in self.entity_vocab.get_word(mentionId).split(" ")]
            for w in words:
                word_vocab.add(w)

            if len(words) > max_entity_len:
                max_entity_len = len(words)
                max_entity = words

        for context in self.contexts:
            for patternId in context:
                words = [w for w in self.context_vocab.get_word(patternId).split(" ")]
                for w in words:
                    word_vocab.add(w)

                if len(words) > max_pattern_len:
                    max_pattern_len = len(words)
                    max_pattern = words

            if len(context) > max_num_patterns:
                max_num_patterns = len(context)

        word_vocab.add(CoNLLDataset.PAD, 0) ## todo: is this right ?
        # print (max_entity)
        # print (max_entity_len)
        # print (max_pattern)
        # print (max_pattern_len)
        return word_vocab, max_entity_len, max_pattern_len, max_num_patterns

    def __len__(self):
        return len(self.mentions)

    def pad_item(self, dataitem, isPattern=True):
        if isPattern: ## Note: precessing patterns .. consisting of list of lists (add pad to each list) and a final pad to the list of list
            dataitem_padded = list()
            for datum in dataitem:
                datum_padded = datum + [self.word_vocab.get_id(CoNLLDataset.PAD)] * (self.max_pattern_len - len(datum))
                dataitem_padded.append(datum_padded)
            for _ in range(0, self.max_num_patterns - len(dataitem)):
                dataitem_padded.append([self.word_vocab.get_id(CoNLLDataset.PAD)] * self.max_pattern_len)
        else: ## Note: padding an entity (consisting of a seq of tokens)
            dataitem_padded = dataitem + [self.word_vocab.get_id(CoNLLDataset.PAD)] * (self.max_entity_len - len(dataitem))

        return dataitem_padded

    def get_num_classes(self):
        return len(list({l for l in self.lbl}))

    def get_labels(self):
        return self.lbl

    def __getitem__(self, idx):
        entity_words = [self.word_vocab.get_id(w) for w in self.entity_vocab.get_word(self.mentions[idx]).split(" ")]
        entity_words_padded = self.pad_item(entity_words, isPattern=False)
        entity_datum = torch.LongTensor(entity_words_padded)  ## todo: verify --> Note: this is not necessary [prev .. Note the square brackets .. converting singleton into array [of size 1] ]

        context_words = [[self.word_vocab.get_id(w) for w in self.context_vocab.get_word(ctxId).split(" ")] for ctxId in self.contexts[idx]]
        # max_ctx_len = len(max(context_words, key=lambda x: len(x)))

        if self.transform is not None:
            ## Dropout done .. transform twice (1. student 2. teacher): DONE
            context_words_dropout = self.transform(context_words, self.word_vocab.get_id(CoNLLDataset.ENTITY))
            if len(context_words_dropout) == 2:
                # context_words_padded_1 = [ctx + [self.word_vocab.get_id(CoNLLDataset.PAD)] * (max_ctx_len - len(ctx)) for ctx in context_words_dropout[0]]
                # context_words_padded_2 = [ctx + [self.word_vocab.get_id(CoNLLDataset.PAD)] * (max_ctx_len - len(ctx)) for ctx in context_words_dropout[1]]
                # context_datums = (torch.LongTensor(context_words_padded_1), torch.LongTensor(context_words_padded_2))
                context_words_padded_0 = self.pad_item(context_words_dropout[0])
                context_words_padded_1 = self.pad_item(context_words_dropout[1])
                context_datums = (torch.LongTensor(context_words_padded_0), torch.LongTensor(context_words_padded_1))
            else:
                context_words_padded = self.pad_item(context_words_dropout)
                context_datums = torch.LongTensor(context_words_padded)
        else:
            # context_words_padded = [ ctx + [self.word_vocab.get_id(CoNLLDataset.PAD)]*(max_ctx_len - len(ctx)) for ctx in context_words]
            context_words_padded = self.pad_item(context_words)
            context_datums = torch.LongTensor(context_words_padded)

        # print ("label : " + self.labels[idx])
        # print ("label id : " + str(self.label_ids_all[idx]))
        label = torch.LongTensor([self.lbl[idx]])

        if self.transform is not None:
            return (entity_datum, context_datums[0]), (entity_datum, context_datums[1]), label
        else:
            return (entity_datum, context_datums), None, label

        ##### USING Torchtext ... now reverting to using custom code
        # print ("Dir in CoNLLDataset : " + dir)
        # data_file = "training_data_with_labels_emboot.filtered.txt.processed"
        #
        # LABEL = Field(sequential=False, use_vocab=True)
        # ENTITY = Field(sequential=False, use_vocab=True, lower=True)
        # PATTERN = Field(sequential=True, use_vocab=True, lower=True, tokenize=simple_tokenizer)
        #
        # datafields = [("label", LABEL), ("entity", ENTITY), ("patterns", PATTERN)]
        # dataset, _ = TabularDataset.splits(path=dir, train=data_file, validation=data_file, format='tsv',
        #                                  fields=datafields)
        #
        # LABEL.build_vocab(dataset)
        # ENTITY.build_vocab(dataset)
        # PATTERN.build_vocab(dataset)

        # APPLY THE TRANSFORMATION HERE
        # transform = transform

        # return dataset
        ######################################################################


@export
def riedel10():

    return {
        'train_transformation': data.TransformTwice(data.AddGaussianNoise()),
        'eval_transformation': None,
        'datadir': 'data-local/riedel10',
        'num_classes': 56
    }

@export
def gids():

    return {
        'train_transformation': data.TransformTwice(data.AddGaussianNoise()),
        'eval_transformation': None,
        'datadir': 'data-local/gids',
        'num_classes': 5
    }

class RiedelDataset(Dataset):
    def __init__(self, dir, transform=None):
        numpy_file = dir + '/np_relext.npy'
        lbl_numpy_file = dir + '/np_relext_labels.npy'

        self.data = np.load(numpy_file)
        self.lbl = np.load(lbl_numpy_file)

        # self.tensor = torch.stack([torch.Tensor(datum) for datum in data])
        # self.tensor_lbl = torch.stack([torch.IntTensor([int(lbl)]) for lbl in lbl])
        #
        # self.dataset = torch.utils.data.TensorDataset(self.tensor, self.tensor_lbl)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            tensor_datum = self.transform(torch.Tensor(self.data[idx]))
        else:
            tensor_datum = torch.Tensor(self.data[idx])

        label = self.lbl[idx]

        return tensor_datum, label

    def get_num_classes(self):
        return len(list({l for l in self.lbl}))

    def get_labels(self):
        return self.lbl
