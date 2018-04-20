import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import io
from . import data
from .utils import export

from .processNLPdata.processNECdata import *
from itertools import chain

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
def ontonotes():

    if NECDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(NECDataset.NUM_WORDS_TO_REPLACE, NECDataset.OOV, NECDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(NECDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/nec/ontonotes',
        'num_classes': 11
    }


@export
def conll():

    if NECDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(NECDataset.NUM_WORDS_TO_REPLACE, NECDataset.OOV, NECDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(NECDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/nec/conll',
        'num_classes': 4
    }

##### USING Torchtext ... now reverting to using custom code
# def simple_tokenizer(datapoint):
#     fields = datapoint.split("__")
#     return fields
######################################################################

class NECDataset(Dataset):

    PAD = "@PADDING"
    OOV = "</s>"
    ENTITY = "@ENTITY"
    OOV_ID = 0
    ENTITY_ID = -1
    NUM_WORDS_TO_REPLACE = 1
    WORD_NOISE_TYPE = "drop"

    def __init__(self, dir, args, transform=None):
        entity_vocab_file = dir + "/entity_vocabulary.emboot.filtered.txt"
        context_vocab_file = dir + "/pattern_vocabulary_emboot.filtered.txt"
        dataset_file = dir + "/training_data_with_labels_emboot.filtered.txt"
        w2vfile = dir + "/../../vectors.goldbergdeps.txt"

        self.args = args
        self.entity_vocab = Vocabulary.from_file(entity_vocab_file)
        self.context_vocab = Vocabulary.from_file(context_vocab_file)
        self.mentions, self.contexts, self.labels_str = Datautils.read_data(dataset_file, self.entity_vocab, self.context_vocab)
        self.word_vocab, self.max_entity_len, self.max_pattern_len, self.max_num_patterns = self.build_word_vocabulary()
        if args.pretrained_wordemb:
            if args.eval_subdir not in dir:  # do not load the word embeddings again in eval
                self.gigaW2vEmbed, self.lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)
                self.word_vocab_embed = self.create_word_vocab_embed()
        else:
            print("Not loading the pretrained embeddings ... ")
            assert args.update_pretrained_wordemb, "Pretrained embeddings should be updated but " \
                                                   "--update-pretrained-wordemb = {}".format(args.update_pretrained_wordemb)
            self.word_vocab_embed = None

        # NOTE: Setting some class variables
        NECDataset.OOV_ID = self.word_vocab.get_id(NECDataset.OOV)
        NECDataset.ENTITY_ID = self.word_vocab.get_id(NECDataset.ENTITY)

        type_of_noise, size_of_noise = args.word_noise.split(":")
        NECDataset.WORD_NOISE_TYPE = type_of_noise
        NECDataset.NUM_WORDS_TO_REPLACE = int(size_of_noise)

        categories = sorted(list({l for l in self.labels_str}))
        self.lbl = [categories.index(l) for l in self.labels_str]

        self.transform = transform

    def sanitise_and_lookup_embedding(self, word_id):
        word = Gigaword.sanitiseWord(self.word_vocab.get_word(word_id))

        if word in self.lookupGiga:
            word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga[word]])
        else:
            word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<unk>"]])

        return word_embed

    def create_word_vocab_embed(self):

        word_vocab_embed = list()

        # leave last word = "@PADDING"
        for word_id in range(0, self.word_vocab.size()-1):
            word_embed = self.sanitise_and_lookup_embedding(word_id)
            word_vocab_embed.append(word_embed)

        # NOTE: adding the embed for @PADDING
        word_vocab_embed.append(Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<pad>"]]))
        return np.array(word_vocab_embed).astype('float32')

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

        word_vocab.add(NECDataset.PAD, 0)  # Note: Init a count of 0 to PAD, as we are not using it other than padding
        # print (max_entity)
        # print (max_entity_len)
        # print (max_pattern)
        # print (max_pattern_len)
        return word_vocab, max_entity_len, max_pattern_len, max_num_patterns

    def __len__(self):
        return len(self.mentions)

    def pad_item(self, dataitem, isPattern=True):
        if isPattern: # Note: precessing patterns .. consisting of list of lists (add pad to each list) and a final pad to the list of list
            dataitem_padded = list()
            for datum in dataitem:
                datum_padded = datum + [self.word_vocab.get_id(NECDataset.PAD)] * (self.max_pattern_len - len(datum))
                dataitem_padded.append(datum_padded)
            for _ in range(0, self.max_num_patterns - len(dataitem)):
                dataitem_padded.append([self.word_vocab.get_id(NECDataset.PAD)] * self.max_pattern_len)
        else:  # Note: padding an entity (consisting of a seq of tokens)
            dataitem_padded = dataitem + [self.word_vocab.get_id(NECDataset.PAD)] * (self.max_entity_len - len(dataitem))

        return dataitem_padded

    def get_num_classes(self):
        return len(list({l for l in self.lbl}))

    def get_labels(self):
        return self.lbl

    def __getitem__(self, idx):
        entity_words = [self.word_vocab.get_id(w) for w in self.entity_vocab.get_word(self.mentions[idx]).split(" ")]
        entity_words_padded = self.pad_item(entity_words, isPattern=False)
        entity_datum = torch.LongTensor(entity_words_padded)

        context_words_str = [[w for w in self.context_vocab.get_word(ctxId).split(" ")] for ctxId in self.contexts[idx]]
        context_words = [[self.word_vocab.get_id(w) for w in self.context_vocab.get_word(ctxId).split(" ")] for ctxId in self.contexts[idx]]

        if self.transform is not None:
            # 1. Replace word with synonym word in Wordnet / NIL (whichever is enabled)
            context_words_dropout_str = self.transform(context_words_str, NECDataset.ENTITY)

            if NECDataset.WORD_NOISE_TYPE == 'replace':
                assert len(context_words_dropout_str) == 2, "There is some issue with TransformTwice ... " #todo: what if we do not want to use the teacher ?
                new_replaced_words = [w for ctx in context_words_dropout_str[0] + context_words_dropout_str[1]
                                        for w in ctx
                                        if not self.word_vocab.contains(w)]

                # 2. Add word to word vocab (expand vocab)
                new_replaced_word_ids = [self.word_vocab.add(w, count=1)
                                         for w in new_replaced_words]

                # 3. Add the replaced words to the word_vocab_embed (if using pre-trained embedding)
                if self.args.pretrained_wordemb:
                    for word_id in new_replaced_word_ids:
                        word_embed = self.sanitise_and_lookup_embedding(word_id)
                        self.word_vocab_embed = np.vstack([self.word_vocab_embed, word_embed])

                # print("Added " + str(len(new_replaced_words)) + " words to the word_vocab... New Size: " + str(self.word_vocab.size()))

            context_words_dropout = list()
            context_words_dropout.append([[self.word_vocab.get_id(w)
                                            for w in ctx]
                                           for ctx in context_words_dropout_str[0]])
            context_words_dropout.append([[self.word_vocab.get_id(w)
                                            for w in ctx]
                                           for ctx in context_words_dropout_str[1]])

            if len(context_words_dropout) == 2:  # transform twice (1. student 2. teacher): DONE
                context_words_padded_0 = self.pad_item(context_words_dropout[0])
                context_words_padded_1 = self.pad_item(context_words_dropout[1])
                context_datums = (torch.LongTensor(context_words_padded_0), torch.LongTensor(context_words_padded_1))
            else: # todo: change this to an assert (if we are always using the student and teacher networks)
                context_words_padded = self.pad_item(context_words_dropout)
                context_datums = torch.LongTensor(context_words_padded)
        else:
            context_words_padded = self.pad_item(context_words)
            context_datums = torch.LongTensor(context_words_padded)

        # print ("label : " + self.labels[idx])
        # print ("label id : " + str(self.label_ids_all[idx]))
        label = self.lbl[idx]  # Note: .. no need to create a tensor variable

        if self.transform is not None:
            return (entity_datum, context_datums[0]), (entity_datum, context_datums[1]), label
        else:
            return (entity_datum, context_datums), label

        ##### USING Torchtext ... now reverting to using custom code
        # print ("Dir in NECDataset : " + dir)
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

#fan: add dataset riedel() and class REDataset
@export
def riedel():

    if REDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(REDataset.NUM_WORDS_TO_REPLACE, REDataset.OOV, REDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(REDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/re/Riedel2010',
        'num_classes': 53
    }

class REDataset(Dataset):

    PAD = "@PADDING"
    OOV = "</s>"
    ENTITY = "@ENTITY"
    OOV_ID = 0
    ENTITY_ID = -1
    NUM_WORDS_TO_REPLACE = 1
    WORD_NOISE_TYPE = "drop"

    def __init__(self, dir, args, transform=None, type='train'):

        dataset_file = dir + "/" + type + ".txt"
        w2vfile = dir + "/../../vectors.goldbergdeps.txt"

        self.args = args

        if args.eval_subdir not in dir:
            self.entities1_words, self.entities2_words, self.sentences_words, self.labels_str,\
                self.chunks_left_words, self.chunks_inbetween_words, self.chunks_right_words, \
                self.max_entity_len, self.max_sentence_len, self.max_left_len, self.max_inbetween_len, self.max_right_len \
                = Datautils.read_re_data(dataset_file)

            # print(self.entities1_words[0])
            # print(self.entities2_words[0])
            # print(self.sentences_words[0])
            # print(self.labels_str[0])
            # print(self.chunks_left_words[0])
            # print(self.chunks_inbetween_words[0])
            # print(self.chunks_right_words[0])
            print(self.max_entity_len)
            print(self.max_sentence_len)
            print(self.max_left_len)
            print(self.max_inbetween_len)
            print(self.max_right_len)

            self.word_vocab = Vocabulary()
            for word in chain.from_iterable(zip(*self.entities1_words)):
                self.word_vocab.add(word)
            for word in chain.from_iterable(zip(*self.entities2_words)):
                self.word_vocab.add(word)
            for word in chain.from_iterable(zip(*self.sentences_words)):
                self.word_vocab.add(word)
            self.word_vocab.add("@PADDING", 0)

            vocab_file = dir + "/../vocabulary_train.txt"
            self.word_vocab.to_file(vocab_file)

            maxlen_file = dir + "/../maxlen_train.txt"
            with io.open(maxlen_file, 'w', encoding='utf8') as f:
                f.write(str(self.max_entity_len) + '\t' + str(self.max_sentence_len) + '\t' + str(self.max_left_len) + '\t' + str(self.max_inbetween_len) + '\t' + str(self.max_right_len))

        else:
            vocab_file = dir + "/../vocabulary_train.txt"
            self.word_vocab = Vocabulary.from_file(vocab_file)

            maxlen_file = dir + "/../maxlen_train.txt"
            with io.open(maxlen_file, encoding='utf8') as f:
                for line in f:
                    [self.max_entity_len, self.max_sentence_len, self.max_left_len, self.max_inbetween_len, self.max_right_len] = line.split('\t')

            # print(self.max_entity_len)
            # print(self.max_sentence_len)
            # print(self.max_left_len)
            # print(self.max_inbetween_len)
            # print(self.max_right_len)

            self.entities1_words, self.entities2_words, self.sentences_words, self.labels_str, \
                self.chunks_left_words, self.chunks_inbetween_words, self.chunks_right_words, \
                _, _, _, _, _ \
                = Datautils.read_re_data(dataset_file)

        if args.pretrained_wordemb:
            if args.eval_subdir not in dir:  # do not load the word embeddings again in eval
                self.gigaW2vEmbed, self.lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)
                self.word_vocab_embed = self.create_word_vocab_embed()
        else:
            print("Not loading the pretrained embeddings ... ")
            assert args.update_pretrained_wordemb, "Pretrained embeddings should be updated but " \
                                                   "--update-pretrained-wordemb = {}".format(args.update_pretrained_wordemb)
            self.word_vocab_embed = None

        # NOTE: Setting some class variables
        REDataset.OOV_ID = self.word_vocab.get_id(REDataset.OOV)
        REDataset.ENTITY_ID = self.word_vocab.get_id(REDataset.ENTITY)

        type_of_noise, size_of_noise = args.word_noise.split(":")
        REDataset.WORD_NOISE_TYPE = type_of_noise
        REDataset.NUM_WORDS_TO_REPLACE = int(size_of_noise)

        categories = sorted(list({l for l in self.labels_str}))
        self.lbl = [categories.index(l) for l in self.labels_str]

        self.transform = transform

    def __getitem__(self, idx):
        entity1_words_id = [self.word_vocab.get_id(w) for w in self.entities1_words[idx]]   #entity1_words: a list of ids
        entity2_words_id = [self.word_vocab.get_id(w) for w in self.entities2_words[idx]]
        entity1_words_id_padded = self.pad_item(entity1_words_id)
        entity2_words_id_padded = self.pad_item(entity2_words_id)
        entity1_datum = torch.LongTensor(entity1_words_id_padded)
        entity2_datum = torch.LongTensor(entity2_words_id_padded)

        sentence_words_id = [self.word_vocab.get_id(w) for w in self.sentences_words[idx]]
        left_words_id = [self.word_vocab.get_id(w) for w in self.chunks_left_words[idx]]
        inbetween_words_id = [self.word_vocab.get_id(w) for w in self.chunks_inbetween_words[idx]]
        right_words_id = [self.word_vocab.get_id(w) for w in self.chunks_right_words[idx]]

        if self.transform is not None:

            sentence_words_dropout = self.transform([self.sentences_words[idx]], REDataset.ENTITY)
            left_words_dropout = self.transform([self.chunks_left_words[idx]], REDataset.ENTITY)
            inbetween_words_dropout = self.transform([self.chunks_inbetween_words[idx]], REDataset.ENTITY)
            right_words_dropout = self.transform([self.chunks_right_words[idx]], REDataset.ENTITY)

            sentence_words_id_dropout = list()
            sentence_words_id_dropout.append([self.word_vocab.get_id(w) for w in sentence_words_dropout[0][0]])
            sentence_words_id_dropout.append([self.word_vocab.get_id(w) for w in sentence_words_dropout[1][0]])

            left_words_id_dropout = list()
            left_words_id_dropout.append([self.word_vocab.get_id(w) for w in left_words_dropout[0][0]])
            left_words_id_dropout.append([self.word_vocab.get_id(w) for w in left_words_dropout[1][0]])

            inbetween_words_id_dropout = list()
            inbetween_words_id_dropout.append([self.word_vocab.get_id(w) for w in inbetween_words_dropout[0][0]])
            inbetween_words_id_dropout.append([self.word_vocab.get_id(w) for w in inbetween_words_dropout[1][0]])

            right_words_id_dropout = list()
            right_words_id_dropout.append([self.word_vocab.get_id(w) for w in right_words_dropout[0][0]])
            right_words_id_dropout.append([self.word_vocab.get_id(w) for w in right_words_dropout[1][0]])

            if len(sentence_words_id_dropout) == 2:  # transform twice (1. student 2. teacher): DONE
                sentence_words_padded_0 = self.pad_item(sentence_words_id_dropout[0], 'sentence')
                sentence_words_padded_1 = self.pad_item(sentence_words_id_dropout[1], 'sentence')
                sentence_datums = (torch.LongTensor(sentence_words_padded_0), torch.LongTensor(sentence_words_padded_1))

                left_words_padded_0 = self.pad_item(left_words_id_dropout[0], 'left')
                left_words_padded_1 = self.pad_item(left_words_id_dropout[1], 'left')
                left_datums = (torch.LongTensor(left_words_padded_0), torch.LongTensor(left_words_padded_1))

                inbetween_words_padded_0 = self.pad_item(inbetween_words_id_dropout[0], 'inbetween')
                inbetween_words_padded_1 = self.pad_item(inbetween_words_id_dropout[1], 'inbetween')
                inbetween_datums = (torch.LongTensor(inbetween_words_padded_0), torch.LongTensor(inbetween_words_padded_1))

                right_words_padded_0 = self.pad_item(right_words_id_dropout[0], 'right')
                right_words_padded_1 = self.pad_item(right_words_id_dropout[1], 'right')
                right_datums = (torch.LongTensor(right_words_padded_0), torch.LongTensor(right_words_padded_1))

            # else:  # todo: change this to an assert (if we are always using the student and teacher networks)
            #     context_words_padded = self.pad_item(context_words_dropout)
            #     context_datums = torch.LongTensor(context_words_padded)

        else:
            sentence_words_padded = self.pad_item(sentence_words_id, 'sentence')
            sentence_datums = torch.LongTensor(sentence_words_padded)

            left_words_padded = self.pad_item(left_words_id, 'left')
            left_datums = torch.LongTensor(left_words_padded)

            inbetween_words_padded = self.pad_item(inbetween_words_id, 'inbetween')
            inbetween_datums = torch.LongTensor(inbetween_words_padded)

            right_words_padded = self.pad_item(right_words_id, 'right')
            right_datums = torch.LongTensor(right_words_padded)

        # print ("label : " + self.labels[idx])
        # print ("label id : " + str(self.label_ids_all[idx]))
        label = self.lbl[idx]  # Note: .. no need to create a tensor variable

        if self.transform is not None:
            return (entity1_datum, entity2_datum, sentence_datums[0],left_datums[0], inbetween_datums[0], right_datums[0]), (entity1_datum, entity2_datum, sentence_datums[1],left_datums[1], inbetween_datums[1], right_datums[1]), label
        else:
            return (entity1_datum, entity2_datum, sentence_datums, left_datums, inbetween_datums, right_datums), label

        ##### USING Torchtext ... now reverting to using custom code
        # print ("Dir in NECDataset : " + dir)
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

    def sanitise_and_lookup_embedding(self, word_id):
        word = Gigaword.sanitiseWord(self.word_vocab.get_word(word_id))

        if word in self.lookupGiga:
            word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga[word]])
        else:
            word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<unk>"]])

        return word_embed

    def create_word_vocab_embed(self):

        word_vocab_embed = list()

        # leave last word = "@PADDING"
        for word_id in range(0, self.word_vocab.size() - 1):
            word_embed = self.sanitise_and_lookup_embedding(word_id)
            word_vocab_embed.append(word_embed)

        # NOTE: adding the embed for @PADDING
        word_vocab_embed.append(Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<pad>"]]))
        return np.array(word_vocab_embed).astype('float32')

    def __len__(self):
        return len(self.entities1_words)

    def get_num_classes(self):
        return len(list({l for l in self.lbl}))

    def get_labels(self):
        return self.lbl

    def pad_item(self, dataitem, type='entity'):
        if (type is 'sentence'): # Note: precessing patterns .. consisting of list of lists (add pad to each list) and a final pad to the list of list
            dataitem_padded = dataitem + [self.word_vocab.get_id(REDataset.PAD)] * (self.max_sentence_len - len(dataitem))
        elif (type is 'entity'):  # Note: padding an entity (consisting of a seq of tokens)
            dataitem_padded = dataitem + [self.word_vocab.get_id(REDataset.PAD)] * (self.max_entity_len - len(dataitem))
        elif (type is 'left'):
            dataitem_padded = dataitem + [self.word_vocab.get_id(REDataset.PAD)] * (self.max_left_len - len(dataitem))
        elif (type is 'inbetween'):
            dataitem_padded = dataitem + [self.word_vocab.get_id(REDataset.PAD)] * (self.max_inbetween_len - len(dataitem))
        elif (type is 'right'):
            dataitem_padded = dataitem + [self.word_vocab.get_id(REDataset.PAD)] * (self.max_right_len - len(dataitem))

        return dataitem_padded


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
