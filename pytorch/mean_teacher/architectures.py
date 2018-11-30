import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function

from .utils import export, parameter_count
import logging
LOG = logging.getLogger("arch")
###############
# https://stackoverflow.com/questions/34240703/whats-the-difference-between-softmax-and-softmax-cross-entropy-with-logits

###############
#### TODO: Add the emboot objective functions as another parameter -- NO longer necessary at least for the COLING submission
###############
# @export
# def pushpull_embed(pretrained=True, **kwargs):
#
#
# Look at pytorch implementations like https://github.com/fanglanting/skip-gram-pytorch/blob/master/model.py
# class PushPullSkipGram(nn.Module):
#
#     def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz):
#         super().__init__()


##############################################
###### Model with Positional Embeddings (1 - left of entity, 2 - right of entity) .. concatenated with embeddings to improve lstm
##############################################
@export
def custom_embed_w_pos(pretrained=True, word_vocab_size=7970, wordemb_size=300, hidden_size=300, num_classes=4, word_vocab_embed=None, update_pretrained_wordemb=False, use_dropout=False):

    lstm_hidden_size = 100
    model = SeqModelCustomEmbedWithPos(word_vocab_size, wordemb_size, lstm_hidden_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb)
    return model


# todo: Is padding the way done here ok ?
class SeqModelCustomEmbedWithPos(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_size, lstm_hidden_size, hidden_size, output_size, word_vocab_embed, update_pretrained_wordemb): #todo: add lstm parameters
        super().__init__()
        self.embedding_size = word_embedding_size
        self.entity_word_embeddings = nn.Embedding(word_vocab_size, word_embedding_size)
        self.pat_word_embeddings = nn.Embedding(word_vocab_size, word_embedding_size)

        if word_vocab_embed is not None:  # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            LOG.info("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.entity_word_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))
            self.pat_word_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))

            if update_pretrained_wordemb is False:  # NOTE: do not update the embeddings
                LOG.info("NOT UPDATING the word embeddings ....")
                self.entity_word_embeddings.weight.detach_()
                self.pat_word_embeddings.weight.detach_()
            else:
                LOG.info("UPDATING the word embeddings ....")

        # todo: keeping the hidden sizes of the LSTMs of entities and patterns to be same. To change later ?
        self.lstm_entities = nn.LSTM(word_embedding_size, lstm_hidden_size, num_layers=1, bidirectional=True)
        self.lstm_patterns = nn.LSTM(word_embedding_size+1, lstm_hidden_size, num_layers=1, bidirectional=True) ##NOTE:+1 due to position embedding value ...

        # UPDATE: NOT NECASSARY .. we can directly return from forward method the values that we want,
        #  in this case `entity_lstm_out` and `pattern_lstm_out`
        # Note: saving these representations, so that when they are computed during forward, we can use these variables
        # to dump the custom entity and pattern embeddings
        # self.entity_lstm_out = None
        # self.pattern_lstm_out = None

        # Note: Size of linear layer = [(lstm_hidden_size * 2) bi-LSTM ] * 2 --> concat of entity and context lstm out
        self.layer1 = nn.Linear(lstm_hidden_size * 2 * 2, hidden_size, bias=True)  # concatenate entity and pattern embeddings and followed by a linear layer;
        self.activation = nn.ReLU()  # non-linear activation
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True) # second linear layer from hidden layer to the output logits

    # todo: Is padding the way done here ok ? should I explicitly tell what the pad value is ?
    def forward(self, entity, pattern, pos_info):
        entity_word_embed = self.entity_word_embeddings(entity).permute(1, 0, 2)  # compute the embeddings of the words in the entity (Note the permute step)
        pattern_word_embed = self.pat_word_embeddings(pattern) # NOTE: doing the permute after appending the position tensor ... .permute(1, 0, 2)  # Note: the permute step is to make it compatible to be input to LSTM (seq of words,  batch, dimensions of each word)

        # 1. NOTE find position of entity token -- entity_idx
        #entity_idx = (pattern == self.entity_token_id).nonzero()
        #assert entity_idx.size()[0] == pattern.size()[0], "Something wrong .. more than one entity id present in patterns {} - {}".format(entity_idx.size(), pattern.size())
        #LOG.info(" Entity token ID " + str(self.entity_token_id))
        #LOG.info("Entity_idx  = " + str(entity_idx.size()))
        #LOG.info("Entity_idx  = " + str(entity_idx.data.cpu().numpy()))
        #LOG.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        LOG.info(" pattern size " + str(pattern.size()))
        LOG.info(" pattern embed size " + str(pattern_word_embed.size()))
        #LOG.info(" entity idx size " + str(entity_idx.size()))
        #LOG.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # 2. NOTE in a simple for loop if token_id < entity_idx --> 1 (left) else if token_id > entity_idx --> 2 (right)
        LOG.info("Position Info len(): " + str(len(pos_info)))
        LOG.info("pos_info[0] len() " + str(len(pos_info[0])))
        position_seq = torch.cuda.FloatTensor(pos_info)
        LOG.info("Position Seq : " + str(position_seq))

        # 3. NOTE create torch tensor and append to pattern_word_embed (note the permute step while appending) : Can be a single operation .... DONE
        LOG.info("size before .. " + str(pattern_word_embed.size()))
        pattern_word_embed = torch.cat([pattern_word_embed, position_seq.unsqueeze(2)], dim=2).permute(1, 0, 2) #DONE: permute in the same operation .. retaining and commenting the following lines ///
        LOG.info("size after concat and permute.. " + str(pattern_word_embed.size()))
        #pattern_word_embed = pattern_word_embed.permute(1, 0, 2)
        #LOG.info("size permute ... .. " + str(pattern_word_embed.size()))
        ###############################################
        # bi-LSTM computation here

        # https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
        self.lstm_entities.flatten_parameters()
        self.lstm_patterns.flatten_parameters()

        _, (entity_lstm_out, _) = self.lstm_entities(entity_word_embed)  # bi-LSTM over entities, hidden state is initialized to 0 if not provided
        entity_lstm_out = torch.cat([entity_lstm_out[0], entity_lstm_out[1]], 1) # roll out the 2 tuple output each of the LSTMs

        _, (pattern_lstm_out, _) = self.lstm_patterns(pattern_word_embed)  # bi-LSTM over pattern, hidden state is initialized to 0 if not provided
        pattern_lstm_out = torch.cat([pattern_lstm_out[0], pattern_lstm_out[1]], 1)  # roll out the 2 tuple output each of the LSTMs

        # concatenate the entity_lstm and pattern_lstm representations
        entity_and_pattern_lstm_out = torch.cat([entity_lstm_out, pattern_lstm_out], dim=1)

        ###############################################
        # LOG.info("###############################################")
        # LOG.info("entity_word_embed = " + str(entity_word_embed.size()))
        # LOG.info("pattern_word_embed_list = " + str(len(pattern_word_embed_list)))
        # LOG.info("pattern_word_embed_list[i] = " + str(pattern_word_embed_list[0].size()))
        # LOG.info("entity_lstm_out = " + str(entity_lstm_out.size()))
        # LOG.info("pattern_lstm_out = " + str(pattern_lstm_out.size()))
        # LOG.info("pattern_lstm_out_avg = " + str(pattern_lstm_out_avg.size()))
        # LOG.info("entity_and_pattern_lstm_out = " + str(entity_and_pattern_lstm_out.size()))
        # LOG.info("###############################################")

        ###############################################

        res = self.layer1(entity_and_pattern_lstm_out)
        res = self.activation(res)
        res = self.layer2(res)
        return res, entity_lstm_out, pattern_lstm_out

##############################################
##### More advanced architecture where the entity and pattern embeddings are computed by a Sequence model (like a biLSTM) and then concatenated together
##############################################
@export
def custom_embed(pretrained=True, word_vocab_size=7970, wordemb_size=300, hidden_size=300, num_classes=4, word_vocab_embed=None, update_pretrained_wordemb=False, use_dropout=False):

    lstm_hidden_size = 100
    model = SeqModelCustomEmbed(word_vocab_size, wordemb_size, lstm_hidden_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb, use_dropout)
    return model


# todo: Is padding the way done here ok ?
class SeqModelCustomEmbed(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_size, lstm_hidden_size, hidden_size, output_size, word_vocab_embed, update_pretrained_wordemb, use_dropout):
        super().__init__()
        self.embedding_size = word_embedding_size
        self.entity_word_embeddings = nn.Embedding(word_vocab_size, word_embedding_size)
        self.pat_word_embeddings = nn.Embedding(word_vocab_size, word_embedding_size)

        if word_vocab_embed is not None:  # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            LOG.info("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.entity_word_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))
            self.pat_word_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))

            if update_pretrained_wordemb is False:  # NOTE: do not update the embeddings
                LOG.info("NOT UPDATING the word embeddings ....")
                self.entity_word_embeddings.weight.detach_()
                self.pat_word_embeddings.weight.detach_()
            else:
                LOG.info("UPDATING the word embeddings ....")

        # todo: keeping the hidden sizes of the LSTMs of entities and patterns to be same. To change later ?
        self.lstm_entities = nn.LSTM(word_embedding_size, lstm_hidden_size, num_layers=1, bidirectional=True)
        self.lstm_patterns = nn.LSTM(word_embedding_size, lstm_hidden_size, num_layers=1, bidirectional=True)

        # UPDATE: NOT NECASSARY .. we can directly return from forward method the values that we want,
        #  in this case `entity_lstm_out` and `pattern_lstm_out`
        # Note: saving these representations, so that when they are computed during forward, we can use these variables
        # to dump the custom entity and pattern embeddings
        # self.entity_lstm_out = None
        # self.pattern_lstm_out = None

        # Note: Size of linear layer = [(lstm_hidden_size * 2) bi-LSTM ] * 2 --> concat of entity and context lstm out
        self.layer1 = nn.Linear(lstm_hidden_size * 2 * 2, hidden_size, bias=True)  # concatenate entity and pattern embeddings and followed by a linear layer;
        self.activation = nn.ReLU()  # non-linear activation
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True) # second linear layer from hidden layer to the output logits

        self.use_dropout = use_dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    # todo: Is padding the way done here ok ? should I explicitly tell what the pad value is ?
    def forward(self, entity, pattern):
        entity_word_embed = self.entity_word_embeddings(entity).permute(1, 0, 2)  # compute the embeddings of the words in the entity (Note the permute step)
        pattern_word_embed = self.pat_word_embeddings(pattern).permute(1, 0, 2)  # Note: the permute step is to make it compatible to be input to LSTM (seq of words,  batch, dimensions of each word)

        ###############################################
        # bi-LSTM computation here

        # https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
        self.lstm_entities.flatten_parameters()
        self.lstm_patterns.flatten_parameters()

        _, (entity_lstm_out, _) = self.lstm_entities(entity_word_embed)  # bi-LSTM over entities, hidden state is initialized to 0 if not provided
        entity_lstm_out = torch.cat([entity_lstm_out[0], entity_lstm_out[1]], 1) # roll out the 2 tuple output each of the LSTMs

        
        _, (pattern_lstm_out, _) = self.lstm_patterns(pattern_word_embed)  # bi-LSTM over pattern, hidden state is initialized to 0 if not provided
        pattern_lstm_out = torch.cat([pattern_lstm_out[0], pattern_lstm_out[1]], 1) # roll out the 2 tuple output each of the LSTMs

        # concatenate the entity_lstm and avgeraged pattern_lstm representations
        entity_and_pattern_lstm_out = torch.cat([entity_lstm_out, pattern_lstm_out], dim=1)

        ###############################################
        # LOG.info("###############################################")
        # LOG.info("entity_word_embed = " + str(entity_word_embed.size()))
        # LOG.info("pattern_word_embed_list = " + str(len(pattern_word_embed_list)))
        # LOG.info("pattern_word_embed_list[i] = " + str(pattern_word_embed_list[0].size()))
        # LOG.info("entity_lstm_out = " + str(entity_lstm_out.size()))
        # LOG.info("pattern_lstm_out = " + str(pattern_lstm_out.size()))
        # LOG.info("pattern_lstm_out_avg = " + str(pattern_lstm_out_avg.size()))
        # LOG.info("entity_and_pattern_lstm_out = " + str(entity_and_pattern_lstm_out.size()))
        # LOG.info("###############################################")

        ###############################################

        if self.use_dropout:  # dropout after the lstm layer
            entity_and_pattern_lstm_out = self.dropout_layer(entity_and_pattern_lstm_out)

        res = self.layer1(entity_and_pattern_lstm_out)

        if self.use_dropout:  # dropout in the hidden layer
            res = self.dropout_layer(res)

        res = self.activation(res)
        res = self.layer2(res)

        # NOTE: no dropout in the output layer

        return res, entity_lstm_out, pattern_lstm_out

##############################################
##### Simple architecture where the entity and pattern embeddings are computed by an average
##############################################
@export
def simple_MLP_embed(pretrained=True, num_classes=4, word_vocab_embed=None, word_vocab_size=7970, wordemb_size=300, hidden_size=50, update_pretrained_wordemb=False, use_dropout=False):

    # Note: custom embeddings sz in Emboot was 15 (used in conjunction of gigaword init embeddings as features in the classifier). This is similar to ladder networks

    model = FeedForwardMLPEmbed(word_vocab_size, wordemb_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb, use_dropout)
    return model


class FeedForwardMLPEmbed(nn.Module):
    def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz, word_vocab_embed, update_pretrained_wordemb, use_dropout):
        super().__init__()
        self.embedding_size = embedding_size
        self.entity_embeddings = nn.Embedding(word_vocab_size, embedding_size)
        self.pat_embeddings = nn.Embedding(word_vocab_size, embedding_size)

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222
        if word_vocab_embed is not None: # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            LOG.info("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.entity_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))
            self.pat_embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))

            if update_pretrained_wordemb is False:
                # NOTE: do not update the emebddings
                # https://discuss.pytorch.org/t/how-to-exclude-embedding-layer-from-model-parameters/1283
                LOG.info ("NOT UPDATING the word embeddings ....")
                self.entity_embeddings.weight.detach_()
                self.pat_embeddings.weight.detach_()
            else:
                LOG.info("UPDATING the word embeddings ....")

        ## Intialize the embeddings if pre-init enabled ? -- or in the fwd pass ?
        ## create : layer1 + ReLU
        self.layer1 = nn.Linear(embedding_size*2, hidden_sz, bias=True) ## concatenate entity and pattern embeddings
        self.activation = nn.ReLU()
        ## create : layer2 + Softmax: Create softmax here
        self.layer2 = nn.Linear(hidden_sz, output_sz, bias=True)
        # self.softmax = nn.Softmax(dim=1) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function

        self.use_dropout = use_dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, entity, pattern):
        entity_embed = torch.mean(self.entity_embeddings(entity), 1)             # Note: Average the word-embeddings
        pattern_embed = torch.mean(self.pat_embeddings(pattern), 1)            # Note: Average the pattern-embeddings 
        # LOG.info (entity_embed.size())
        # LOG.info (pattern_embed.size())
        concatenated = torch.cat([entity_embed, pattern_embed], 1)
        res = self.layer1(concatenated)
        res = self.activation(res)
        res = self.layer2(res)
        # LOG.info (res)
        # LOG.info (res.shape)
        # res = self.softmax(res) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function
        # LOG.info ("After softmax : " + str(res))

        if self.use_dropout:
            res = self.dropout_layer(res)

        return res

@export
def simple_MLP(pretrained=True, num_classes=10):

    ## Hard-coding the parameters
    input_sz = 900
    hidden_sz = 400
    output_sz = num_classes
    model = FeedForwardMLP(input_sz, hidden_sz, output_sz)

    return model

class FeedForwardMLP(nn.Module):
    def __init__(self, input_sz, hidden_sz, output_sz):
        ## Write code to initialize the MLP module
        super().__init__()
        self.layer1 = nn.Linear(input_sz, hidden_sz, bias=True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_sz, output_sz, bias=True)
        # self.softmax = nn.Softmax() ## IMPT NOTE: Removing the softmax from here as it is done in the loss function

    def forward(self, x):
        ## code to to the forward pass of the MLP module
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        # x = self.softmax(x) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function
        return x

@export
def cifar_shakeshake26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model


@export
def resnext152(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet224x224(BottleneckBlock,
                          layers=[3, 8, 36, 3],
                          channels=32 * 4,
                          groups=32,
                          downsample='basic', **kwargs)
    return model



class ResNet224x224(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 4
        self.downsample_mode = downsample
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, channels * 8, groups, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


class ResNet32x32(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BottleneckBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        if groups > 1:
            return 2 * planes
        else:
            return 4 * planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv_a1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn_a2 = nn.BatchNorm2d(planes)
        self.conv_a3 = nn.Conv2d(planes, self.out_channels(
            planes, groups), kernel_size=1, bias=False)
        self.bn_a3 = nn.BatchNorm2d(self.out_channels(planes, groups))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x

        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = self.relu(a)
        a = self.conv_a2(a)
        a = self.bn_a2(a)
        a = self.relu(a)
        a = self.conv_a3(a)
        a = self.bn_a3(a)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(residual + a)


class ShakeShakeBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab


class Shake(Function):
    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1,
                                                              grad_output.dim() - 1)]
        gate = Variable(grad_output.data.new(*gate_size).uniform_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)


class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=2 * in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2],
                       x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
