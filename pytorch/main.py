import re
import argparse
import os
import shutil
import time
import math
import logging

import torch.cuda
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torch_extras
setattr(torch, 'one_hot', torch_extras.one_hot)

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.NeuralLPutils.utils import get_rules as NeuralILPRules
from mean_teacher.NeuralLPutils.utils import get_predictions as NeuralILPPredictions

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def main(context):
    global global_step
    global best_prec1

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, dataset, dataset_test = create_data_loaders(**dataset_config, args=args)

    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        if dataset is not None:
            model_params['dataset'] = dataset
            model_params['num_step'] = args.num_step
            model_params['num_layer'] = args.num_layer
            model_params['query_embed_size'] = args.query_embed_size
            model_params['rnn_state_size'] = args.rnn_state_size

        model = model_factory(**model_params)
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()  # NOTE: removing data-parallelism in the model .. nn.DataParallel(model) #.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    if dataset is None:
        ema_model = create_model(ema=True)
    else:
        ema_model = None

    LOG.info(parameters_string(model))

    if dataset is None:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch, dataset=dataset_test)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch, dataset=dataset_test)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, epoch, training_log, dataset=dataset)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1, dataset=dataset_test)
            if dataset is None: #todo: Currently not on the NeuralLP model
                LOG.info("Evaluating the EMA model:")
                ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1, dataset=dataset)
                is_best = ema_prec1 > best_prec1
                best_prec1 = max(ema_prec1, best_prec1)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
        else:
            is_best = False

        if dataset is None: # todo: checkpoint the NeuralLP models later
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)

    if dataset is not None:
        # todo: parameterize:
        qq = np.array([i for i in range(0, dataset.family_data.num_operator)])
        hh = [0] * len(qq)
        tt = [0] * len(qq)
        batch_ids = [-1 for _ in range(len(qq))]
        mdb = {r: ([(0,0)], [0.], (dataset.family_data.num_entity, dataset.family_data.num_entity))
                for r in range(int(dataset.family_data.num_operator / 2))}
        input_var = [torch.autograd.Variable(torch.LongTensor(batch_ids)),
                     torch.autograd.Variable(torch.LongTensor(qq), volatile=True),
                     torch.autograd.Variable(torch.LongTensor(tt), volatile=True)]

        if torch.cuda.is_available():
            input_var[0] = input_var[0].cuda()
            input_var[1] = input_var[1].cuda()
            # NOTE: not converting input_var[2] to cuda() since we need to use one_hot ..

        x = model(input_var, mdb, save_attention_vectors=True)
        print("Dumping the Rules ...")
        rule_thr = 0.01 #todo: parameterize
        NeuralILPRules(model, dataset.family_data, context.result_directory(), rule_thr)
        NeuralILPPredictions(model, eval_loader, dataset_test, context.result_directory())
        print("Done!!!!")


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):

    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False


    if args.dataset in ['family', 'umls', 'kinship', 'wn18', 'fb237']:
        dataset = datasets.ILP_dataset(datadir, 'train')
        # TODO: For now not consider the teacher-student arch .. but just the supervised mode
        assert args.exclude_unlabeled, "For now not consider the teacher-student arch .. but just the supervised mode"
        train_dataset_size = dataset.__len__()
        sampler = SubsetRandomSampler(list(range(0, train_dataset_size)))  # TODO: generate labeled indices .. for now using all the examples in train
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=args.workers,
                                                   pin_memory=pin_memory)

        dataset_test = datasets.ILP_dataset(datadir, 'test')

        eval_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2 * args.workers,  # Needs images twice as fast
            pin_memory=pin_memory,
            drop_last=False)

    else:

        traindir = os.path.join(datadir, args.train_subdir)
        evaldir = os.path.join(datadir, args.eval_subdir)

        assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

        dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

        if args.labels:
            with open(args.labels) as f:
                labels = dict(line.split(' ') for line in f.read().splitlines())
            labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

        if args.exclude_unlabeled:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
        elif args.labeled_batch_size:
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(args.labeled_batch_size)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=args.workers,
                                                   pin_memory=pin_memory)

        eval_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(evaldir, eval_transformation),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2 * args.workers,  # Needs images twice as fast
            pin_memory=pin_memory,
            drop_last=False)

    if args.dataset in ['family', 'umls', 'kinship', 'wn18', 'fb237']:
        return train_loader, eval_loader, dataset, dataset_test
    else:
        return train_loader, eval_loader, None


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def filter_matrix_db(dataset, batch_input, type):

    if type == 'train':
        train_facts = dataset.family_data.train
        batch_facts = list(zip(batch_input[0], zip(batch_input[1], batch_input[2], batch_input[3])))
        batch_ids = [i for i in batch_facts[0]]
        extra_facts = [fact for idx, fact in enumerate(train_facts) if idx not in batch_ids]

        extra_mdb = dataset.family_data._db_to_matrix_db(extra_facts)
        augmented_mdb = dataset.family_data._combine_two_mdbs(extra_mdb, dataset.family_data.matrix_db_train)
    elif type == 'test':
        augmented_mdb = dataset.family_data.augmented_mdb_test
    elif type == 'valid':
        augmented_mdb = dataset.family_data.augmented_mdb_valid
    else:
        assert False, "Wrong type of dataset type : " + type

    return augmented_mdb


def train(train_loader, model, ema_model, optimizer, epoch, log, dataset):
    global global_step

    if torch.cuda.is_available():
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
        loss_criterion = nn.NLLLoss().cuda()  # todo: does it need any params ?? -- this is for the NeuralLP model
    else:
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()
        loss_criterion = nn.NLLLoss().cpu()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    if dataset is None:
        ema_model.train()

    end = time.time()

    for i, data_minibatch in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        if dataset is None:
            ((input, ema_input), target) = data_minibatch
            input_var = torch.autograd.Variable(input)
            ema_input_var = torch.autograd.Variable(ema_input, volatile=True)
            target_var = torch.autograd.Variable(target.cuda(async=True))
            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            ema_model_out = ema_model(ema_input_var)
            model_out = model(input_var)

            if isinstance(model_out, Variable):
                assert args.logit_distance_cost < 0
                logit1 = model_out
                ema_logit = ema_model_out
            else:
                assert len(model_out) == 2
                assert len(ema_model_out) == 2
                logit1, logit2 = model_out
                ema_logit, _ = ema_model_out

                ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

            if args.logit_distance_cost >= 0:
                class_logit, cons_logit = logit1, logit2
                res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
                meters.update('res_loss', res_loss.data[0])
            else:
                class_logit, cons_logit = logit1, logit1
                res_loss = 0

            class_softmax, cons_softmax = F.softmax(class_logit, dim=1), F.softmax(cons_logit, dim=1)
            ema_softmax = F.softmax(ema_logit, dim=1)

            class_loss = class_criterion(class_logit, target_var) / minibatch_size
            meters.update('class_loss', class_loss.data[0])

            ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.data[0])

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch)
                meters.update('cons_weight', consistency_weight)
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                meters.update('cons_loss', consistency_loss.data[0])
            else:
                consistency_loss = 0
                meters.update('cons_loss', 0)

            loss = class_loss + consistency_loss + res_loss

        else:

            input_batch_ids = data_minibatch[0]

            # not necessary to compute the one-hot --- http://pytorch.org/docs/master/nn.html#nllloss
            # size = (len(data_minibatch[2]), dataset.family_data.num_entity)
            # target = torch.one_hot(size, data_minibatch[2].view(-1, 1))
            qq = torch.cat([data_minibatch[1], torch.add(data_minibatch[1], dataset.family_data.num_relation)]) # NOTE: augment with reverse ...
            tt = torch.cat([data_minibatch[3], data_minibatch[2]]) # NOTE: augment with reverse ...
            target = torch.cat([data_minibatch[2], data_minibatch[3]])  # augment with reverse ...

            input_var = [input_batch_ids + input_batch_ids] + [torch.autograd.Variable(qq),
                                                               torch.autograd.Variable(tt)]

            if torch.cuda.is_available():
                input_var[0] = input_var[0].cuda()
                input_var[1] = input_var[1].cuda()
                # NOTE: not converting input_var[2] to cuda() since we need to use one_hot ..

            matrix_db = filter_matrix_db(dataset, data_minibatch, 'train')
            model_out = model(input_var, matrix_db)

            if torch.cuda.is_available():
                target_var = torch.autograd.Variable(target.cuda(async=True))
            else:
                target_var = torch.autograd.Variable(target.cpu())  # NOTE: the heads in the input is the target .. we are predicting a ranked list of these ... #todo: verify

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            class_loss = loss_criterion(model_out, target_var)
            meters.update('class_loss', class_loss.data[0])

            # print ("LOSS is " + str(loss.data[0]))
            loss = class_loss
            class_logit = model_out

        assert not (np.isnan(loss.data[0]) or loss.data[0] > 1e5), 'Loss explosion: {}'.format(loss.data[0])
        meters.update('loss', loss.data[0])

        prec1, prec10 = accuracy(class_logit.data, target_var.data, topk=(1, 10))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top10', prec10[0], labeled_minibatch_size)
        meters.update('error10', 100. - prec10[0], labeled_minibatch_size)

        if dataset is None: # todo: Currently not for the NeuralLP model
            ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
            meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_top5', ema_prec5[0], labeled_minibatch_size)
            meters.update('ema_error5', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        if dataset is None: # todo: not currently done in the NeuralLP model
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if dataset is None:
            if i % args.print_freq == 0:
                LOG.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Cons {meters[cons_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@5 {meters[top5]:.3f}'.format(
                        epoch, i, len(train_loader), meters=meters))
                log.record(epoch + i / len(train_loader), {
                    'step': global_step,
                    **meters.values(),
                    **meters.averages(),
                    **meters.sums()
                })
        else:
            if i % args.print_freq == 0:
                LOG.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@10 {meters[top10]:.3f}'.format(
                        epoch, i, len(train_loader), meters=meters))
                log.record(epoch + i / len(train_loader), {
                    'step': global_step,
                    **meters.values(),
                    **meters.averages(),
                    **meters.sums()
                })


def validate(eval_loader, model, log, global_step, epoch, dataset):

    if torch.cuda.is_available():
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
        loss_criterion = nn.NLLLoss().cuda()  # todo: does it need any params ?? -- this is for the NeuralLP model
    else:
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()
        loss_criterion = nn.NLLLoss().cpu()

    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data_minibatch in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        if dataset is None:
            (input, target) = data_minibatch
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        else:
            input_batch_ids = data_minibatch[0]
            qq = torch.cat([data_minibatch[1], torch.add(data_minibatch[1],
                                                         dataset.family_data.num_relation)])  # NOTE: augment with reverse ...
            tt = torch.cat([data_minibatch[3], data_minibatch[2]])  # NOTE: augment with reverse ...
            target = torch.cat([data_minibatch[2], data_minibatch[3]])  # augment with reverse ...

            input_var = [input_batch_ids] + [torch.autograd.Variable(qq, volatile=True),
                                             torch.autograd.Variable(tt, volatile=True)]
            if torch.cuda.is_available():
                input_var[0] = input_var[0].cuda()
                input_var[1] = input_var[1].cuda()
                # NOTE: not converting input_var[2] to cuda() since we need to use one_hot ..
                target_var = torch.autograd.Variable(target.cuda(async=True))
            else:
                target_var = torch.autograd.Variable(target.cpu())

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        if dataset is None:
            # compute output
            output1, output2 = model(input_var)
            softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
            class_loss = class_criterion(output1, target_var) / minibatch_size
        else:
            matrix_db = filter_matrix_db(dataset, data_minibatch, 'test')
            output1 = model(input_var, matrix_db)
            class_loss = loss_criterion(output1, target_var)

        # measure accuracy and record loss
        prec1, prec10 = accuracy(output1.data, target_var.data, topk=(1, 10))
        meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top10', prec10[0], labeled_minibatch_size)
        meters.update('error10', 100.0 - prec10[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@10 {meters[top10]:.3f}'.format(
                    i, len(eval_loader), meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@10 {top10.avg:.3f}'
          .format(top1=meters['top1'], top10=meters['top10']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    print('----------------')
    print("Running mean teacher experiment with args:")
    print('----------------')
    print(args)
    print('----------------')
    main(RunContext(__file__, 0, args.run_name))
