import torch
import logging
import torch.nn as nn
import os
from torch.utils.data.sampler import BatchSampler
import time

from mean_teacher import architectures, datasets, cli
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

LOG = logging.getLogger('main')


def create_model(arch, num_classes, word_vocab_embed, word_vocab_size,
                 wordemb_size, hidden_size, ema=False):
    LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
        pretrained='pre-trained ' if False else '',
        ema='EMA ' if ema else '',
        arch=arch))

    model_factory = architectures.__dict__[arch]
    model_params = dict(pretrained=False, num_classes=num_classes)

    model_params['word_vocab_embed'] = word_vocab_embed
    model_params['word_vocab_size'] = word_vocab_size
    model_params['wordemb_size'] = wordemb_size
    model_params['hidden_size'] = hidden_size
    model_params['update_pretrained_wordemb'] = False

    model = model_factory(**model_params)
    model = model.cuda()  # nn.DataParallel(model).cuda() .. # NOTE: Disabling data parallelism

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def generate_prediction_minibatch(min_batch_id, mini_batch_outputs, dataset, batch_size, category_labels):

    min_batch_predictions_gold = list()
    softmax = nn.Softmax(dim=1)
    results = torch.max(softmax(mini_batch_outputs), dim=1)

    for idx, (max_value, predictionId) in enumerate(results):
        dataset_id = (min_batch_id * batch_size) + idx
        mention_str = dataset.entity_vocab.get_word(dataset.mentions[dataset_id])
        gold_label = dataset.labels_str[dataset_id]
        min_batch_predictions_gold.append((mention_str, gold_label, category_labels[predictionId], max_value))

    return min_batch_predictions_gold


def predict_validate(eval_loader, model, model_type, arch, dataset, batch_size, result_filename):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    category_labels = dict(enumerate(sorted(list({l for l in dataset.labels_str}))))
    # switch to evaluate mode
    model.eval()

    entity_prediction_gold_list = list()

    end = time.time()
    for i, datapoint in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        entity = datapoint[0][0]
        patterns = datapoint[0][1]
        target = datapoint[1]

        entity_var = torch.autograd.Variable(entity, volatile=True).cuda()
        patterns_var = torch.autograd.Variable(patterns, volatile=True).cuda()

        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output

        if arch == "custom_embed":
            output1, entity_custom_embed, pattern_custom_embed = model(entity_var, patterns_var)
        else:
            output1 = model(entity_var, patterns_var)

        entity_prediction_gold_list += generate_prediction_minibatch(i, output1, dataset, batch_size, category_labels)

        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output1.data, target_var.data, topk=(1, 2)) #Note: Ajay changing this to 2 .. since there are only 4 labels in CoNLL dataset
        meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top2', prec2[0], labeled_minibatch_size)
        meters.update('error2', 100.0 - prec2[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'ClassLoss {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}'.format(
                    i, len(eval_loader), meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tClassLoss {class_loss.avg:.3f}'
          .format(top1=meters['top1'], class_loss=meters['class_loss']))

    LOG.info("Writing the predictions and the gold labels to the file :=> " + result_filename)
    with open(result_file_name+"_"+model_type+"_.txt", 'w') as rf:
        for item in entity_prediction_gold_list:
            rf.write(item[0] + "\t" + item[1] + "\t" + item[2] + "\t" + item[3] + "\n")
    rf.close()
    LOG.info("DONE ..")
    return meters['top1'].avg


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


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):

    evaldir = os.path.join(datadir, args.train_subdir)  # NOTE: test data is the same as train data. To load the word_vectors using the train_subdir
    LOG.info("evaldir : " + evaldir)

    dataset_test = datasets.NECDataset(evaldir, args, eval_transformation)

    eval_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2 * args.workers,
                                              pin_memory=True,
                                              drop_last=False)

    return eval_loader, dataset_test.word_vocab_embed, dataset_test.word_vocab.size()


if __name__ == '__main__':

    # 1. Set the following arguments
    ckpt_file = sys.argv[1]  # "best.ckpt"
    dataset = sys.argv[2]  # 'conll'
    print ("Loading the checkpoint from : " + ckpt_file)
    print ("Working on the dataset :=> " + dataset)

    result_file_name = "predictions"
    word_embed_size = 300
    hidden_size = 500
    batch_size = 64

    # 2. Initialize the configuration
    ckpt = torch.load(ckpt_file)
    arch = ckpt['arch']
    parser = cli.create_parser()
    parser.set_defaults(dataset=dataset,
                        train_subdir='train',
                        eval_subdir='val',
                        arch=arch,
                        pretrained_wordemb=True,
                        word_noise='drop:1',
                        batch_size=batch_size)
    args = parser.parse_known_args()[0]

    # 3. Load the eval data
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    eval_loader, word_vocab_embed, word_vocab_size = \
        create_data_loaders(**dataset_config, args=args)

    # 4. Load the models
    student_model = create_model(arch, num_classes, word_vocab_embed,
                                 word_vocab_size, word_embed_size, hidden_size, ema=False)

    teacher_model = create_model(arch, num_classes, word_vocab_embed,
                                 word_vocab_size, word_embed_size, hidden_size, ema=True)

    # 5. Init model state dicts
    student_model.load_state_dict(ckpt['state_dict'])
    teacher_model.load_state_dict(ckpt['ema_state_dict'])

    # 6. Call the evaluation code AND # 7. Generate the predictions file for the student model and the teacher model
    predict_validate(eval_loader, student_model, "student", args.arch, args.dataset, args.batch_size, result_file_name)
    predict_validate(eval_loader, teacher_model, "teacher", args.arch, args.dataset, args.batch_size, result_file_name)



