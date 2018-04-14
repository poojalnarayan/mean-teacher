#!/usr/bin/env python

import numpy as np
from collections import defaultdict
from .vocabulary import Vocabulary


class Datautils:

    ## read the data from the file with the entity_ids provided by entity_vocab and context_ids provided by context_vocab
    ## data format:
    #################
    ## [label]\t[Entity Mention]\t[[context mention1]\t[context mention2],\t...[]]
    ## NOTE: The label to be removed later from the dataset and the routine suitably adjusted. Inserted here for debugging

    @classmethod
    def read_data(cls, filename, entity_vocab, context_vocab):
        labels = []
        entities = []
        contexts = []

        with open(filename) as f:
            word_counts = dict()
            for line in f:
                vals = line.strip().split('\t')
                labels.append(vals[0])
                if vals[1] not in word_counts:
                    word_counts[vals[1]] = 1
                else:
                    word_counts[vals[1]] += 1
                word_id = entity_vocab.get_id(vals[1])
                if word_id is not None:
                    entities.append(word_id)
                contexts.append([context_vocab.get_id(c) for c in vals[2:] if context_vocab.get_id(c) is not None])

            num_count_words = 0
            for word in word_counts:
                if word_counts[word] >= 6:
                    num_count_words+=1
            print('num count words:',num_count_words)

        # return np.array(entities), np.array([np.array(c) for c in contexts]), np.array(labels)
        return entities, contexts, labels

    @classmethod
    def read_re_data(cls, filename):
        labels = []
        entities1 = []
        entities2 = []
        sentences = []
        chunks_left = []
        chunks_inbetween = []
        chunks_right = []

        max_entity_len = 0
        max_sentence_len = 0
        vocabulary = Vocabulary()

        with open(filename) as f:
            for line in f:
                vals = line.strip().split('\t')
                labels.append(vals[4])
                entities1_words = vals[2].strip().split('_')
                entities2_words = vals[3].strip().split('_')

                if len(entities1_words) > max_entity_len:
                    max_entity_len = len(entities1_words)
                if len(entities2_words) > max_entity_len:
                    max_entity_len = len(entities2_words)

                sentence_str = vals[5].strip().replace(vals[2], "@ENTITY", 1)
                sentence_str = sentence_str.replace(vals[3], "@ENTITY", 1)
                left_str = sentence_str.partition("@ENTITY")[0]
                inbetween_str = sentence_str.partition("@ENTITY")[2].partition("@ENTITY")[0]
                right_str = sentence_str.partition("@ENTITY")[2].partition("@ENTITY")[2]

                chunks_left.append(left_str.split(' '))
                chunks_inbetween.append(inbetween_str.split(' '))
                chunks_right.append(right_str.split(' ')[:-1])
                sentences_words = sentence_str.split(' ')[:-1]    #the last word is "###END###"

                if len(sentences_words) > max_sentence_len:
                    max_sentence_len = len(sentences_words)

                vocabulary.add(word for word in entities1_words)
                vocabulary.add(word for word in entities2_words)
                vocabulary.add(word for word in sentences_words)
                entities1.append(entities1_words)
                entities2.append(entities2_words)
                sentences.append(sentences_words)
        vocabulary.add("@PADDING", 0)

        return entities1, entities2, sentences, labels, vocabulary, chunks_left, chunks_inbetween, chunks_right, max_entity_len, max_sentence_len


    ## Takes as input an array of entity mentions(ids) along with their contexts(ids) and converts them to individual pairs of entity and context
    ## Entity_Mention_1  -- context_mention_1, context_mention_2, ...
    ## ==>
    ## Entity_Mention_1 context_mention_1 0 ## Note the last number is the mention id, needed later to associate entity mention with all its contexts
    ## Entity_Mention_1 context_mention_2 1
    ## ....

    @classmethod
    def prepare_for_skipgram(cls, entities, contexts):

        entity_ids = []
        context_ids = []
        mention_ids = []
        for i in range(len(entities)):
            word = entities[i]
            context = contexts[i]
            for c in context:
                entity_ids.append(word)
                context_ids.append(c)
                mention_ids.append(i)
        return np.array(entity_ids), np.array(context_ids), np.array(mention_ids)


    ## NOTE: To understand the current negative sampling and replace it with a more simpler version. Keeping it as it is now.
    @classmethod
    def collect_negatives(cls, entities_for_sg, contexts_for_sg, entity_vocab, context_vocab):

        n_entities = entity_vocab.size()
        n_contexts = context_vocab.size()
        negatives = np.empty((n_entities, n_contexts))

        for i in range(n_entities):
            negatives[i,:] = np.arange(n_contexts)
        negatives[entities_for_sg, contexts_for_sg] = 0
        return negatives

    @classmethod
    def construct_indices(cls, mentions, contexts):

        entityToPatternsIdx = defaultdict(set)
        for men, ctxs in zip(list(mentions), list([list(c) for c in contexts])):
            for ctx in ctxs:
                tmp = entityToPatternsIdx[men]
                tmp.add(ctx)
                entityToPatternsIdx[men] = tmp

        patternToEntitiesIdx = defaultdict(set)
        for men, ctxs in zip(list(mentions), list([list(c) for c in contexts])):
            for ctx in ctxs:
                tmp = patternToEntitiesIdx[ctx]
                tmp.add(men)
                patternToEntitiesIdx[ctx] = tmp

        return entityToPatternsIdx, patternToEntitiesIdx
