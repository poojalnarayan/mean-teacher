#!/usr/bin/env python

import numpy as np
from collections import defaultdict

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
    def read_nec_ctx_data(cls, filename):
        labels = []
        entities = []
        contexts = []
        pos_info_array = []

        lbl_set = set()
        word_set = set()

        max_entity_len = 0
        max_context_len = 0

        max_entity = ""

        with open(filename) as f:
            # word_counts = dict()
            for line in f:
                vals = line.strip().split('\t')

                cur_ent = vals[1].split(' ')
                word_set.update(set(cur_ent))
                entities.append(vals[1])
                if max_entity_len < len(cur_ent):
                    max_entity_len = len(cur_ent)
                    max_entity = vals[1]

                lbl_set.add(vals[0])
                labels.append(vals[0])

                cur_context = vals[2].split(' ')
                entity_idx = cur_context.index("@ENTITY")
                pos_info =  [1 if idx < entity_idx
                               else 2 if idx > entity_idx
                               else 0 for idx, word in enumerate(cur_context)]
                pos_info_array.append(pos_info)
                word_set.update(set(cur_context))
                contexts.append(vals[2])
                if max_context_len < len(cur_context):
                    max_context_len = len(cur_context)

            word_set.add("</s>")
            word_list = list(sorted(word_set))
            lbl_list = list(sorted(lbl_set))

            #Add padding to the end because in create_word_vocab_embed() we add it last
            word_list.append("@PADDING")

            #print("max_entity = ", max_entity)
            label_dict = dict([(t[1], t[0]) for t in list(enumerate(lbl_list))])
            entity_pattern_dict = dict([(t[1], t[0]) for t in list(enumerate(word_list))])
        # return np.array(entities), np.array([np.array(c) for c in contexts]), np.array(labels)
        return labels, entities, contexts, label_dict, entity_pattern_dict, max_entity_len, max_context_len, pos_info_array

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
