import numpy as np 
import os


class Data(object):
    def __init__(self, folder, type):
        # Note: setting 'share_db = true'
        self.query_include_reverse = True
        self.relation_file = os.path.join(folder, "relations.txt")
        self.entity_file = os.path.join(folder, "entities.txt")
        
        self.relation_to_number, self.entity_to_number = self._numerical_encode()
        self.number_to_entity = {v: k for k, v in self.entity_to_number.items()}
        self.num_relation = len(self.relation_to_number)
        self.num_query = self.num_relation * 2
        self.num_entity = len(self.entity_to_number)
        self.facts_file = os.path.join(folder, "facts.txt")

        self.facts, self.num_fact = self._parse_triplets(self.facts_file)
        self.matrix_db = self._db_to_matrix_db(self.facts)

        if type == 'train':
            self.train_file = os.path.join(folder, "train.txt")
            self.train, self.num_train = self._parse_triplets(self.train_file)
            self.matrix_db_train = self.matrix_db
            extra_mdb = self._db_to_matrix_db(self.train)
            self.augmented_mdb = self._combine_two_mdbs(self.matrix_db, extra_mdb)
        elif type == 'valid':
            self.valid_file = os.path.join(folder, "valid.txt")
            if os.path.isfile(self.valid_file):
                self.valid, self.num_valid = self._parse_triplets(self.valid_file)
            else:
                self.valid, self.train = self._split_valid_from_train()
                self.num_valid = len(self.valid)
                self.num_train = len(self.train)
            self.matrix_db_valid = self.matrix_db
            self.augmented_mdb_valid = self.augmented_mdb
        elif type == 'test':
            self.test_file = os.path.join(folder, "test.txt")
            self.test, self.num_test = self._parse_triplets(self.test_file)
            self.matrix_db_test = self.matrix_db
            self.augmented_mdb_test = self.augmented_mdb
        else:
            assert False, "Invalid type of dataset {}".format(type)

        self.num_operator = 2 * self.num_relation

        # get rules for queries and their inverses appeared in train and test
        # Note: https://stackoverflow.com/questions/27431390/typeerror-zip-object-is-not-subscriptable
        # self.query_for_rules = list(set(next(zip(*self.train))) | set(next(zip(*self.test))) | set(next(zip(*self._augment_with_reverse(self.train)))) | set(next(zip(*self._augment_with_reverse(self.test)))))
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create a parser that maps numbers to queries and operators given queries"""
        assert(self.num_query==2*len(self.relation_to_number)==2*self.num_relation)
        parser = {"query":{}, "operator":{}}
        number_to_relation = {value: key for key, value 
                                         in self.relation_to_number.items()}
        for key, value in self.relation_to_number.items():
            parser["query"][value] = key
            parser["query"][value + self.num_relation] = "inv_" + key
        for query in range(self.num_relation):
            d = {}
            for k, v in number_to_relation.items():
                d[k] = v
                d[k + self.num_relation] = "inv_" + v
            parser["operator"][query] = d
            parser["operator"][query + self.num_relation] = d
        return parser
    
    def _numerical_encode(self):
        relation_to_number = {}
        with open(self.relation_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                relation_to_number[l[0]] = len(relation_to_number)
        
        entity_to_number = {}
        with open(self.entity_file) as f:
            for line in f:
                l = line.strip().split()
                assert(len(l) == 1)
                entity_to_number[l[0]] = len(entity_to_number)
        return relation_to_number, entity_to_number

    def _parse_triplets(self, file):
        """Convert (head, relation, tail) to (relation, head, tail)"""
        output = []
        with open(file) as f:
            for line in f:
                l = line.strip().split("\t")
                assert(len(l) == 3)
                output.append((self.relation_to_number[l[1]], 
                               self.entity_to_number[l[0]], 
                               self.entity_to_number[l[2]]))
        return output, len(output)

    def _split_valid_from_train(self):
        valid = []
        new_train = []
        for fact in self.train:
            dice = np.random.uniform()
            if dice < 0.1:
                valid.append(fact)
            else:
                new_train.append(fact)
        np.random.shuffle(new_train)
        return valid, new_train

    def _db_to_matrix_db(self, db):
        matrix_db = {r: ([[0,0]], [0.], [self.num_entity, self.num_entity]) 
                     for r in range(self.num_relation)}
        for i, fact in enumerate(db):
            rel = fact[0]
            head = fact[1]
            tail = fact[2]
            value = 1.
            matrix_db[rel][0].append([head, tail])
            matrix_db[rel][1].append(value)
        return matrix_db

    def _combine_two_mdbs(self, mdbA, mdbB):
        """Assume mdbA and mdbB contain distinct elements."""
        new_mdb = {}
        for key, value in mdbA.items():
            new_mdb[key] = value
        for key, value in mdbB.items():
            try:
                value_A = mdbA[key]
                new_mdb[key] = [value_A[0] + value[0], value_A[1] + value[1], value_A[2]]
            except KeyError:
                new_mdb[key] = value
        return new_mdb

    def _augment_with_reverse(self, triplets):
        augmented = []
        for triplet in triplets:
            augmented += [triplet, (triplet[0]+self.num_relation, 
                                    triplet[2], 
                                    triplet[1])]
        return augmented
