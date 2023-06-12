import csv
import functools
import json
import os
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

import json
from tqdm import tqdm
import pandas as pd

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


def split_bagging(id_prop,start, bagging_size, folder):
    df = pd.read_csv(id_prop, header=None)

    # Split positive/unlabeled data
    exp = []
    vir = []
    for i in range(len(df)):
        if df[1][i] == 1:
            exp.append(df[0][i])
        elif df[1][i] == 0:
            vir.append(df[0][i])
        else:
            raise Exception("ERROR: prop value must be 1 or 0")

    positive = pd.DataFrame()
    positive[0] = exp
    positive[1] = [1 for _ in range(len(exp))]

    unlabeled = pd.DataFrame()
    unlabeled[0] = vir
    unlabeled[1] = [0 for _ in range(len(vir))]

    # Sample positive data for validation and training
    valid_positive = positive.sample(frac=0.2,random_state=1234)
    train_positive = positive.drop(valid_positive.index)

    os.makedirs(folder, exist_ok=True)

    # Sample negative data for training
    for i in tqdm(range(start,start+bagging_size)):
        # Randomly labeling to negative
        negative = unlabeled.sample(n=len(positive[0]))
        valid_negative = negative.sample(frac=0.2,random_state=1234)
        train_negative = negative.drop(valid_negative.index)

        valid = pd.concat([valid_positive,valid_negative])
        valid.to_csv(os.path.join(folder, 'id_prop_bag_'+str(i+1)+'_valid.csv'), mode='w', index=False, header=False)

        train = pd.concat([train_positive,train_negative])
        train.to_csv(os.path.join(folder, 'id_prop_bag_'+str(i+1)+'_train.csv'), mode='w', index=False, header=False)

    # Generate unlabeled data
        test_unlabel = unlabeled.drop(negative.index)
        test_unlabel.to_csv(os.path.join(folder, 'id_prop_bag_'+str(i+1)+'_test-unlabeled.csv'), mode='w', index=False, header=False)


def bootstrap_aggregating(bagging_size, prediction=False):

    predval_dict = {}

    print("Do bootstrap aggregating for %d models.............." % (bagging_size))
    for i in range(1, bagging_size+1):
        if prediction:
            filename = 'test_results_prediction_'+str(i)+'.csv'
        else:
            filename = 'test_results_bag_'+str(i)+'.csv'
        df = pd.read_csv(os.path.join(filename), header=None)
        id_list = df.iloc[:,0].tolist()
        pred_list = df.iloc[:,2].tolist()
        for idx, mat_id in enumerate(id_list):
            if mat_id in predval_dict:
                predval_dict[mat_id].append(float(pred_list[idx]))
            else:
                predval_dict[mat_id] = [float(pred_list[idx])]

    print("Writing CLscore file....")
    with open('test_results_ensemble_'+str(bagging_size)+'models.csv', "w") as g:
        g.write("id,CLscore,bagging")                                       # mp-id, CLscore, # of bagging size

        for key, values in predval_dict.items():
            g.write('\n')
            g.write(key+','+str(np.mean(np.array(values)))+','+str(len(values)))
    print("Done")

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomEmbedding(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def encode(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomEmbedding_JSON(AtomEmbedding):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomEmbedding_JSON, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    def __init__(self, config,
                 cif_dir,
                 id_prop_file='id_prop.csv',
                 atom_init_file='atom_init.json'):
        self.max_neighbor, self.radius = config.max_neighbor, config.radius
        
        assert os.path.exists(cif_dir), 'cif_dir does not exist!'
        self.cif_dir = cif_dir
        
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f); next(reader)
            id_prop_data = [row for row in reader]    # [['mp-69', '0'], ['mp-80', '1'], ['mp-697196', '0'] ... ]
        self.id_prop_P = np.array([data for data in id_prop_data if data[1] == '1'])
        self.id_prop_U = np.array([data for data in id_prop_data if data[1] == '0'])
        
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.AE = AtomEmbedding_JSON(atom_init_file)
        
        self.GDF = GaussianDistance(dmin=config.dmin, dmax=self.radius, step=config.step)

        self.id_prop_bag = id_prop_data

    def __len__(self):
        return len(self.id_prop_bag)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_bag[idx]
        crystal = Structure.from_file(self.cif_dir+cif_id+'.cif')
        
        atom = np.vstack([self.AE.encode(crystal[i].specie.number)
                              for i in range(len(crystal))])
        
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        bond_idx, bond = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_neighbor: ## Pad if Too Short
                warnings.warn(f'{cif_id} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase radius.')
                bond_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_neighbor - len(nbr)))
                bond.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_neighbor -
                                                     len(nbr)))
            else: ## Cut if Too Long
                bond_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_neighbor])))
                bond.append(list(map(lambda x: x[1],
                                        nbr[:self.max_neighbor])))
        bond = np.array(bond)
        bond = self.GDF.expand(bond)
        atom = torch.Tensor(atom)
        bond = torch.Tensor(bond)
        bond_idx = torch.LongTensor(bond_idx)
        target = torch.LongTensor([int(target)])
        return (atom, bond, bond_idx), target, cif_id
    
    def PU_split(self, ):
        assert len(self.id_prop_U)*len(self.id_prop_P) != 0, "This data is strictly skewed! This data may be for the test, not for PU train."
        id_prop_N = self.id_prop_U[np.random.choice(
            len(self.id_prop_U), len(self.id_prop_P), replace=False)]
        PU_bag = np.concatenate((self.id_prop_P,id_prop_N))
        self.id_prop_bag = np.random.permutation(PU_bag).tolist()
        return
