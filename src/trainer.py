import os
from src.loader import CIFData, collate_pool

import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 cif_dir, atom_init_file,
                 config,):
        self.model = model.to(config.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.cif_dir = cif_dir
        self.atom_init_file = atom_init_file
        
        self.config = config
        self.device = config.device
        return
    
    def PU_train(self, train_prop_file, result_file=None):
        bagging = self.config.PU_iter
        epoch   = self.config.epoch
        
        b, e = self._load_model(result_file)

        dataset = CIFData(self.config, self.cif_dir, train_prop_file, self.atom_init_file)
        dataset.PU_split()
        loader = DataLoader(dataset, self.config.batch_size, shuffle=True, collate_fn=collate_pool)

        loss_lst = None
        if self.config.use_board:
            writer = SummaryWriter() #f'./{self.model._get_name()}'
            loss_lst = []
            t = 0
        if self.config.use_tqdm:
            timecheck = tqdm(total=bagging*epoch*loader.__len__())
            timecheck.update((b*epoch + e) * loader.__len__())
            loss_lst = []

        while b < bagging:
            
            while e < epoch:
                for data in loader:
                    (atom, bond, bond_idx, atom_idx), target, cif_idx = data
                    atom, bond, bond_idx, atom_idx, target = atom.to(self.device), bond.to(self.device), \
                        bond_idx.to(self.device), [idx.to(self.device) for idx in atom_idx], target.squeeze().to(torch.long).to(self.device)
                    
                    predict = self.model(atom, bond, bond_idx, atom_idx)
                    
                    loss = self.criterion(predict, target)
                    if loss_lst != None:
                        loss_lst.append(loss.item())
                        if len(loss_lst) > 30:
                            loss_lst.pop(0)
                        loss_score = np.mean(loss_lst)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    
                    if self.config.use_board:
                        writer.add_scalar("loss", loss_score, t)
                        writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], t)
                        t += 1
                    if self.config.use_tqdm:
                        timecheck.set_description(f"loss:{loss_score:.3f} at bag {b}epoch {e}")
                        timecheck.update(1)
                    
                if result_file != None:
                    with open(result_file, 'wb') as f:
                        torch.save({
                        'model'  :self.model.state_dict(),
                        'optim'  :self.optimizer.state_dict(),
                        'bagging':b+1,
                        'epoch'  :e+1
                    }, f)
                
                self.scheduler.step()
                e += 1

            if b != bagging-1:
                dataset.PU_split()
                loader = DataLoader(dataset, self.config.batch_size, shuffle=True, collate_fn=collate_pool)
            b += 1
            e = 0

        return
    
    def test(self, test_prop_file, model_pt=None, result_csv=None):
        
        self._load_model(model_pt)
        self.model.eval()
        dataset = CIFData(self.config, self.cif_dir, test_prop_file, self.atom_init_file)
        
        loader = DataLoader(dataset, self.config.batch_size, collate_fn=collate_pool)

        if self.config.use_tqdm:
            timecheck = tqdm(total=loader.__len__())
        
        score, mat_idx = [], []
        with torch.no_grad():
            for data in loader:
                (atom, bond, bond_idx, atom_idx), target, cif_idx = data
                atom, bond, bond_idx, atom_idx = atom.to(self.device), bond.to(self.device), \
                    bond_idx.to(self.device), [idx.to(self.device) for idx in atom_idx]
                
                predict = self.model(atom, bond, bond_idx, atom_idx)
                
                score += predict.softmax(dim=1)[:,1].tolist()
                mat_idx += cif_idx
                
                if self.config.use_tqdm:
                    timecheck.update()
        if result_csv != None:
            with open(result_csv, "w") as f:
                f.write("ID,CLscore\n")
                for m, s in zip(mat_idx, score):
                    f.write(f"{m},{s}\n")
        return mat_idx, score
    
    def _load_model(self, filename):
        try:
            with open(filename, "rb") as f:
                load = torch.load(f, map_location=self.device)
            self.model.load_state_dict(load['model'])
            self.optimizer.load_state_dict(load['optim'])
            bagging= load['bagging']
            epoch  = load['epoch']
        except:
            bagging, epoch = 0, 0
        return bagging, epoch
    
    ## Ensemble 
    def PU_ensemble_train(self, train_prop_file, result_dir="/sample"):
        n_bag = self.config.n_bag
        bagging = self.config.bag_PU_iter
        epoch   = self.config.epoch

        dataset = CIFData(self.config, self.cif_dir, train_prop_file, self.atom_init_file)
        dataset.PU_split()
        loader = DataLoader(dataset, self.config.batch_size, shuffle=True, collate_fn=collate_pool)

        loss_lst = None
        if self.config.use_board:
            writer = SummaryWriter() #f'./{self.model._get_name()}'
            loss_lst = []
            t = 0
        if self.config.use_tqdm:
            timecheck = tqdm(total=n_bag*bagging*epoch*loader.__len__())
            loss_lst = []

        # make dir and save starting point
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        with open(result_dir+"/start.pt", 'wb') as f:
            torch.save({
                'model'  :self.model.state_dict(),
                'optim'  :self.optimizer.state_dict(),
            }, f)
        
        for n in range(n_bag):
            for b in range(bagging):
                for e in range(epoch):
                    for data in loader:
                        (atom, bond, bond_idx, atom_idx), target, cif_idx = data
                        atom, bond, bond_idx, atom_idx, target = atom.to(self.device), bond.to(self.device), \
                            bond_idx.to(self.device), [idx.to(self.device) for idx in atom_idx], target.squeeze().to(torch.long).to(self.device)
                        
                        predict = self.model(atom, bond, bond_idx, atom_idx)
                        
                        loss = self.criterion(predict, target)
                        if loss_lst != None:
                            loss_lst.append(loss.item())
                            if len(loss_lst) > 30:
                                loss_lst.pop(0)
                            loss_score = np.mean(loss_lst)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                        
                        if self.config.use_board:
                            writer.add_scalar("loss", loss_score, t)
                            writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], t)
                            t += 1
                        if self.config.use_tqdm:
                            timecheck.set_description(f"loss:{loss_score:.3f} at bag {n}epoch {b}-{e}")
                            timecheck.update(1)
                        
                    with open(result_dir + f"/{n+1}.pt", 'wb') as f:
                            torch.save({
                            'model'  :self.model.state_dict(),
                            'optim'  :self.optimizer.state_dict(),
                            'bagging':b+1,
                            'epoch'  :e+1
                        }, f)
                    
                    self.scheduler.step()

                # mix and sample new set
                dataset.PU_split()
                loader = DataLoader(dataset, self.config.batch_size, shuffle=True, collate_fn=collate_pool)
            ## reset model
            if n < n_bag-1:
                _ = self._load_model(result_dir+"/start.pt")
        return
    
    def ensemble_test(self, test_prop_file, model_dir, result_csv=None):
        model_pt = []
        for file in os.listdir(model_dir):
            if file == "start.pt":
                continue
            if file.endswith(".pt"):
                model_pt.append(os.path.join(model_dir, file))
        dataset = CIFData(self.config, self.cif_dir, test_prop_file, self.atom_init_file)
        
        loader = DataLoader(dataset, self.config.batch_size, shuffle=False,collate_fn=collate_pool)

        if self.config.use_tqdm:
            timecheck = tqdm(total=loader.__len__()*len(model_pt))
        
        scores = {}
        for pt in model_pt:
            self._load_model(pt)
            self.model.eval()
            with torch.no_grad():
                for data in loader:
                    (atom, bond, bond_idx, atom_idx), target, cif_idx = data
                    atom, bond, bond_idx, atom_idx = atom.to(self.device), bond.to(self.device), \
                        bond_idx.to(self.device), [idx.to(self.device) for idx in atom_idx]
                    
                    predict = self.model(atom, bond, bond_idx, atom_idx)
                    
                    score = predict.softmax(dim=1)[:,1].tolist()
                    for i, cif_id in enumerate(cif_idx):
                        if cif_id in scores:
                            scores[cif_id].append(score[i])
                        else:
                            scores[cif_id] = [score[i]]
                    
                    if self.config.use_tqdm:
                        timecheck.update()
        for cif_id in scores:
            scores[cif_id] = np.mean(scores[cif_id])
        if result_csv != None:
            with open(result_csv, "w") as f:
                f.write("ID,CLscore\n")
                for cif_id, score in scores.items():
                    f.write(f"{cif_id},{score}\n")
        return scores