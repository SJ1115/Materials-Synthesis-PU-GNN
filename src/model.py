import torch
import torch.nn as nn

def FC(dim, init:nn.init=None,):
    module = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Softplus())
    if init != None:
        init(module[0].weight)
    return module

class GraphConv(nn.Module):
    def __init__(self, config,):
        super(GraphConv, self).__init__()
        self.atom_dim = config.atom_dim
        self.bond_dim = config.bond_dim
        self.FC_full = nn.Linear(2*self.atom_dim+self.bond_dim,
                                 2*self.atom_dim)
        self.Sig = nn.Sigmoid()
        self.SPlus1 = nn.Softplus()
        self.BNorm1 = nn.BatchNorm1d(2*self.atom_dim, affine=False)
        self.BNorm2 = nn.BatchNorm1d(self.atom_dim, affine=False)
        self.SPlus2 = nn.Softplus()

    def forward(self, atom, bond, bond_idx):
        N, M = bond_idx.shape
        # Convolution
        atom_bond = atom[bond_idx, :]
        total_bond = torch.cat([
            atom.unsqueeze(1).expand(N, M, self.atom_dim),
            atom_bond,
            bond], dim=2) # N*M*(2dimA+dimB)
        
        #print(total_bond.size())
        total_gated = self.FC_full(total_bond) # transpose() for BNorm1
        #print(total_gated.size())
        total_gated = total_gated.transpose(1,2).contiguous()
        #print(total_gated.size())
        total_gated = self.BNorm1(total_gated)
        total_gated = total_gated.transpose(1,2).contiguous()
        
        gated_filter, gated_core = total_gated.chunk(2, dim=2) # N*M*dimA
        gated_filter, gated_core = self.Sig(gated_filter), self.SPlus1(gated_core)
        sumed_bond = torch.sum(gated_filter * gated_core, dim=1)
        sumed_bond = self.BNorm2(sumed_bond)
        out = self.SPlus2(atom + sumed_bond)
        return out

class Seq_GC(nn.Module):
    def __init__(self, config,):
        super(Seq_GC, self).__init__()
        self.Seq = nn.ModuleList([GraphConv(config)
                                    for _ in range(config.conv_depth)])
    
    def forward(self, atom, bond, bond_idx):
        for layer in self.Seq:
            atom = layer(atom, bond, bond_idx)
        return atom

class CrystalGraphConvNet(nn.Module):
    def __init__(self, config,):
        super(CrystalGraphConvNet, self).__init__()
        self.Embedding = nn.Linear(config.atom_in_dim, config.atom_dim)
        self.Convs = Seq_GC(config)
        
        OutConv = [nn.Softplus(),
            nn.Linear(config.atom_dim, config.hid_dim),
            nn.Softplus()]
        
        Penultimate = [nn.Dropout(config.dropout)]
        for _ in range(config.n_hid-1):
            Penultimate.append(FC(config.hid_dim, ))
        
        fin_dim = 2 if config.is_cls else 1
        self.Out = nn.Linear(config.hid_dim, fin_dim)
        
        self.OutConv = nn.Sequential(*OutConv)
        self.Penultimate = nn.Sequential(*Penultimate)
                
        self.final_fea = None ## buffer for feature visualization

    def forward(self, atom, bond, bond_idx, atom_idx):
        atom = self.Embedding(atom)
        atom = self.Convs(atom, bond, bond_idx)
        crst = self.pooling(atom, atom_idx)
        crst = self.OutConv(crst)
        crst = self.Penultimate(crst)
        out  = self.Out(crst)

        self.final_fea = crst
        return out

    def pooling(self, atom, atom_idx):
        assert sum([len(idx) for idx in atom_idx]) == atom.data.shape[0]
        pool = [torch.mean(atom[idx], dim=0, keepdim=True)
                      for idx in atom_idx]
        return torch.cat(pool, dim=0)