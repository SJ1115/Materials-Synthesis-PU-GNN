import sys
sys.path.insert(0, "../")

import argparse
import warnings
warnings.filterwarnings('ignore')


from config import config
from src.util import callpath, terminal_bool
from src.trainer import Trainer
from src.model import CrystalGraphConvNet as CGCNN
from torch import nn, optim

parser = argparse.ArgumentParser(description='Synthesizability prediction from PU-learning by CGCNN')
parser.add_argument('--device', default='cpu', type=str, choices = ['cpu', 'cuda:0', 'cuda:1', 'cuda'],
                    help="device where training is executed")
parser.add_argument('--train_id', default='id_prop_train.csv', type=str,
                    help="filename of train.csv in data folder")
parser.add_argument('--model_out', default='sample', type=str,
                    help="filename of model out file in result folder")

parser.add_argument("--lab_sm", default=0.1, type=float, help="label smoothing rate")

parser.add_argument("--if_test", default='f', type=str, choices=['f','F','t','T','y','Y','n','N'], help="whether to test")
parser.add_argument('--test_id', default='id_prop_test.csv', type=str,
                    help="filename of train.csv in data folder")
parser.add_argument('--test_dir', default='sample.csv', type=str,
                    help="filename of model out file in result folder")


option = parser.parse_args()
config.device = option.device

model = CGCNN(config)
optimizer = optim.SGD(model.parameters(), lr=0.01,)

trainer = Trainer(
    model, nn.CrossEntropyLoss(label_smoothing=option.lab_sm), optimizer, optim.lr_scheduler.MultiStepLR(optimizer, [100], gamma=.1),
    cif_dir=callpath("./data/cif/"),
    atom_init_file=callpath("./data/atom_init.json"),
    config=config
)


trainer.PU_ensemble_train(
        callpath("./data/" + option.train_id),
        result_dir=callpath("./result/model/" + option.model_out))

if terminal_bool(option.if_test):
    trainer.test(
        test_prop_file=callpath("./data/" + option.test_id),
        result_csv=callpath("./result/score/" + option.test_csv)
    )