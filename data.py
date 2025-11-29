import inspect
import yaml
import torch
import lightning as L
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Subset


class DataModule(L.LightningDataModule):
    def __init__(self, 
                 name, 
                 root: str='./data',
                 config_name: str=None,
                 transforms: list=None, 
                 batch_size: int=32, 
                 num_workers: int=4, 
                 n_train: int=0
    ) -> None:
        super().__init__()
        self.transforms = transforms if transforms is not None else []       

        config = yaml.safe_load(open("configs/data.yaml"))
        if config_name is None:
            config_name = name
        assert name in config, f"Config {config_name} not found in config."
        self.config = config[config_name]
        self.cls = {
            "flowers102": datasets.Flowers102,
            "cifar10": datasets.CIFAR10,
            "mnist": datasets.MNIST,
        }[name.lower()] 

        self.dataparams = inspect.signature(self.cls.__init__).parameters
        self.save_hyperparameters(ignore=["transforms"])

    def prepare_data(self):
        if 'split' in self.dataparams:
            for split in self.config["splits"]:
                self.cls(root=self.hparams.root, split=split, download=True)
        elif 'train' in self.dataparams:
            self.cls(root=self.hparams.root, train=True, download=True)
            self.cls(root=self.hparams.root, train=False, download=True)
        
    def setup(self, stage):
        if stage == 'fit':
            transform = v2.Compose([
                v2.ToTensor(),
                v2.Resize(self.config["resize"]),
                *self.transforms,
                v2.Normalize(self.config["mean"], self.config["std"]) 
            ])
            
            if 'split' in self.dataparams:
                self.train = self.cls(root=self.hparams.root, split='train', transform=transform)
                self.val = self.cls(root=self.hparams.root, split='val', transform=transform)
            else:
                train = self.cls(root=self.hparams.root, train=True, transform=transform)
                self.train, self.val = random_split(train, [0.8, 0.2])
            
            if self.hparams.n_train > 0:
                self.train = Subset(self.train, 
                                    torch.randperm(len(self.train))[:self.hparams.n_train]
                                    )
                self.val = Subset(self.val, 
                                    torch.randperm(len(self.val))[:self.hparams.n_train]
                                    )
       
        if stage == 'test' or 'predict':
            transform = v2.Compose([
                v2.ToTensor(),
                v2.Resize(self.config["resize"]),
                *self.transforms,
                v2.Normalize(self.config["mean"], self.config["std"]) 
            ])

            if 'split' in self.dataparams:
                self.test = self.cls(root=self.hparams.root, split='test', transform=transform)
            else:
                self.test = self.cls(root=self.hparams.root, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, persistent_workers=True)
    