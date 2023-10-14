from pathlib import Path
import os
import sys

from comet_ml import Experiment
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.dataloader import load_dataset, get_data_loaders
from utils.lightning_module import LitModel
from utils.general import *


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

COMETML_KEY = os.getenv('COMETML_KEY')

def main(opt):
    if opt.use_comet_ml:
        experiment = Experiment(
            api_key=COMETML_KEY,
            project_name="PSD_ML",
            workspace="szymkwiatkowski"
        )
    data_ann = read_yaml(opt.data)        
    datasets = load_dataset(data_ann, ROOT)
    data_loaders = get_data_loaders(datasets=datasets, datasets_names=datasets.keys(), batch_size=opt.batch_size)
    
    model = LitModel(data_ann['nc'])
    
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', verbose=True)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=opt.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['val'])
    
    if opt.use_comet_ml:
        experiment.end()

if __name__ == '__main__':
    opt = parse_opt(ROOT)
    main(opt)
