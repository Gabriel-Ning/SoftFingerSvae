import torch
import pytorch_lightning as pl
from torchvision import models
from src.datamodules.SoftFingerSequence_datamodule import SoftFingerSequenceDataModule

from src.models import VisualFingerForceNet_v2
from src.callbacks.printing_callback import MyPrintingCallback, GenerateCallback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    # VAE Learning for Soft Finger
    ## Training Data
    dm  = SoftFingerSequenceDataModule(image_folder_dir="/home/ghost/Documents/workspace/Thesis-Project/AmphibiousSoftFinger/images",
                                     lable_npy_file="/home/ghost/Documents/workspace/Thesis-Project/AmphibiousSoftFinger/force_vecs.npy",
                                     seq_len= 3,
                                     num_workers=4)
    
    dm.setup()
    
        
    ## Training Pipeline
    trainer = pl.Trainer(max_epochs = 300,gpus = [0],callbacks=[ModelCheckpoint(
        save_weights_only=True,),
        # GenerateCallback(dm.get_train_images(4), every_n_epochs=1),
        LearningRateMonitor("epoch")],)

    # defaut Model Parameter: 
    model = VisualFingerForceNet_v2()
    
    ## Model Training
    trainer.fit(model, dm)

    ## Model Evaluation
    model.eval()        
    trainer.test(model,dm)


