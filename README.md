# Supervised Variational Autoencoder for Soft Finger Force Prediction

SVAE leverages the powerful interpretability of latent variables in the original VAE model and constrain these learned cause factors to relate to our soft finger deformation where the restoring force can be measured during training
and act as a supervised signal to guide the learning of latent space.

The learned deformation related latent variables can be used for 6d force prediction, which enables tactile-based reactive grasping adaptation in underwater envrionment.

This repository contains the implementation of the Supervised Variational Autoencoder for Soft Finger Force Prediction from the paper:

**Autoencoding a Soft Touch to Learn Grasping Underwater**

*[Ning Guo](https://gabriel-ning.github.io/), [Xudong Han],[Xiaobo Liu],[Fang Wan],[Chaoyang Song]*


**Contact**

Any questions or comments contact [Doug Morrison](mailto:doug.morrison@roboticvision.org).

## Installation

This code was developed with Python 3.6 on Ubuntu 16.04.  Python requirements can installed by:

```bash
pip install -r requirements.txt
```

## Datasets

Currently, both the [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/) are supported.

### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

## Pre-trained Models

Some example pre-trained models for GG-CNN and GG-CNN2 can be downloaded from [here](https://github.com/dougsm/ggcnn/releases/tag/v0.1).  The models are trained on the Cornell grasping
dataset using the depth images.  Each zip file contains 1) the full saved model from `torch.save(model)` and 2) the weights state dict from `torch.save(model.state_dict())`. 

For example loading GG-CNN (replace ggcnn with ggcnn2 as required):

```bash
# Enter the directory where you cloned this repo
cd /path/to/ggcnn

# Download the weights
wget https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn_weights_cornell.zip

# Unzip the weights.
unzip ggcnn_weights_cornell.zip

# Load the weights in python, e.g.
python
>>> import torch

# Option 1) Load the model directly.
# (this may print warning based on the installed version of python)
>>> model = torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell')
>>> model

GGCNN(
  (conv1): Conv2d(1, 32, kernel_size=(9, 9), stride=(3, 3), padding=(3, 3))
  (conv2): Conv2d(32, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
  (conv3): Conv2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (convt1): ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  (convt2): ConvTranspose2d(8, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
  (convt3): ConvTranspose2d(16, 32, kernel_size=(9, 9), stride=(3, 3), padding=(3, 3), output_padding=(1, 1))
  (pos_output): Conv2d(32, 1, kernel_size=(2, 2), stride=(1, 1))
  (cos_output): Conv2d(32, 1, kernel_size=(2, 2), stride=(1, 1))
  (sin_output): Conv2d(32, 1, kernel_size=(2, 2), stride=(1, 1))
  (width_output): Conv2d(32, 1, kernel_size=(2, 2), stride=(1, 1))
)


# Option 2) Instantiate a model and load the weights.
>>> from models.ggcnn import GGCNN
>>> model = GGCNN()
>>> model.load_state_dict(torch.load('ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'))

<All keys matched successfully>

```

## Training

Training is done by the `train_ggcnn.py` script.  Run `train_ggcnn.py --help` to see a full list of options, such as dataset augmentation and validation options.

Some basic examples:

```bash
# Train GG-CNN on Cornell Dataset
python train_ggcnn.py --description training_example --network ggcnn --dataset cornell --dataset-path <Path To Dataset>

# Train GG-CNN2 on Jacquard Datset
python train_ggcnn.py --description training_example2 --network ggcnn2 --dataset jacquard --dataset-path <Path To Dataset>
```

Trained models are saved in `output/models` by default, with the validation score appended.

## Evaluation/Visualisation

Evaluation or visualisation of the trained networks are done using the `eval_ggcnn.py` script.  Run `eval_ggcnn.py --help` for a full set of options.

Important flags are:
* `--iou-eval` to evaluate using the IoU between grasping rectangles metric.
* `--jacquard-output` to generate output files in the format required for simulated testing against the Jacquard dataset.
* `--vis` to plot the network output and predicted grasping rectangles.

For example:

```bash
python eval_ggcnn.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --jacquard-output --iou-eval
```


## Running on a Robot

Our ROS implementation for running the grasping system see [https://github.com/dougsm/mvp_grasp](https://github.com/dougsm/mvp_grasp).

The original implementation for running experiments on a Kinva Mico arm can be found in the repository [https://github.com/dougsm/ggcnn_kinova_grasping](https://github.com/dougsm/ggcnn_kinova_grasping).