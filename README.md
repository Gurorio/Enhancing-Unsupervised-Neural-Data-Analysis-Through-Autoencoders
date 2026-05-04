# Master Thesis Autoencoder Models

**Disclaimer**: This README file and the code commentary were drafted with assistance from ChatGPT and reviewed by the author.

This repository contains PyTorch autoencoder models and an experiment notebook developed for a master thesis on learned latent representations from MNIST-derived image data. The code includes a fully connected autoencoder in [`autoEncoder.py`](./autoEncoder.py), convolutional autoencoders in [`ConvAE.py`](./ConvAE.py), and a full training, augmentation, inference, and benchmark workflow in [`Run_code.ipynb`](./Run_code.ipynb), in addition to the computation of the ISA-test.

The thesis computes an ISA test on the latent representations from the autoencoders trained on the constructed nonlinear data. It supports same-shape reconstruction and reconstruction into a different target shape, which is useful when the input and target representations are related but not identical.

## Interesting techniques

- **Latent representation as a first-class output**  
  Each model returns both the reconstruction and the latent vector. This makes the models useful for reconstruction tasks and downstream analysis of compressed representations.

- **Fully connected descriptor compression**  
  [`autoEncoder.py`](./autoEncoder.py) defines a feedforward encoder-decoder that maps an input vector through progressively smaller layers into a latent space, then reconstructs the original vector shape.

- **Convolutional downsampling and transposed-convolution upsampling**  
  [`ConvAE.py`](./ConvAE.py) uses [`torch.nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) for spatial encoding and [`torch.nn.ConvTranspose2d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) for learned upsampling in the decoder.

- **Runtime shape inference with dummy tensors**  
  The convolutional models infer flattened encoder dimensions by passing a zero tensor through the encoder inside [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html). This avoids hand-calculating convolution output sizes and makes the model easier to adapt to new input shapes.

- **Flatten and unflatten bridge between CNN and dense layers**  
  The CNN models use [`torch.nn.Flatten`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) and [`torch.nn.Unflatten`](https://pytorch.org/docs/stable/generated/torch.nn.Unflatten.html) to move between spatial feature maps and dense latent vectors.

- **Different input and target reconstruction shapes**  
  [`Conv_AE_diff_targ`](./ConvAE.py) supports an input shape and a separate target shape. This lets the model learn a mapping from padded or transformed inputs back to a cleaner target representation.

- **Explicit weight initialization**  
  The feedforward autoencoder uses Kaiming initialization for ReLU layers and Xavier initialization for the final sigmoid decoder layer through [`torch.nn.init`](https://pytorch.org/docs/stable/nn.init.html).

- **Reproducible data preparation and training**  
  [`Run_code.ipynb`](./Run_code.ipynb) sets seeds across NumPy, Python, and PyTorch, then uses deterministic train/validation splits and data loader generators.

- **Image augmentation pipeline**  
  The notebook builds augmented MNIST-derived samples with [Albumentations](https://albumentations.ai/), [imutils](https://github.com/PyImageSearch/imutils), custom salt-and-pepper noise, Gaussian noise, and crops.

- **Bootstrap and benchmark evaluation**  
  [`Run_code.ipynb`](./Run_code.ipynb) includes bootstrap confidence interval utilities, threshold sweeps, and ISA-style benchmark analysis for comparing latent representations.

  - **ISA-test**  
  [`Run_code.ipynb`](./Run_code.ipynb) includes the ISA-test described in the thesis and a setup for computing it on state initializations.
  

## Technologies and libraries

- [PyTorch](https://pytorch.org/) — model definition, training, tensor operations, and checkpoint loading.
- [torchvision](https://pytorch.org/vision/stable/index.html) — MNIST dataset access.
- [NumPy](https://numpy.org/) — array manipulation and numerical preprocessing.
- [pandas](https://pandas.pydata.org/) — tabular benchmark results.
- [Matplotlib](https://matplotlib.org/) — plots for reconstructions, losses, and benchmark metrics.
- [Albumentations](https://albumentations.ai/) — image augmentation.
- [imutils](https://github.com/PyImageSearch/imutils) — image rotation helpers.
- [scikit-learn](https://scikit-learn.org/stable/) — k-fold validation through `KFold`.
- Python standard library modules including [`copy`](https://docs.python.org/3/library/copy.html), [`json`](https://docs.python.org/3/library/json.html), [`math`](https://docs.python.org/3/library/math.html), [`os`](https://docs.python.org/3/library/os.html), [`pickle`](https://docs.python.org/3/library/pickle.html), and [`random`](https://docs.python.org/3/library/random.html).

No external fonts are used in the provided files.

## Project structure

```text
.
├── autoEncoder.py
├── ConvAE.py
├── Run_code.ipynb
├── data/
├── data_for_inference/
├── data_inference/
└── models_for_inference/
```

`data/` stores the MNIST dataset used by [`Run_code.ipynb`](./Run_code.ipynb).

`data_inference/` is used by the notebook when saving trained model checkpoints.

`models_for_inference/` is used by the notebook when loading saved model checkpoints for evaluation.

`data_for_inference/` stores exported latent vectors created during inference.

## MNIST data download

[`Run_code.ipynb`](./Run_code.ipynb) uses the MNIST dataset through [torchvision.datasets.MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html). When `download=True` is set, torchvision downloads the dataset into the configured root directory, which is `./data` in this repository.

The downloaded files are stored under [`data/`](./data/) and reused on later runs, so the dataset does not need to be committed to the repository. If the directory is missing, rerunning the notebook with `download=True` will recreate it.

## Model overview

[`autoEncoder.py`](./autoEncoder.py) defines `AutoEncoder_desp`, a fully connected autoencoder for vector inputs. It uses ReLU activations in the encoder and hidden decoder layers, then applies a sigmoid output layer for normalized reconstruction values.

[`ConvAE.py`](./ConvAE.py) defines two convolutional models:

- `Conv_AE` reconstructs inputs into the same spatial shape.
- `Conv_AE_diff_targ` reconstructs inputs into a different target shape.

Both convolutional models encode images into a dense latent vector, then decode that vector back into image-like tensors.

## Notebook workflow

[`Run_code.ipynb`](./Run_code.ipynb) contains the full thesis experiment pipeline:

1. Load and filter MNIST data.
2. Create padded and unpadded digit-pair representations.
3. Apply image augmentation.
4. Train feedforward and convolutional autoencoders.
5. Save model checkpoints.
6. Load trained models for inference.
7. Export latent representations.
8. Run benchmark and ISA-test analysis.
9. Plot reconstruction and evaluation results.
