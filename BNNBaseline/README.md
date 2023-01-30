# ResNet-50 BNN class
### Contains:
* Dataset partitioning as defined in paper [to be updated to URL]
* Feature extraction to extract log-mel filterbank features
* Model training and evaluation
* Uncertainty quantification and meta-analysis

### Code structure:
* Code configuration in `lib/config.py`
* Feature extraction in `lib/extract_feat.py`
* Model training loop in `lib/train_model.py`
* Model evaluation and meta-analysis in `lib/evaluate_model.py`

### Requirements:
This module is designed to be run with the dockerfile supplied in the main README. If you wish to skip the docker file installation and only use this package, the code was developed with the `conda` environment on AWS of `pytorch_latest_p36` with an additional command of:
```bash
conda install -c conda-forge -y librosa
```

### How to configure code:
* Select the parameters of the feature transform you wish to use in `lib/vggish/vggish_params.py`.
* You may also bypass this and implement your own features directly in `lib/extract_feat.py`.
* Select the output directories, and model hyperparameter options in `lib/config.py`.

### About the model
#### Introduction
ResNet-50 has seen success in many computer vision tasks, and has shown state-of-the-art performance in related acoustic methods [acoustic mosquito detection](https://arxiv.org/abs/2110.07607), [heart murmur detection](https://cinc.org/2022/Program/accepted/355_Preprint.pdf).

#### Feature extraction
The feature extraction pipeline is imported in `lib/vggish` from [torch vggish](https://github.com/harritaylor/torchvggish), which is a port from [Audioset](https://arxiv.org/abs/1609.09430).

#### Training loop
The model is trained and early stopped according to the pre-defined splits of `train`, `val` and `test` for each carefully selected data partition. These are created upon feature extraction in `extract_feat.py`. Please refer to the papers [URL here] and [here] for a thorough description of the reasoning to create these data partitions. By default, we use the `BCELoss()` criterion with the `Adam` optimiser with the learning rate set in `config.py`.

#### Bayesian Neural Network modification
We modify the final layers for compatibility with our data in `lib/ResNetSource.py` Furthermore, we have augmented the construction blocks `BasicBlock()` and `Bottleneck()`, as well as the overall model construction, to feature dropout layers to act as an approximation for the model posterior at test-time. Dropout is implemented implicitly in `ResNetSource.py`, to not interfere with the behaviour of `model.eval()`, which by default disables dropout layers at test time, removing the necessary stochastic component. The final `self.fc1 layer` is of size `[2048,N]`, where `N` is the number of classes for the cross-entropy loss function, or 1 if used with the binary cross-entropy loss (default). A quick way to check the requirement is to `print x.shape()` before the creation of the `fc1 layer`.

#### Uncertainty quantification and evaluation
To help understand the role of predictive entropy and mutual information in the outputs of a Bayesian neural network, I strongly recommend reading [How do we go about analysing uncertainty?](https://adamcobb.github.io/journal/bnn.html)


