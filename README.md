# jbc-cough-in-a-box
This repository details the code required to replicate the results in the following three papers:
- Audio-based AI classifiers show no evidence of improved COVID-19 diagnosis over simple symptoms checkers
- A large-scale and PCR-referenced bioacoustics dataset for COVID-19
- Statistical Design and Analysis for Robust Machine Learning: A Case Study from COVID-19

_note: In the repository's current form it requires access to the an S3 environment with the currently private dataset on it. The dataset is soon to be released publicly through UKDS. When this happends the repository will be updated to fit that structure._

_note: as both the code to generate the SSAST experiments and perform openSMILE feature extration exist as a git submodule when cloning this repo, if you are intending on running the above analysis make sure to add the recursive submodule flag to the clone command:_
```bash
git clone --recurse-submodules <repo.git> 
```

## Contents
**[Data Paper](data-paper/)** --> notebook to produce summary statistics and plotly figures in UK COVID-19 Vocal Audio Dataset data descriptor.

**[SVM Baseline](SvmBaseline/)** --> code used to generate the openSMILE-SVM baseline results along with weak-robust and nearest neighbour mapping ablation studies.

**[BNN Baseline](BNNBaseline/)** --> code used to generate the ResNet-50 BNN baseline results and uncertainty metrics.

**[Code for plotting](plotting/)** --> code used to generate the plots for the three papers.

**[Utilities](utils/)** --> helper functions + main dataset class for machine learning training.


**[Unit Tests](tests/)** --> unit tests for checking validitiy of train/val/test splits and other functionality.

**[Self Supervised Audio Spectrogram Transformer](https://github.com/harrygcoppock/ssast_ciab/)** --> folder `ssast_ciab/` is a git submodule pointer to a particular commit in the ssast transformer repo used to generate the main results of the study

**[Docker](docker/)** --> code used to create the docker image for the experimental environment (also contains the requirements.txt repo if a python virtual environment is preferred)

## Docker
To make the replication of results easy we have provided a docker image of the experimental environment. To boot up a docker container run:
```bash
docker run -it --name <name_for_container> -v <location_of_git_repo>:/workspace/ --gpus=all --ipc=host harrycoppock/ciab:ciab_v4
```
This will open a new terminal inside the docker. Do not worry about having to download the docker image from the hub, the above command with handle this.

If you are on macOS please add the flag  ```--platform=linux/amd64```


### Data
We have made the dataset available subject to approval and a data sharing contract. To apply please email DataAccess@ukhsa.gov.uk and request 'The UK COVID-19 Vocal Audio Dataset'. To learn about how to apply for UKHSA data, visit: [https://www.gov.uk/government/publications/accessing-ukhsa-protected-data/accessing-ukh](https://www.gov.uk/government/publications/accessing-ukhsa-protected-data/accessing-ukhsa-protected-data). For details concerning how this dataset was collected please consult the 3 cited papers, particulary the 'data' paper.

### SSAST results
**Warning** preprocessing and training take a considerable amount of time and require access to a V100 GPU or equivalent.

To replicate the SSAST results first the audio files need to be preprocessed:
```bash
cd ssast_ciab/src/finetune/ciab/
python prep_ciab.py
```
Once this is complete then training can begin:
```bash
sh run_ciab.sh
```

### BNN results
For more more detailed description please consult the [BNN README](/BNNBaseline).

**Warning** Please note that the full run is very compute intensive, and was performed on a K4 Tesla GPU/V100 GPU with at least 64 GB of system RAM. There are options to train on sub-samples of the dataset provided in the appropriate files. The code is configured with the config file in `BNNBaseline/lib/config.py`.

To replicate BNN results, first `cd BNNBaseline/lib` and extract features with:
```bash
python extract_feat.py
```
Once complete, train the model with
```bash
python train.py
```

To evaluate results and save to the folder specified in `BNNBaseline/lib/config.py`, run
```bash
python evaluate.py
```


### SVM-Opensmile baseline
To run OpenSmile feature extraction first build the OpenSmile audio feature extraction package from source by following these [instructions](https://github.com/audeering/opensmile). Then run:
```python
python SvmBaseline/opensmile_feat_extraction.py
```
This will extract opensmile features for the test and train sets in the s3 bucket. It will save them in features/opensmile/

To run SVM classificaiton on extracted features:
```python
python SvmBaseline/svm.py
```
### Dummy config
To run experiments please fill in the fields in ./dummy_config.yaml

### Replicate experimental splits [optional]
To replicate the creation of the 3 training sets, 3 validation sets and 5 testing sets the following commands can be run:


The pipeline for generating splits is as follows:
1. Execute all cells in analysis_splits/Exploratory Analysis and Split Generation.ipynb (generates train + test splits)
2. Execute: (generates the validation sets for train). 

```
cd utils
python dataset_stats.py --create_meta=yes
cd .. 
```
  
3. Execute all cells in notebooks/matching/matching_final.ipynb (generates the matched training and test sets). 
4. Execute:   

```
cd utils
python dataset_stats.py --create_matched_validation=yes
cd .. 
```

(creates the matched validation set)

### Tests
There are no unit tests for this code base. Assert statements however feature throughout the codebase to test for expected functionality. There are a set of tests which should be run once train-test splits are created. This tests for over lapping splits, duplicate results and much more.



### Citations
This repository details the code used to create the results presented in the following three papers. Please cite.
``` 
@article{coppock2022,
  author = {Coppock, Harry and Nicholson, George and Kiskin, Ivan and Koutra, Vasiliki and Baker, Kieran and Budd, Jobie and Payne, Richard and Karoune, Emma and Hurley, David and Titcomb, Alexander and Egglestone, Sabrina and Cañadas, Ana Tendero and Butler, Lorraine and Jersakova, Radka and Mellor, Jonathon and Patel, Selina and Thornley, Tracey and Diggle, Peter and Richardson, Sylvia and Packham, Josef and Schuller, Björn W. and Pigoli, Davide and Gilmour, Steven and Roberts, Stephen and Holmes, Chris},
  title = {Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers},
  journal = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2212.08570},
  url = {https://arxiv.org/abs/2212.08570},
}

@article{budd2022,
    author={Jobie Budd and Kieran Baker and Emma Karoune and Harry Coppock and Selina Patel and Ana Tendero Cañadas and Alexander Titcomb and Richard Payne and David Hurley and Sabrina Egglestone and Lorraine Butler and George Nicholson and Ivan Kiskin and Vasiliki Koutra and Radka Jersakova and Peter Diggle and Sylvia Richardson and Bjoern Schuller and Steven Gilmour and Davide Pigoli and Stephen Roberts and Josef Packham Tracey Thornley Chris Holmes},
    title={A large-scale and PCR-referenced vocal audio dataset for COVID-19},
    year={2022},
    journal={arXiv},
    doi = {10.48550/ARXIV.2212.07738}
}

@article{Pigoli2022,
    author={Davide Pigoli and Kieran Baker and Jobie Budd and Lorraine Butler and Harry Coppock
        and Sabrina Egglestone and Steven G.\ Gilmour and Chris Holmes and David Hurley and Radka Jersakova and Ivan Kiskin and Vasiliki Koutra and George Nicholson and Joe Packham and Selina Patel and Richard Payne and Stephen J.\ Roberts and Bj\"{o}rn W.\ Schuller and Ana Tendero-Ca$\tilde{n}$adas and Tracey Thornley and Alexander Titcomb},
    title={Statistical Design and Analysis for Robust Machine Learning: A Case Study from Covid-19},
    year={2022},
    journal={arXiv},
    doi = {10.48550/ARXIV.2212.08571}
}
```
