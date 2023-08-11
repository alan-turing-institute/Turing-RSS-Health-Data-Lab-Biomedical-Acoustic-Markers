# The UK COVID-19 Vocal Audio Dataset - Open Access Version

This dataset is the open access version of The UK COVID-19 Vocal Audio Dataset. We point the user to [A large-scale and PCR-referenced vocal audio dataset for COVID-19](https://arxiv.org/pdf/2212.07738.pdf) and [Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers](https://arxiv.org/abs/2212.08570) for a full description of the dataset. 


## Contents

### Meta data

- **_particpant_metadata.csv_** row wise, participant identifier indexed information on particpant. Please see [A large-scale and PCR-referenced vocal audio dataset for COVID-19](https://arxiv.org/pdf/2212.07738.pdf) for the full data dictionary.
- **_audio_metadata.csv_** row wise, particpant indentifier indexed information on four recorded audio modalities. Please see [A large-scale and PCR-referenced vocal audio dataset for COVID-19](https://arxiv.org/pdf/2212.07738.pdf) for the full data dictionary.
- **_train_test_splits.csv_** row wise, particpant indetifier indexed information on train test splits for the following sets: ‘Randomised’ train and test set, Standard’ train and test set, Matched’ train and test sets, ‘Longitudinal’ test set and ‘Matched Longitudinal’ test set. Please see [Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers](https://arxiv.org/abs/2212.08570) for a full description of the train test splits. 

### The Dublin Core™ Metadata Initiative

- Title: The UK COVID-19 Vocal Audio Dataset
- Creator: The UKHSA in collaboration with The Turing-RSS Health Data Lab
- Subject: COVID-19 in Respiratory audio
- Description: Vocal records of PCR validated individuals
- Publisher: ? Government licence?
- Contributor: The UKHSA and The Alan Turing Institute
- Date: Data was collected between March 2021 to March 2022
- Type: Acoustic
- Format: .wav, .csv
- Identifier: TODO add doi
- Source: The (full) UK COVID-19 Vocal Audio Dataset available through signing of a data sharing contract with the UKHSA
- Language: English
- Relation: a related data resource - not sure what to put here
- Coverage: spatial or temporal topic of the resource, spatial applicability of the Data - not sure what to put here either
- Rights: the information about rights held in and over the resource - Government licence? 


### Citations
Please cite.

``` 
@article{budd2022,
    author={Jobie Budd and Kieran Baker and Emma Karoune and Harry Coppock and Selina Patel and Ana Tendero Cañadas and Alexander Titcomb and Richard Payne and David Hurley and Sabrina Egglestone and Lorraine Butler and George Nicholson and Ivan Kiskin and Vasiliki Koutra and Radka Jersakova and Peter Diggle and Sylvia Richardson and Bjoern Schuller and Steven Gilmour and Davide Pigoli and Stephen Roberts and Josef Packham Tracey Thornley Chris Holmes},
    title={A large-scale and PCR-referenced vocal audio dataset for COVID-19},
    year={2022},
    journal={arXiv},
    doi = {10.48550/ARXIV.2212.07738}
}
@article{coppock2022,
  author = {Coppock, Harry and Nicholson, George and Kiskin, Ivan and Koutra, Vasiliki and Baker, Kieran and Budd, Jobie and Payne, Richard and Karoune, Emma and Hurley, David and Titcomb, Alexander and Egglestone, Sabrina and Cañadas, Ana Tendero and Butler, Lorraine and Jersakova, Radka and Mellor, Jonathon and Patel, Selina and Thornley, Tracey and Diggle, Peter and Richardson, Sylvia and Packham, Josef and Schuller, Björn W. and Pigoli, Davide and Gilmour, Steven and Roberts, Stephen and Holmes, Chris},
  title = {Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers},
  journal = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2212.08570},
  url = {https://arxiv.org/abs/2212.08570},
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