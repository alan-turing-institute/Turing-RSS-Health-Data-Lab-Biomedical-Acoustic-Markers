# The UK COVID-19 Vocal Audio Dataset, Open Access Edition

This dataset is the open access version of The UK COVID-19 Vocal Audio Dataset. We point the user to [A large-scale and PCR-referenced vocal audio dataset for COVID-19](https://arxiv.org/pdf/2212.07738.pdf) and [Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers](https://arxiv.org/abs/2212.08570) for a full description of the dataset. 


## Contents

### Metadata

- **_participant_metadata.csv_** row wise, participant identifier indexed information on participant demographics and health status. Please see [A large-scale and PCR-referenced vocal audio dataset for COVID-19](https://arxiv.org/pdf/2212.07738.pdf) for a full description of the dataset.
- **_audio_metadata.csv_** row wise, participant indentifier indexed information on three recorded audio modalities including audio filepaths. Please see [A large-scale and PCR-referenced vocal audio dataset for COVID-19](https://arxiv.org/pdf/2212.07738.pdf) for a full description of the dataset.
- **_train_test_splits.csv_** row wise, participant indetifier indexed information on train test splits for the following sets: ‘Randomised’ train and test set, Standard’ train and test set, Matched’ train and test sets, ‘Longitudinal’ test set and ‘Matched Longitudinal’ test set. Please see [Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers](https://arxiv.org/abs/2212.08570) for a full description of the train test splits. 

### The Dublin Core™ Metadata Initiative

- Title: The UK COVID-19 Vocal Audio Dataset, Open Access Edition.
- Creator: The UK Health Security Agency (UKHSA) in collaboration with The Turing-RSS Health Data Lab.
- Subject: COVID-19, Respiratory symptom, Other audio, Cough, Asthma, Influenza. 
- Description:  The UK COVID-19 Vocal Audio Dataset Open Access Edition is designed for the training and evaluation of machine learning models that classify SARS-CoV-2 infection status or associated respiratory symptoms using vocal audio. The UK Health Security Agency recruited voluntary participants through the national Test and Trace programme and the REACT-1 survey in England from March 2021 to March 2022, during dominant transmission of the Alpha and Delta SARS-CoV-2 variants and some Omicron variant sublineages. Audio recordings of volitional coughs and exhalations were collected in the 'Speak up to help beat coronavirus' digital survey alongside demographic, self-reported symptom and respiratory condition data, and linked to SARS-CoV-2 test results. The UK COVID-19 Vocal Audio Dataset Open Access Edition represents the largest collection of SARS-CoV-2 PCR-referenced audio recordings to date. PCR results were linked to 70,794 of 72,999 participants and 24,155 of 25,776 positive cases. Respiratory symptoms were reported by 45.62% of participants. This dataset has additional potential uses for bioacoustics research, with 11.30% participants reporting asthma, and 27.20% with linked influenza PCR test results.
- Publisher: The UK Health Security Agency (UKHSA).
- Contributor: The UK Health Security Agency (UKHSA) and The Alan Turing Institute. 
- Date: 2021-03/2022-03
- Type: Dataset
- Format:  Waveform Audio File Format audio/wave, Comma-separated values text/csv, python pickle pkl using protocol=4.
- Identifier: TODO add zenodo reserved doi
- Source: The UK COVID-19 Vocal Audio Dataset Protected Edition, accessed via application to [Accessing UKHSA protected data](https://www.gov.uk/government/publications/accessing-ukhsa-protected-data/accessing-ukhsa-protected-data).
- Language: eng
- Relation: The UK COVID-19 Vocal Audio Dataset Protected Edition, accessed via application to [Accessing UKHSA protected data](https://www.gov.uk/government/publications/accessing-ukhsa-protected-data/accessing-ukhsa-protected-data).
- Coverage: United Kingdom, 2021-03/2022-03.
- Rights: Open Government Licence version 3 (OGL v.3), © Crown Copyright UKHSA 2023.


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
