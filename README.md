# EMT-LM

### Author: Shi Pan, UCL Genetics Institute

EMT-LM is a large language model that can classify epithelial-mesenchymal transition (EMT) states in single cell RNA-seq data.

![EMT-LM pipeline](https://github.com/secrierlab/EMT-LM/blob/main/EMTLM-fig.png)

### Description
Epithelial–mesenchymal plasticity plays a significant role in various biological processes including tumour progression and chemoresistance. However, the expression programmes underlying the epithelial–mesenchymal transition (EMT) in cancer are diverse, and accurately defining the EMT status of tumour cells remains a challenging task. In this study, we employed a pre-trained single-cell large language model (LLM) to develop an EMT-language model (EMT-LM) that allows us to capture discrete states within the EMT continuum in single cell cancer data. In capturing EMT states, we achieved an average Area Under the Receiver Operating Characteristic curve (AUROC) of 90% across multiple cancer types. We propose a new metric, ADESI, to aid the biological interpretability of our model, and derive EMT signatures liked with energy metabolism and motility reprogramming underlying these state switches. We further employ our model to explore the emergence of EMT states in spatial transcriptomics data, uncovering hybrid EMT niches with contrasting potential for antitumour immunity or immune evasion. Our study provides a proof of concept that LLMs can be applied to characterise cell states in single cell data, and proposes a generalisable framework to predict EMT in single cell RNA-seq that can be adapted and expanded to characterise other cellular states.

The preprint presenting this tool **Classifying epithelial-mesenchymal transition (EMT) states in single cell cancer data using large language models** is available on [biorXiv](https://www.biorxiv.org/content/10.1101/2024.08.16.608311v1).

## Environment Setup

To set up the environment, you can either use Conda or Pip:

### Conda
Run the following command to recreate the environment using the saved `conda` environment file:
```bash
conda env create -f environment.yml
```

### Pip

Run the following command to recreate the environment using the saved `pip` environment file:
```bash
pip install -r requirements.txt
```

## Usage

The code of the scMultiNet generic classifier is included in the scLLM folder. All the code for training, validating and applying the EMT-LM model is included in the Experiment folder. 

The Experiment folder is structured as follows:

### Step_0_preprocess_raw_data  
All the code for preprocessing the raw data in our manuscript, including the generation of the count matrix and the annotation file.
Please use the "0_preprocess_example.ipynb" to generate the count matrix and the annotation file for your own dataset.
And please create a Data folder in the Step_0_preprocess_raw_data folder to store the processed data.

### Step_1_train_phase_1  
All the code for training the EMT-LM model in phase 1 in our manuscript.

### Step_2_train_phase_2  
All the code for training the EMT-LM model in phase 2 in our manuscript.

### Step_3_visualise_performances  
baseline_roc_confusion.ipynb: visualise the ROC curve and the confusion matrix of the baseline models. It provides a comparison between the baseline models and the EMT-LM model.

plot_ROC_confusion.ipynb: visualise the ROC curve and the confusion matrix of the EMT-LM model for different tissue types.

### Step_4_validate_on_unseen_dataset  
All of the code for validating the EMT-LM model on the unseen dataset in our paper.

### Step_5_Embedding_Space  
Visualise the embedding space of the EMT-LM model and plot the trajectory of the EMT states in the embedding space.

### Step_6_ADESI_score  support_data
Visualise the ADESI score of the EMT-LM model.


## Contributing

If you find a bug or want to suggest a new feature for EMT-LM, please open a GitHub issue in this repository. Pull requests are also welcome!

## License

EMT-LM is released under the GNU-GPL License. This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.
