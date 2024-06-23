# DeepGenerativeAttack

This is the code for ICJB 2024 paper Deep Generative Attacks and Countermeasures for Data-Driven Writer-Dependent Static Signature Verification. 

# Generation process

As discussed in the paper, we used CGAN and VAE to generate synthetic signatures. The codes included in this repository is made for the CEDAR dataset from https://cedar.buffalo.edu/handwriting/HRdatabase.html. Make sure to check the required paths for the files. 

To run generation with, run file CGAN generation. The same with VAE generation. For controlled SSIM generation with VAE as discussed in the paper, see the VAE SSI file. 

The datasets are preprocessed using OTSU binarializing as put in file utils.py. 

# Preprocessing and training/testing data

To preprocess the data, run the file preprocessing_data.py. The folders used in the codes are downloaded from *[link](https://drive.google.com/drive/folders/1KbbJ5pCx9CVjlFgt28j4bY9KaFcELHb_)*. The generated data folders got 9 generated images for each forgery signatures from CEDAR. After running the files, the user should get 3 numpy files containing the processed real, forg and generated data as well as the average SSIM score for the generated data. 

Follow the instructions from the file DenseNet201_single.ipynb to run the baseline model DenseNet201 and test it with the generated attack. For the testing with retrained, see the demo in DenseNet201_cgan_retrain_cgan file.ipynb. Similar preprocessing, training/testing protocols are done with other datasets, generated datasets and model architectures. 

# Synthetic data from BHSIG260 and CEDAR datasets links:

The synthetic data included in the paper are located in *[link](https://drive.google.com/drive/folders/1KbbJ5pCx9CVjlFgt28j4bY9KaFcELHb_)*. 


