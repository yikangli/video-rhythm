# video-rhythm
This project focuses on dealing with different sampling rates (video rhythm problem) for testing videos. The model is trained only with fixed sampling rates and the challenge is to recognize the action/event type for testing videos with different sampling rates.

# How to train
1. Download the preprocessed data for the Breakfast and extract the feature under the same folder. Name the feature folader as "deep_frame"
2. Modify the path and create the train and test list by running the "create_list.py" file.
3. Train the model by running the "IndRNN_BF_baseline.py" for IndRNN model, and "Skip_IndRNN_BF.py" for SkipIndRNN model.

# Miscellaneous
We also provide the "extract_deep_features.py" file for extracting the deep features of BreakFast dataset with pretrained ResNet152 model. The BreakFast datasaet should be downloaded first and stored in frame form. http://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/
