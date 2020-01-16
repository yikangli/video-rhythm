# video-rhythm
This project focuses on dealing with different sampling rates (video rhythm problem) for testing videos. The paper "Recognizing Video Events with Varying Rhythms" can be found: https://arxiv.org/abs/2001.05060  
The model is trained only with fixed sampling rates and the challenge is to recognize the action/event type for testing videos with different sampling rates.

# Requirement
1. Python 2
2. Pytorch 0.4.0

# How to train
1. Download the preprocessed data and extract the feature under the same folder. Name the feature folader as "deep_frame"
2. Modify the path and create the train and test list by running the "create_list.py" file.
3. Train the model with IndRNN model, or the SkipIndRNN model.

# Miscellaneous
The codes of extracting video frames and their corresponding features are also provided. Please download the Breakfast and VIRAT 2.0 dataset first and then preprocess the dataset with the codes.

