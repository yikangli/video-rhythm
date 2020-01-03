# video-rhythm
This project focuses on dealing with different sampling rates (video rhythm problem) for testing videos. The model is trained only with fixed sampling rates and the challenge is to recognize the action/event type for testing videos with different sampling rates.

# How to train
1. Download the preprocessed data for the Breakfast and extract the feature under the same folder. Name the feature folader as "deep_frame"
2. Create the train and test list by running the "create_list.py" file.
3. Train the model by running the "IndRNN_BF_baseline.py" for IndRNN model, and "Skip_IndRNN_BF.py" for SkipIndRNN model.
