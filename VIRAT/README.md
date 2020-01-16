# How to train
1. Download the preprocessed data for the VIRAT-2.0 6 events and extract the feature under the same folder. Name the feature folader as "deep_frame"
The Downloadable Link can be found here: https://drive.google.com/file/d/1QIPn-gzC5Nx1LVVgMmVVtGLqgV7S7qHH/view?usp=sharing
2. Modify the path and create the train and test list by running the "create_list.py" file.
3. Train the model by running the "IndRNN_VIRAT.py" for IndRNN model, and "Skip_IndRNN_VIRAT.py" for SkipIndRNN model.

# Miscellaneous
The "extract_deep_features.py" file is provided for extracting the deep features of BreakFast dataset with pretrained ResNet152 model. The BreakFast datasaet should be downloaded first and stored in frame form. http://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/

The file "App_VIRAT_train_list.txt", "App_VIRAT_test_list.txt" are the example train and test list that can be created by the "create_list.py" file. The text file will save the path for training and testing data.
