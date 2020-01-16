# How to train
1. Download the preprocessed data for the Breakfast and extract the feature under the same folder. Name the feature folader as "deep_frame"
The Downloadable Link can be found here: https://drive.google.com/open?id=1RFuglmNPl3DqEKp6xtd25xkssFIHyhMU
2. Modify the path and create the train and test list by running the "create_list.py" file.
3. Train the model by running the "IndRNN_BF_baseline.py" for IndRNN model, and "Skip_IndRNN_BF.py" for SkipIndRNN model.

# Miscellaneous
The "extract_deep_features.py" file is provided for extracting the deep features of BreakFast dataset with pretrained ResNet152 model. The BreakFast datasaet should be downloaded first and stored in frame form. http://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/

The file "App_train_TSN_list.txt", "App_test_TSN_list.txt", "Label_train_TSN_list.txt", and "Label_test_TSN_list.txt" are the example train and test list that can be created by the "create_list.py" file. The text file will save the path for training and testing data.
