# How to train
1. Download the preprocessed data for the VIRAT-2.0 6 events and extract the feature under the same folder. Name the feature folader as "deep_frame"
The Downloadable Link can be found here: https://drive.google.com/file/d/1QIPn-gzC5Nx1LVVgMmVVtGLqgV7S7qHH/view?usp=sharing
2. Modify the path and create the train and test list by running the "create_list.py" file.
3. Train the model by running the "IndRNN_VIRAT.py" for IndRNN model, and "Skip_IndRNN_VIRAT.py" for SkipIndRNN model.

# Miscellaneous
The "preprocess_data" folder is provided for collecting and extracting the bounding box area in each frame of 6 events of the VIRAT dataset with pretrained VGG16 model. The VIRAT 2.0 6 events datasaet should be downloaded first. The cropped 6 events of the VIRAT 2.0 dataset can be found here: https://drive.google.com/file/d/1GzeN7K4EzWHlTBbC6GsZhIqt6EsISWSX/view?usp=sharing

The file "App_VIRAT_train_list.txt", "App_VIRAT_test_list.txt" are the example train and test list that can be created by the "create_list.py" file. The text file will save the path for training and testing data.
