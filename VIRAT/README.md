# How to train
1. Download the preprocessed data for the VIRAT-2.0 6 events and extract the feature under the same folder. Name the feature folader as "deep_frame"
The Downloadable Link can be found here: https://drive.google.com/file/d/1LdHVFvnPL-EqHySDru5M2VaO9ajcPPaE/view?usp=sharing
2. Modify the path and create the sample list by running the "create_list.py" file.
3. Train the model by running the "IndRNN_VIRAT.py" for IndRNN model, and "Skip_IndRNN_VIRAT.py" for SkipIndRNN model.

# Miscellaneous
1. The matlab codes within "preprocess_data" folder are provided for collecting and extracting the bounding box area in each frame of 6 events of the VIRAT dataset. The VIRAT 2.0 6 events datasaet should be downloaded first. The cropped 6 events of the VIRAT 2.0 dataset can be found here: https://drive.google.com/file/d/1GzeN7K4EzWHlTBbC6GsZhIqt6EsISWSX/view?usp=sharing
Or the original VIRAT 2.0 dataset can be found here: https://viratdata.org/

2. The file "App_list.txt" is the example list for the provided features. Since the official VIRAT dataset does not provide the training and testing splits. Therefore, we randomly split the training and testing samples with 2:1 ratio by running the "load_VIRAT_data.py" file. We recommend to run the training file several times to obtain an average accuracy for the VIRAT dataset.  
