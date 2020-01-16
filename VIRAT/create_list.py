import os
train_app_root = '/home/local/ASUAD/yikangli/Documents/sub_activity_dataset/code/Video_Metric_Emb/virat_vgg_data/'
test_app_root = '/home/local/ASUAD/yikangli/Documents/sub_activity_dataset/code/Video_Metric_Emb/virat_vgg_data/'
load_train_root = '/home/local/ASUAD/yikangli/Documents/sub_activity_dataset/code/Video_Metric_Emb/virat_c3d_preprocess/virat_train_test/train/'
load_test_root = '/home/local/ASUAD/yikangli/Documents/sub_activity_dataset/code/Video_Metric_Emb/virat_c3d_preprocess/virat_train_test/test/'
events = os.listdir(train_app_root)
events.sort()

file_train = open('App_VIRAT_train_list.txt','w')

for i in range(len(events)):
    label = i+1
    events_name = events[i]
    clip_root = os.path.join(load_train_root,events_name)
    clip_list = os.listdir(clip_root)
    clip_list.sort()
    for j in range(len(clip_list)):
        clip_name = clip_list[j]
        train_root = os.path.join(train_app_root,events_name+'/',clip_name)
        file_train.write(train_root + '  ')
        file_train.write(str(label))
        file_train.write('\n')

file_train.close()


file_test = open('App_VIRAT_test_list.txt','w')

for i in range(len(events)):
    label = i+1
    events_name = events[i]
    clip_root = os.path.join(load_test_root,events_name)
    clip_list = os.listdir(clip_root)
    clip_list.sort()
    for j in range(len(clip_list)):
        clip_name = clip_list[j]
        test_root = os.path.join(test_app_root,events_name+'/',clip_name)
        file_test.write(test_root + '  ')
        file_test.write(str(label))
        file_test.write('\n')

file_test.close()

        
