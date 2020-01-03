import os
import pdb
import scipy.io as sio

app_root = './deep_frame/'
label_root = './Label/'
events = os.listdir(app_root)
events.sort()

app_file_train = open('App_train_TSN_list.txt','w')
label_file_train = open('Label_train_TSN_list.txt','w')

for events_name in events[13:]:
    app_camera_root = os.path.join(app_root,events_name)
    app_camera_list = os.listdir(app_camera_root)
    app_camera_list.sort()

    label_camera_root = os.path.join(label_root,events_name)
    label_camera_list = os.listdir(label_camera_root)
    label_camera_list.sort()
    for camera_name in app_camera_list:
        app_video_root = os.path.join(app_root,events_name+'/',camera_name)
        app_video_name = os.listdir(app_video_root)
        app_video_name.sort()

        label_video_root = os.path.join(label_root,events_name+'/',camera_name)
        label_video_name = os.listdir(label_video_root)
        label_video_name.sort()
        for i in app_video_name:
            app_train_root = os.path.join(app_root,events_name+'/',camera_name+'/',i)
            label_train_root = os.path.join(label_root,events_name+'/',camera_name+'/',i)

            app_file_train.write(app_train_root + '\n')
            label_file_train.write(label_train_root + '\n')


app_file_train.close()
label_file_train.close()


app_file_test = open('App_test_TSN_list.txt','w')
label_file_test = open('Label_test_TSN_list.txt','w')


for events_name in events[:13]:
    app_camera_root = os.path.join(app_root,events_name)
    app_camera_list = os.listdir(app_camera_root)
    app_camera_list.sort()

    label_camera_root = os.path.join(label_root,events_name)
    label_camera_list = os.listdir(label_camera_root)
    label_camera_list.sort()
    for camera_name in app_camera_list:
        app_video_root = os.path.join(app_root,events_name+'/',camera_name)
        app_video_name = os.listdir(app_video_root)
        app_video_name.sort()

        label_video_root = os.path.join(label_root,events_name+'/',camera_name)
        label_video_name = os.listdir(label_video_root)
        label_video_name.sort()
        for i in app_video_name:
            app_test_root = os.path.join(app_root,events_name+'/',camera_name+'/',i)
            label_test_root = os.path.join(label_root,events_name+'/',camera_name+'/',i)

            app_file_test.write(app_test_root + '\n')
            label_file_test.write(label_test_root + '\n')


app_file_test.close()
label_file_test.close()

        
