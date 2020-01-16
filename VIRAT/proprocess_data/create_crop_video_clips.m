clc
clear
close all

fid = fopen('../docs/README_annotations_evaluations.csv'); % 'Data' worksheet
aline = fgetl(fid); % get a line (header?)
acols = length(find(aline==','))+1; % number of columns
aformat = repmat('%s ', 1, acols); % create format
content = textscan(fid,aformat,'Delimiter',',');
fclose(fid);

LAV = str2double(content{10});
UAV = str2double(content{11});
OAV = str2double(content{12});
CAV = str2double(content{13});
GIV = str2double(content{14});
GOV = str2double(content{15});

LAV_clips = find(LAV(1:end-1)~=0);
UAV_clips = find(UAV(1:end-1)~=0);
OAV_clips = find(OAV(1:end-1)~=0);
CAV_clips = find(CAV(1:end-1)~=0);
GIV_clips = find(GIV(1:end-1)~=0);
GOV_clips = find(GOV(1:end-1)~=0);

LAV_clip_names = content{1}(LAV_clips);
UAV_clip_names = content{1}(UAV_clips);
OAV_clip_names = content{1}(OAV_clips);
CAV_clip_names = content{1}(CAV_clips);
GIV_clip_names = content{1}(GIV_clips);
GOV_clip_names = content{1}(GOV_clips);

%% Extract Events
DIR_TOP = '..';

EXT_EVENT = '.viratdata.events.txt';
EXT_VIDEO = '.mp4';

for i = 1 : numel(LAV_clip_names)
    file_video = [DIR_TOP filesep 'videos_original' filesep [LAV_clip_names{i}, EXT_VIDEO]];
    file_event = [DIR_TOP filesep 'annotations' filesep [LAV_clip_names{i}, EXT_EVENT]];
    dir_output = [DIR_TOP filesep '6_events_crop_video' filesep 'LAV'];
    extract_6_events_save_video(file_event, file_video, dir_output,LAV_clip_names{i},'LAV');
end
fprintf('Finish Extracting LAV event \n');

for i = 1 : numel(UAV_clip_names)
    file_video = [DIR_TOP filesep 'videos_original' filesep [UAV_clip_names{i}, EXT_VIDEO]];
    file_event = [DIR_TOP filesep 'annotations' filesep [UAV_clip_names{i}, EXT_EVENT]];
    dir_output = [DIR_TOP filesep '6_events_crop_video' filesep 'UAV'];
    extract_6_events_save_video(file_event, file_video, dir_output,UAV_clip_names{i},'UAV');
end
fprintf('Finish Extracting UAV event \n');

for i = 1 : numel(OAV_clip_names)
    file_video = [DIR_TOP filesep 'videos_original' filesep [OAV_clip_names{i}, EXT_VIDEO]];
    file_event = [DIR_TOP filesep 'annotations' filesep [OAV_clip_names{i}, EXT_EVENT]];
    dir_output = [DIR_TOP filesep '6_events_crop_video' filesep 'OAV'];
    extract_6_events_save_video(file_event, file_video, dir_output,OAV_clip_names{i},'OAV');
end
fprintf('Finish Extracting OAV event \n');

for i = 23 : numel(CAV_clip_names)
    file_video = [DIR_TOP filesep 'videos_original' filesep [CAV_clip_names{i}, EXT_VIDEO]];
    file_event = [DIR_TOP filesep 'annotations' filesep [CAV_clip_names{i}, EXT_EVENT]];
    dir_output = [DIR_TOP filesep '6_events_crop_video' filesep 'CAV'];
    extract_6_events_save_video(file_event, file_video, dir_output,CAV_clip_names{i},'CAV');
end
fprintf('Finish Extracting CAV event \n');

for i = 1 : numel(GIV_clip_names)
    file_video = [DIR_TOP filesep 'videos_original' filesep [GIV_clip_names{i}, EXT_VIDEO]];
    file_event = [DIR_TOP filesep 'annotations' filesep [GIV_clip_names{i}, EXT_EVENT]];
    dir_output = [DIR_TOP filesep '6_events_crop_video' filesep 'GIV'];
    extract_6_events_save_video(file_event, file_video, dir_output,GIV_clip_names{i},'GIV');
end
fprintf('Finish Extracting GIV event \n');

for i = 1 : numel(GOV_clip_names)
    file_video = [DIR_TOP filesep 'videos_original' filesep [GOV_clip_names{i}, EXT_VIDEO]];
    file_event = [DIR_TOP filesep 'annotations' filesep [GOV_clip_names{i}, EXT_EVENT]];
    dir_output = [DIR_TOP filesep '6_events_crop_video' filesep 'GOV'];
    extract_6_events_save_video(file_event, file_video, dir_output,GOV_clip_names{i},'GOV');
end
fprintf('Finish Extracting GOV event \n');