% draw_viratdata1(file_object, file_event, file_video, dir_output)
% draw viratdata annotations (event only) on video and outputs rendered images
% tailored for Release 2.0
%
% NOTE: requires Matlab 2010b or newer version for VideoReader functionalities.

function extract_6_events_save_video(file_event, file_video, dir_output, video_name, event_type)

IDX_EID   = 1;
IDX_ETYPE = 2;
%IDX_DURATION = 3;
IDX_FRAME_S  = 4;
IDX_FRAME_E  = 5;
IDX_FRAME    = 6;
IDX_X = 7;
IDX_Y = 8;
IDX_W = 9;
IDX_H = 10;

STR_EVENT{1} = 'Loading';
STR_EVENT{2} = 'Unloading';
STR_EVENT{3} = 'Opening Trunk';
STR_EVENT{4} = 'Closing Trunk';
STR_EVENT{5} = 'Getting Into Vehicle';
STR_EVENT{6} = 'Getting Out of Vehicle';

%% Annotation
E = load(file_event);

if numel(E)==0
    disp('no events to draw')
    return
end

% Events
switch event_type
    case 'LAV'
        event_ind = find(E(:,2)==1);
        eids_EVENTs = unique(E(event_ind,1));
    case 'UAV'
        event_ind = find(E(:,2)==2);
        eids_EVENTs = unique(E(event_ind,1));
    case 'OAV'
        event_ind = find(E(:,2)==3);
        eids_EVENTs = unique(E(event_ind,1));
    case 'CAV'
        event_ind = find(E(:,2)==4);
        eids_EVENTs = unique(E(event_ind,1));
    case 'GIV'
        event_ind = find(E(:,2)==5);
        eids_EVENTs = unique(E(event_ind,1));
    case 'GOV'
        event_ind = find(E(:,2)==6);
        eids_EVENTs = unique(E(event_ind,1));
end


%% Output Related Processing

% converts from 1-base matlab framenumber to 0-base VIRAT framenumber
func_frameIdx_virat2videoreader = @(i) i+1;

disp('Start Reading Video')
tic;
% Access video file
vobj = VideoReader(file_video);

nFrames = vobj.NumberOfFrames;
framerate = vobj.FrameRate;
t = toc;

fprintf('Done using %f seconds \n',t);


% draw every event, around person only, with expanded bbox
for i=1:numel(eids_EVENTs)
    
    % time range
    ei = E( E(:,IDX_EID)==eids_EVENTs(i), :);
    t_start = ei(1,IDX_FRAME_S);
    t_end = ei(1,IDX_FRAME_E);
    if t_end >= nFrames
        t_end = nFrames-1;
    end
    
    i_etype = ei(1, IDX_ETYPE);
    
    row = find(ei(:,IDX_FRAME)== t_start);
    exmin = ei(row, IDX_X);
    %exmax = exmin + ei(row, IDX_W);
    width = ei(row, IDX_W);
    eymin = ei(row, IDX_Y);
    height = ei(row,IDX_H);
    
    time_step = 1;
    %         time_step = ceil(framerate/6);
    event_name = sprintf('event_id%02d_name_%s',eids_EVENTs(i),STR_EVENT{i_etype});
    save_name = [video_name '_' event_name];
    save_dir = [dir_output filesep save_name];
    if ~exist(save_dir,'dir')
        mkdir(save_dir)
    end
    New_name = [save_name '.avi'];
    Vid = VideoWriter([save_dir filesep New_name]);
    open(Vid);
    for v_=t_start:time_step:t_end
        v = func_frameIdx_virat2videoreader(v_);
        v_img = read(vobj,v);
        crop_image = imcrop(v_img,[exmin eymin width height]);
        writeVideo(Vid,crop_image);
    end % for v_
    close(Vid);
    fprintf('Finish Extracting the %s video %02d event %s \n',video_name, i_etype, STR_EVENT{i_etype});
end % for i

end % MAIN Function

