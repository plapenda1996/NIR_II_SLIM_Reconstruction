function [context,if_ok] = init_camera(fps,tint,build_bias,burst)
%% init camera lib
[notfound, warnings] = FliSdk.openLib();
context = FliSdk.init();
[listOfGrabbers] = FliSdk.detectGrabbers(context);
[listOfCameras] = FliSdk.detectCameras(context);
cameras = split(listOfCameras, ';');
if_ok = FliSdk.setCamera(context, cameras(1,1));
FliSdk.update(context);
if if_ok
    msgbox('Camera connected!')
else
    msgbox('Camera init failed!')
    return
end

% %% set temp
% [~, response] = FliSdk.sendCommandToCamera(context, 'set temperatures sensor -15');
% %% check temp
% [~, response] = FliSdk.sendCommandToCamera(context, 'temperatures frontend');
% disp(response)
% [~, response] = FliSdk.sendCommandToCamera(context, 'temperatures powerboard');
% disp(response)
% [~, response] = FliSdk.sendCommandToCamera(context, 'temperatures sensor');
% disp(response)

%% FPS & Tint setting
[ok] = FliSdk.setCameraFps(context, fps);
[ok, fps] = FliSdk.getCameraFps(context);
fprintf('Current FPS: %.4f\n',fps);

[ok] = FliSdk.setTint(context, tint);
[ok, tint] = FliSdk.getTint(context);
fprintf('Current Tint: %.4f ms\n',tint*1e3);

%% set unsigned reading
[~, response] = FliSdk.sendCommandToCamera(context, 'set unsigned off');
disp(response)
[~, response] = FliSdk.sendCommandToCamera(context, 'set hdr extended off');
disp(response)
[~, response] = FliSdk.sendCommandToCamera(context, 'set hdr off');
disp(response)
disp('unsigned reading OFF!')
[~, response] = FliSdk.sendCommandToCamera(context, 'set bias off');
[~, response] = FliSdk.sendCommandToCamera(context, 'set flat off');

%% Build bias
if build_bias
    % Build bias
    camera_build_bias(context)
end

%% set burst mode / ring buffer
if burst
    % burst ON
    % stop camera
    FliSdk.stop(context);
    FliSdk.resetBuffer(context);
    % turn on triggered acquisition mode
    [~, response] = FliSdk.sendCommandToCamera(context, 'set swsynchro on');
    disp(response)
    [~, response] = FliSdk.sendCommandToCamera(context, 'set nbframesperswtrig 1');
    disp(response)
    % start camera
    FliSdk.start(context);
    disp('Burst mode enabled.')
else
    % burst OFF and start ring buffer
    % stop camera
    FliSdk.stop(context);
    % turn off triggered acquisition mode
    [~, response] = FliSdk.sendCommandToCamera(context, 'set swsynchro off');
    disp(response)
    FliSdk.resetBuffer(context);
    % start camera
    FliSdk.start(context);
    % show buffer size
    disp('Burst mode off; ring buffer started!')
    fprintf('Buffer size: %d images\n', FliSdk.getbufferImagesCapacity(context))
end

% %% test getting images after a grabN
% N = 10;
% close all
% imgcube = captureN(context,N);
% 
% % show
% figure;
% sliceViewer(rescale(imgcube));
