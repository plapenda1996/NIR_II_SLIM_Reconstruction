function Image4DViewer(im4d, varargin)
    % Image4DViewer - Interactive viewer for 4D XYZT image data
    % 
    % Usage:
    %   Image4DViewer(im4d)
    %   Image4DViewer(im4d, 'VolumeRate', fps, 'ZResolution', z_um, 'XYResolution', xy_um, 'AmpAll', amp_all, ...
    %                         'DefaultSavePath', path, 'FilePrefix', prefix)
    %
    % Inputs:
    %   im4d - 4D matrix (Y, X, Z, T)
    %   Optional parameters:
    %     'VolumeRate'      - Frame rate (Hz)
    %     'ZResolution'     - Z step size (um)
    %     'XYResolution'    - XY pixel size (um)
    %     'AmpAll'          - Amplitude array (1xT or Tx1) with values 0-1 for each time point
    %     'DefaultSavePath' - Default directory for saving files (string)
    %     'FilePrefix'      - Default prefix for saved filenames (string)
    
    % --- [MODIFICATION 1] START: Add new optional parameters ---
    p = inputParser;
    addRequired(p, 'im4d', @(x) ndims(x) == 4);
    addParameter(p, 'VolumeRate', 30, @isnumeric);
    addParameter(p, 'ZResolution', 1, @isnumeric);
    addParameter(p, 'XYResolution', 1, @isnumeric);
    addParameter(p, 'AmpAll', [], @isnumeric);
    addParameter(p, 'DefaultSavePath', pwd, @ischar); % New: Default save path
    addParameter(p, 'FilePrefix', 'export', @ischar); % New: File prefix
    addParameter(p, 'ExportZList', [], @(x) isnumeric(x) || ischar(x) || isstring(x));
    parse(p, im4d, varargin{:});
    
    % Initialize data
    data = struct();
    data.original = im4d;
    data.normalized = im4d/max(im4d(:));  % Normalize to 0-1
    data.processed = data.normalized;
    [data.ny, data.nx, data.nz, data.nt] = size(data.normalized);
    data.volumeRate = p.Results.VolumeRate;
    data.zRes = p.Results.ZResolution;
    data.xyRes = p.Results.XYResolution;
    data.defaultSavePath = p.Results.DefaultSavePath; % New: Store save path
    data.filePrefix = p.Results.FilePrefix;         % New: Store file prefix
    % --- [MODIFICATION 1] END ---
    
    % Handle amplitude data
    if isempty(p.Results.AmpAll)
        data.ampAll = ones(data.nt, 1);
    else
        data.ampAll = p.Results.AmpAll(:);
        if length(data.ampAll) ~= data.nt
            error('AmpAll length (%d) must match time dimension (%d)', length(data.ampAll), data.nt);
        end
        if any(data.ampAll < 0) || any(data.ampAll > 1)
            warning('AmpAll values should be in range [0,1]. Clipping to valid range.');
            data.ampAll = max(0, min(1, data.ampAll));
        end
    end
    
    % Current state
    state = struct();
    state.currentZ = 1;
    state.currentT = 1;
    state.frameAvg = 1;
    state.lowPerc = 0.01;
    state.highPerc = 99.9;
    state.showScalebar = true;
    state.scalebarX = 85;
    state.scalebarY = 20;
    state.colormap = 'gray';
    state.scrollFrames = 5;
    state.lrFrames = 1;
    state.udFrames = 15;
    state.includeOverlay = true;
    state.exportFPS = 30; % Note: This is now controlled by fpsEdit, this is just an initial value.
    state.exportRange = [1 data.nt];
    state.applyAmplitude = ~isempty(p.Results.AmpAll);
    
    % Create main figure
    fig = figure('Name', 'Image4D Viewer', 'NumberTitle', 'off', ...
                 'Position', [100 100 1200 800], ...
                 'WindowScrollWheelFcn', @scrollWheel, ...
                 'WindowKeyPressFcn', @keyPress, ...
                 'CloseRequestFcn', @closeFig, ...
                 'Resize', 'on');
    
    % --- UI Elements (No changes here, code omitted for brevity) ---
    % ... (Your original UI code from line 81 to 318) ...
    % Left panel for image and sliders (75% width)
    leftPanel = uipanel('Parent', fig, 'BorderType', 'none', ...
                       'Position', [0 0 0.75 1]);
    
    % Right control panel (25% width)
    rightPanel = uipanel('Parent', fig, 'BorderType', 'none', ...
                        'Position', [0.75 0 0.25 1], ...
                        'BackgroundColor', [0.94 0.94 0.94]);
    
    % === LEFT PANEL COMPONENTS ===
    
    % Info text at top
    infoText = uicontrol('Parent', leftPanel, 'Style', 'text', ...
                         'Units', 'normalized', 'Position', [0.02 0.96 0.96 0.03], ...
                         'String', '', 'FontSize', 10, ...
                         'HorizontalAlignment', 'left', ...
                         'BackgroundColor', 'white');
    
    % Image axes
    ax = axes('Parent', leftPanel, 'Units', 'normalized', ...
             'Position', [0.05 0.15 0.9 0.8]);
    imgHandle = imshow(data.processed(:,:,1,1), [0 1], 'Parent', ax);
    colormap(ax, gray);
    
    % Z slider
    uicontrol('Parent', leftPanel, 'Style', 'text', 'String', 'Z Layer:', ...
             'Units', 'normalized', 'Position', [0.02 0.08 0.08 0.025], ...
             'HorizontalAlignment', 'right');
    zSlider = uicontrol('Parent', leftPanel, 'Style', 'slider', ...
                        'Units', 'normalized', 'Position', [0.11 0.08 0.78 0.025], ...
                        'Min', 1, 'Max', data.nz, 'Value', 1, ...
                        'SliderStep', [1/(data.nz-1) 5/(data.nz-1)]);
    zText = uicontrol('Parent', leftPanel, 'Style', 'text', ...
                     'Units', 'normalized', 'Position', [0.9 0.08 0.08 0.025], ...
                     'String', sprintf('1/%d', data.nz));
    
    % T slider
    uicontrol('Parent', leftPanel, 'Style', 'text', 'String', 'Time:', ...
             'Units', 'normalized', 'Position', [0.02 0.04 0.08 0.025], ...
             'HorizontalAlignment', 'right');
    tSlider = uicontrol('Parent', leftPanel, 'Style', 'slider', ...
                        'Units', 'normalized', 'Position', [0.11 0.04 0.78 0.025], ...
                        'Min', 1, 'Max', data.nt, 'Value', 1, ...
                        'SliderStep', [1/(data.nt-1) 10/(data.nt-1)]);
    tText = uicontrol('Parent', leftPanel, 'Style', 'text', ...
                     'Units', 'normalized', 'Position', [0.9 0.04 0.08 0.025], ...
                     'String', sprintf('1/%d', data.nt));
    
    % Instructions
    uicontrol('Parent', leftPanel, 'Style', 'text', ...
             'Units', 'normalized', 'Position', [0.02 0.005 0.96 0.025], ...
             'String', 'Controls: Mouse wheel/←→/↑↓: time | Q/W (±1), A/S (±2), Z/X (±5): Z layers', ...
             'FontSize', 8, 'HorizontalAlignment', 'center');
    
    % === RIGHT PANEL CONTROLS ===
    yPos = 0.95;  % Starting Y position
    yStep = 0.04;  % Step size for each control
    
    % Title
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', '=== Controls ===', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.03], ...
             'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;
    
    % Frame averaging
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Frame Averaging:', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.6 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    avgEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', 'String', '1', ...
                       'Units', 'normalized', 'Position', [0.65 yPos 0.3 0.025], ...
                       'Callback', @updateFrameAvg);
    yPos = yPos - yStep;
    
    % Amplitude control (only show if amplitude data was provided)
    if state.applyAmplitude
        ampCheck = uicontrol('Parent', rightPanel, 'Style', 'checkbox', ...
                            'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
                            'String', 'Apply Amplitude Scaling', 'Value', 1, ...
                            'Callback', @updateDisplay, 'BackgroundColor', [0.94 0.94 0.94]);
        yPos = yPos - yStep;
    else
        ampCheck = [];
    end
    
    % Adjustment controls
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', '--- Adjust Percentiles ---', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
             'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;
    
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Low %:', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.25 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    lowPercEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                           'Units', 'normalized', 'Position', [0.3 yPos 0.2 0.025], ...
                           'String', '0.01', 'Callback', @updateAdjust);
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'High %:', ...
             'Units', 'normalized', 'Position', [0.52 yPos 0.25 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    highPercEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                            'Units', 'normalized', 'Position', [0.77 yPos 0.2 0.025], ...
                            'String', '99.9', 'Callback', @updateAdjust);
    yPos = yPos - yStep;
    
    % Quick adjust buttons
    uicontrol('Parent', rightPanel, 'Style', 'pushbutton', ...
             'Units', 'normalized', 'Position', [0.03 yPos 0.2 0.025], ...
             'String', '0.1%', 'Callback', @(~,~) quickAdjust(0.1));
    uicontrol('Parent', rightPanel, 'Style', 'pushbutton', ...
             'Units', 'normalized', 'Position', [0.25 yPos 0.2 0.025], ...
             'String', '0.01%', 'Callback', @(~,~) quickAdjust(0.01));
    uicontrol('Parent', rightPanel, 'Style', 'pushbutton', ...
             'Units', 'normalized', 'Position', [0.47 yPos 0.2 0.025], ...
             'String', '0.001%', 'Callback', @(~,~) quickAdjust(0.001));
    uicontrol('Parent', rightPanel, 'Style', 'pushbutton', ...
             'Units', 'normalized', 'Position', [0.69 yPos 0.2 0.025], ...
             'String', '0%', 'Callback', @(~,~) quickAdjust(0));
    yPos = yPos - yStep * 1.2;
    
    % Navigation controls
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', '--- Frame Navigation ---', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
             'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;
    
    % Navigation settings in one row
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Scroll:', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.2 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    scrollEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                          'Units', 'normalized', 'Position', [0.25 yPos 0.12 0.025], ...
                          'String', '5', 'Callback', @updateNavSettings);
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', '←→:', ...
             'Units', 'normalized', 'Position', [0.38 yPos 0.12 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    lrEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                      'Units', 'normalized', 'Position', [0.5 yPos 0.12 0.025], ...
                      'String', '1', 'Callback', @updateNavSettings);
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', '↑↓:', ...
             'Units', 'normalized', 'Position', [0.63 yPos 0.12 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    udEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                      'Units', 'normalized', 'Position', [0.75 yPos 0.12 0.025], ...
                      'String', '15', 'Callback', @updateNavSettings);
    yPos = yPos - yStep * 1.2;
    
    % Scalebar controls
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', '--- Display Options ---', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
             'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;
    
    scaleCheck = uicontrol('Parent', rightPanel, 'Style', 'checkbox', ...
                          'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
                          'String', 'Show Scalebar', 'Value', 1, ...
                          'Callback', @updateDisplay, 'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;
    
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Position X%:', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.35 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    scaleXEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                          'Units', 'normalized', 'Position', [0.4 yPos 0.15 0.025], ...
                          'String', '85', 'Callback', @updateDisplay);
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Y%:', ...
             'Units', 'normalized', 'Position', [0.57 yPos 0.15 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    scaleYEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                          'Units', 'normalized', 'Position', [0.72 yPos 0.15 0.025], ...
                          'String', '15', 'Callback', @updateDisplay);
    yPos = yPos - yStep;
    
    % Colormap selection
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Colormap:', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.35 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    cmapMenu = uicontrol('Parent', rightPanel, 'Style', 'popupmenu', ...
                        'Units', 'normalized', 'Position', [0.4 yPos 0.55 0.025], ...
                        'String', {'gray', 'hot', 'bone'}, 'Value', 1, ...
                        'Callback', @updateColormap);
    yPos = yPos - yStep * 1.2;
    
    % Export controls
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', '--- Export Options ---', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
             'FontWeight', 'bold', 'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;
    
    % 计算默认填充值
    if isnumeric(p.Results.ExportZList) && ~isempty(p.Results.ExportZList)
        zListDefaultStr = strjoin(arrayfun(@num2str, p.Results.ExportZList, 'UniformOutput', false), ',');
    elseif ischar(p.Results.ExportZList) || isstring(p.Results.ExportZList)
        zListDefaultStr = char(p.Results.ExportZList);
    else
        zListDefaultStr = num2str(state.currentZ);  % 默认当前 Z
    end
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Z Index List:', ...
        'Units', 'normalized', 'Position', [0.05 yPos 0.4 0.025], ...
        'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    zListEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
        'Units', 'normalized', 'Position', [0.46 yPos 0.34 0.025], ...
        'String', zListDefaultStr);
    uicontrol('Parent', rightPanel, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [0.82 yPos 0.13 0.025], ...
        'String', 'All', ...
        'Callback', @(~,~) set(zListEdit, 'String', sprintf('1-%d', data.nz)));
    yPos = yPos - yStep;

    uicontrol('Parent', rightPanel, 'Style', 'text', ...
          'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
          'String', 'Eg. 1,2,4 | 3-8 | 1:2:9 | all | current', ...
          'FontSize', 8, 'HorizontalAlignment', 'left', ...
          'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;

    overlayCheck = uicontrol('Parent', rightPanel, 'Style', 'checkbox', ...
                            'Units', 'normalized', 'Position', [0.05 yPos 0.9 0.025], ...
                            'String', 'Include Overlay (MP4 only)', 'Value', 1, ...
                            'BackgroundColor', [0.94 0.94 0.94]);
    yPos = yPos - yStep;
    
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'MP4 FPS:', ... % Label clarifies this is for MP4
             'Units', 'normalized', 'Position', [0.05 yPos 0.2 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    fpsEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                       'Units', 'normalized', 'Position', [0.25 yPos 0.2 0.025], ...
                       'String', '30'); % This is the requested input box
    uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Range:', ...
             'Units', 'normalized', 'Position', [0.5 yPos 0.25 0.025], ...
             'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94]);
    rangeEdit = uicontrol('Parent', rightPanel, 'Style', 'edit', ...
                         'Units', 'normalized', 'Position', [0.75 yPos 0.2 0.025], ...
                         'String', sprintf('1-%d', data.nt));
    yPos = yPos - yStep;
    
    uicontrol('Parent', rightPanel, 'Style', 'pushbutton', ...
             'Units', 'normalized', 'Position', [0.05 yPos 0.42 0.03], ...
             'String', 'Export MP4', 'Callback', @exportMP4);
    uicontrol('Parent', rightPanel, 'Style', 'pushbutton', ...
             'Units', 'normalized', 'Position', [0.52 yPos 0.42 0.03], ...
             'String', 'Export TIF', 'Callback', @exportTIF);
    
    % Store handles
    handles = guihandles(fig);
    guidata(fig, handles);
    
    % Add listeners and initial update
    addlistener(zSlider, 'Value', 'PostSet', @(~,~) sliderUpdate('z'));
    addlistener(tSlider, 'Value', 'PostSet', @(~,~) sliderUpdate('t'));
    updateDisplay();

    % --- Nested Functions (updateDisplay, keyPress etc. are unchanged) ---
    % ... (Your original nested functions from line 324 to 516, they don't need changes) ...
    function sliderUpdate(type)
        if strcmp(type, 'z')
            state.currentZ = round(get(zSlider, 'Value'));
            set(zText, 'String', sprintf('%d/%d', state.currentZ, data.nz));
        else
            state.currentT = round(get(tSlider, 'Value'));
            set(tText, 'String', sprintf('%d/%d', state.currentT, data.nt));
        end
        updateDisplay();
    end
    
    function updateDisplay(~, ~)
        % Get current frame
        currentFrame = data.processed(:,:,state.currentZ,state.currentT);
        
        % Apply adjustment
        lowVal = prctile(currentFrame(:), state.lowPerc);
        highVal = prctile(currentFrame(:), state.highPerc);
        if lowVal < highVal
            currentFrame = imadjust(currentFrame, [lowVal highVal], [0 1]);
        end
        
        % Apply amplitude scaling if checkbox is checked and amplitude data exists
        if state.applyAmplitude && ~isempty(ampCheck) && get(ampCheck, 'Value')
            currentFrame = currentFrame * data.ampAll(state.currentT);
        end
        
        % Update image
        set(imgHandle, 'CData', currentFrame);
        
        % Update scalebar if needed
        if get(scaleCheck, 'Value')
            hold(ax, 'on');
            % Clear previous scalebar
            delete(findobj(ax, 'Tag', 'scalebar'));
            delete(findobj(ax, 'Tag', 'scalelabel'));
            
            % Calculate scalebar
            scaleLength = 200; % um
            scalePixels = scaleLength / data.xyRes;
            xPos = data.nx * str2double(get(scaleXEdit, 'String'))/100;
            yPos = data.ny * str2double(get(scaleYEdit, 'String'))/100;
            
            % Draw scalebar
            plot(ax, [xPos-scalePixels xPos], [yPos yPos], 'r-', ...
                'LineWidth', 3, 'Tag', 'scalebar');
            text(ax, xPos-scalePixels/2, yPos-10, sprintf('%d μm', scaleLength), ...
                'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center', ...
                'Tag', 'scalelabel');
            hold(ax, 'off');
        else
            delete(findobj(ax, 'Tag', 'scalebar'));
            delete(findobj(ax, 'Tag', 'scalelabel'));
        end
        
        % Update info text
        timeStr = sprintf('Time: %.2f/%.2f s', ...
                         (state.currentT-1)/data.volumeRate, ...
                         (data.nt-1)/data.volumeRate);
        frameStr = sprintf('Frame: %d/%d', state.currentT, data.nt);
        layerStr = sprintf('Z: %d/%d (%.1f μm)', ...
                          state.currentZ, data.nz, (state.currentZ-1)*data.zRes);
        ampStr = '';
        if state.applyAmplitude
            ampStr = sprintf(' | Amp: %.3f', data.ampAll(state.currentT));
        end
        set(infoText, 'String', sprintf('%s | %s | %s | Adjust: [%.3f%%, %.3f%%]%s', ...
                                       timeStr, frameStr, layerStr, state.lowPerc, state.highPerc, ampStr));
    end
    
    function updateFrameAvg(~, ~)
        newAvg = str2double(get(avgEdit, 'String'));
        if ~isnan(newAvg) && newAvg >= 1
            state.frameAvg = round(newAvg);
            % Apply frame averaging
            if state.frameAvg > 1
                % Create averaging kernel
                kernel = ones(1, 1, 1, min(state.frameAvg, data.nt));
                kernel = kernel / sum(kernel(:));
                data.processed = convn(data.normalized, kernel, 'same');
            else
                data.processed = data.normalized;
            end
            updateDisplay();
        end
    end
    
    function updateAdjust(~, ~)
        state.lowPerc = str2double(get(lowPercEdit, 'String'));
        state.highPerc = str2double(get(highPercEdit, 'String'));
        updateDisplay();
    end
    
    function quickAdjust(perc)
        state.lowPerc = perc;
        state.highPerc = 100 - perc;
        set(lowPercEdit, 'String', num2str(perc));
        set(highPercEdit, 'String', num2str(100 - perc));
        updateDisplay();
    end
    
    function zList = getZListForExport(~,~)
        % 读取并解析 Z 列表输入框
        raw = strtrim(get(zListEdit, 'String'));
        try
            zList = parseZListStr(raw, data.nz, state.currentZ);
        catch ME
            errordlg(['Z list error: ' ME.message], 'Error');
            zList = [];
        end
    end

    function zList = parseZListStr(raw, nz, currentZ)
        % 支持: "1,2,4" | "3-8" | "1:2:9" | "all" | "current"
        if isempty(raw)
            zList = currentZ; return;
        end
        rawL = lower(strtrim(raw));
        if any(strcmp(rawL, {'all','*'}))
            zList = 1:nz; return;
        elseif any(strcmp(rawL, {'cur','current','this'}))
            zList = currentZ; return;
        end
        tokens = regexp(raw, '[,; ]+', 'split');
        zList = [];
        for i = 1:numel(tokens)
            tok = strtrim(tokens{i});
            if isempty(tok), continue; end
            if contains(tok, ':')
                % 允许 "start:step:end" 或 "start:end"
                seq = str2num(tok); %#ok<ST2NM> 使用 MATLAB 语法解析
                if isempty(seq), error('Invalid colon expression: "%s"', tok); end
                zList = [zList, seq]; %#ok<AGROW>
            elseif contains(tok, '-')
                pr = strsplit(tok, '-');
                if numel(pr) ~= 2, error('Invalid range token: "%s"', tok); end
                a = str2double(pr{1}); b = str2double(pr{2});
                if any(isnan([a b])), error('Invalid range token: "%s"', tok); end
                if a <= b, zList = [zList, a:b]; else, zList = [zList, a:-1:b]; end %#ok<AGROW>
            else
                val = str2double(tok);
                if isnan(val), error('Invalid index token: "%s"', tok); end
                zList = [zList, val]; %#ok<AGROW>
            end
        end
        zList = unique(round(zList));
        if any(zList < 1 | zList > nz)
            error('Z indices must be within 1..%d.', nz);
        end
    end

    function updateColormap(~, ~)
        maps = {'gray', 'hot', 'bone'};
        state.colormap = maps{get(cmapMenu, 'Value')};
        colormap(ax, state.colormap);
    end
    
    function updateNavSettings(~, ~)
        state.scrollFrames = str2double(get(scrollEdit, 'String'));
        state.lrFrames = str2double(get(lrEdit, 'String'));
        state.udFrames = str2double(get(udEdit, 'String'));
    end
    
    function scrollWheel(~, evt)
        delta = -evt.VerticalScrollCount * state.scrollFrames;
        newT = max(1, min(data.nt, state.currentT + delta));
        state.currentT = newT;
        set(tSlider, 'Value', newT);
        updateDisplay();
    end
    
    function keyPress(~, evt)
        switch evt.Key
            case 'leftarrow'
                state.currentT = max(1, state.currentT - state.lrFrames);
                set(tSlider, 'Value', state.currentT);
            case 'rightarrow'
                state.currentT = min(data.nt, state.currentT + state.lrFrames);
                set(tSlider, 'Value', state.currentT);
            case 'uparrow'
                state.currentT = max(1, state.currentT - state.udFrames);
                set(tSlider, 'Value', state.currentT);
            case 'downarrow'
                state.currentT = min(data.nt, state.currentT + state.udFrames);
                set(tSlider, 'Value', state.currentT);
            case 'q'
                state.currentZ = max(1, state.currentZ - 1);
                set(zSlider, 'Value', state.currentZ);
            case 'w'
                state.currentZ = min(data.nz, state.currentZ + 1);
                set(zSlider, 'Value', state.currentZ);
            case 'a'
                state.currentZ = max(1, state.currentZ - 2);
                set(zSlider, 'Value', state.currentZ);
            case 's'
                state.currentZ = min(data.nz, state.currentZ + 2);
                set(zSlider, 'Value', state.currentZ);
            case 'z'
                state.currentZ = max(1, state.currentZ - 5);
                set(zSlider, 'Value', state.currentZ);
            case 'x'
                state.currentZ = min(data.nz, state.currentZ + 5);
                set(zSlider, 'Value', state.currentZ);
        end
        updateDisplay();
    end
    
    function exportMP4(~, ~)
        % --- 解析 GUI 输入 (与原版相同) ---
        rangeStr = get(rangeEdit, 'String');
        rangeParts = strsplit(rangeStr, '-');
        if length(rangeParts) ~= 2, errordlg('Invalid range format. Use: start-end', 'Error'); return; end
        startFrame = str2double(rangeParts{1});
        endFrame   = str2double(rangeParts{2});
        if isnan(startFrame) || isnan(endFrame) || startFrame < 1 || endFrame > data.nt || startFrame > endFrame
            errordlg(sprintf('Invalid range. Must be between 1 and %d', data.nt), 'Error'); return;
        end
        zList = getZListForExport();
        if isempty(zList), return; end
        multiZ = numel(zList) > 1;
        fpsVal = str2double(get(fpsEdit, 'String'));
        if isnan(fpsVal) || fpsVal <= 0, fpsVal = 30; end
        includeOverlay = get(overlayCheck, 'Value');
        applyAmp = state.applyAmplitude && ~isempty(ampCheck) && get(ampCheck, 'Value');
        showScalebarInVideo = get(scaleCheck, 'Value');

        % --- 预计算和设置 (优化点) ---
        exportScale  = 2;
        exportWidth  = data.nx * exportScale;
        exportHeight = data.ny * exportScale;

        % 尺度尺设置
        scaleLength = 200; % um
        scalePixels = scaleLength / data.xyRes;
        scalebarX = str2double(get(scaleXEdit, 'String'));
        scalebarY = str2double(get(scaleYEdit, 'String'));

        % 【优化】在循环外创建颜色图，避免使用 eval
        try
            cmap = feval(state.colormap, 256);
        catch
            warning('Invalid colormap name. Defaulting to gray.');
            cmap = gray(256);
        end

        % 【优化】预定义文本样式，方便调用
        mainFontSize  = 8 * exportScale;
        smallFontSize = 7 * exportScale;
        scalebarFontSize = 8.5 * exportScale;

        % --- 文件路径选择 (与原版相同) ---
        targetDir = '';
        singlePath = '';
        if ~multiZ
            defaultFilename = fullfile(data.defaultSavePath, sprintf('%s_Z%03d.mp4', data.filePrefix, zList(1)));
            [file, path] = uiputfile('*.mp4', 'Save video as', defaultFilename);
            if isequal(file, 0), return; end
            singlePath = fullfile(path, file);
            targetDir = path;
        else
            tgt = uigetdir(data.defaultSavePath, 'Choose output folder');
            if isequal(tgt, 0), return; end
            targetDir = tgt;
        end

        % --- 进度条 (与原版相同) ---
        totalSteps = numel(zList) * (endFrame - startFrame + 1);
        curStep = 0;
        wb = waitbar(0, 'Exporting MP4...', 'Name', 'Export Progress');
        tic;

        try
            for zi = 1:numel(zList)
                z = zList(zi);
                if multiZ
                    vidPath = fullfile(targetDir, sprintf('%s_Z%03d.mp4', data.filePrefix, z));
                else
                    vidPath = singlePath;
                end

                v = VideoWriter(vidPath, 'MPEG-4');
                v.FrameRate = fpsVal;
                open(v);

                % 【优化】预计算 z 相关的文本
                zStr = sprintf('Z: %.1f μm', (z-1)*data.zRes);

                for t = startFrame:endFrame
                    % ================================================================
                    % 【核心优化区域开始】
                    % ================================================================

                    % 步骤 1: 选帧和基础处理
                    if includeOverlay
                        frame = data.processed(:,:,z,t);
                        lowVal = prctile(frame(:), state.lowPerc);
                        highVal = prctile(frame(:), state.highPerc);
                        if lowVal < highVal
                            frame = imadjust(frame, [lowVal highVal], []); % 使用 [] 自动映射到 [0 1]
                        end
                    else
                        frame = data.normalized(:,:,z,t);
                    end

                    % 步骤 2: 幅度调整和缩放
                    if applyAmp
                        frame = frame * data.ampAll(t);
                    end
                    frame = imresize(frame, exportScale, 'bicubic');

                    % 步骤 3: 转换成 RGB
                    frameRGB = ind2rgb(gray2ind(frame, 256), cmap);

                    % 步骤 4: 【核心】使用 insertText/insertShape 添加叠加层
                    if includeOverlay
                        % 标尺
                        if showScalebarInVideo
                            xPos = exportWidth  * scalebarX/100;
                            yPos = exportHeight * scalebarY/100;
                            scaledPixels = scalePixels * exportScale;
                            linePos = [xPos-scaledPixels, yPos; xPos, yPos];
                            textPos = [xPos-scaledPixels/2, yPos-15*exportScale];

                            frameRGB = insertShape(frameRGB, 'Line', linePos, 'LineWidth', 2*exportScale, 'Color', 'white');
                            frameRGB = insertText(frameRGB, textPos, sprintf('%d μm', scaleLength), ...
                                'TextColor', 'white', 'FontSize', scalebarFontSize, ...
                                'Font','Arial Bold',...
                                'BoxColor', 'black', 'BoxOpacity', 0.2, 'AnchorPoint', 'CenterTop');
                        end

                        % 文本信息
                        timeStr  = sprintf('Time: %.2f s', (t-1)/data.volumeRate);
                        frameStr = sprintf('Frame: %d/%d', t, data.nt);
                        fpsStr   = sprintf('FPS (actual/play): %g, %g', v.FrameRate, data.volumeRate);

                        textX = 7 * exportScale;
                        textY = 10 * exportScale;
                        lineSpacing = 8 * exportScale;

                        positions = [textX, textY;
                            textX, textY+lineSpacing;
                            textX, textY+lineSpacing*2;
                            textX, textY+lineSpacing*3];
                        texts = {timeStr; frameStr; fpsStr; zStr};

                        if applyAmp
                            ampStr = sprintf('Amp: %.3f', data.ampAll(t));
                            positions = [positions; textX, textY+lineSpacing*4];
                            texts = [texts; {ampStr}];
                        end

                        frameRGB = insertText(frameRGB, positions, texts, 'TextColor', 'white', ...
                            'Font','Arial Bold','BoxColor', 'black', 'BoxOpacity', 0,...
                            'FontSize', mainFontSize);

                        % 画面内进度条
                        barHeight = 4 * exportScale;
                        barY = exportHeight - 15*exportScale;
                        barWidth = exportWidth * 0.25;
                        barX = 10 * exportScale;
                        progressRatio = (t - startFrame) / (endFrame - startFrame);

                        % 前景条
                        if progressRatio > 0
                            frameRGB = insertShape(frameRGB, 'FilledRectangle', [barX, barY, barWidth*progressRatio, barHeight], ...
                                'Color', 'green', 'Opacity', 0.8);
                        end
                        % 进度文本
                        frameRGB = insertText(frameRGB, [barX + barWidth/2, barY - 5*exportScale], ...
                            sprintf('%.1f%%', progressRatio*100), 'TextColor', 'white', 'FontSize', smallFontSize, ...
                            'Font','Arial Bold','BoxColor', 'black', 'BoxOpacity', 0,'AnchorPoint', 'CenterTop');
                    end

                    % ================================================================
                    % 【核心优化区域结束】
                    % ================================================================

                    writeVideo(v, frameRGB);

                    % 更新总进度 (与原版相同)
                    curStep = curStep + 1;
                    progress = curStep / totalSteps;
                    elapsed  = toc;
                    remaining = elapsed * (1/progress - 1);
                    if ishandle(wb)
                        waitbar(progress, wb, sprintf('Gen MP4... Z%d (%d/%d), Frame %d/%d | %.0f%% (%.1f s left)', ...
                            z, zi, numel(zList), t, endFrame, progress*100, remaining));
                    else
                        close(v);
                        if multiZ && exist(vidPath, 'file'), delete(vidPath); end
                        return;
                    end
                end
                close(v);
            end
            if ishandle(wb), close(wb); end

        catch ME
            if exist('v','var') && isvalid(v), close(v); end
            if ishandle(wb), close(wb); end
            errordlg(['Export failed: ' ME.message], 'Error');
        end
    end
    
    function exportTIF(~, ~)
        % 解析时间范围
        rangeStr = get(rangeEdit, 'String');
        rangeParts = strsplit(rangeStr, '-');
        if length(rangeParts) ~= 2
            errordlg('Invalid range format. Use: start-end', 'Error');
            return;
        end
        startFrame = str2double(rangeParts{1});
        endFrame   = str2double(rangeParts{2});
        if isnan(startFrame) || isnan(endFrame) || startFrame < 1 || endFrame > data.nt || startFrame > endFrame
            errordlg(sprintf('Invalid range. Must be between 1 and %d', data.nt), 'Error');
            return;
        end

        % 解析 Z 列表
        zList = getZListForExport();
        if isempty(zList), return; end
        multiZ = numel(zList) > 1;

        % 单 Z：uiputfile；多 Z：选择文件夹并自动命名
        targetDir = '';
        singlePath = '';
        if ~multiZ
            defaultFilename = fullfile(data.defaultSavePath, sprintf('%s_Z%03d.tif', data.filePrefix, zList(1)));
            [file, path] = uiputfile({'*.tif;*.tiff', 'TIF Files (*.tif, *.tiff)'}, 'Save TIF stack as', defaultFilename);
            if isequal(file, 0), return; end
            singlePath = fullfile(path, file);
            targetDir = path;
        else
            tgt = uigetdir(data.defaultSavePath, 'Choose output folder');
            if isequal(tgt, 0), return; end
            targetDir = tgt;
        end

        % 进度条（Z×T）
        totalSteps = numel(zList) * (endFrame - startFrame + 1);
        curStep = 0;
        wb = waitbar(0, 'Exporting TIF stack...', 'Name', 'Export Progress');
        tic;

        applyAmp = state.applyAmplitude && ~isempty(ampCheck) && get(ampCheck, 'Value');

        try
            for zi = 1:numel(zList)
                z = zList(zi);

                % 当前 Z 的输出路径
                if multiZ
                    fullFilePath = fullfile(targetDir, sprintf('%s_Z%03d.tif', data.filePrefix, z));
                else
                    fullFilePath = singlePath;
                end

                % 如果已存在旧文件，覆盖前先删除（避免追加到旧内容）
                if exist(fullFilePath, 'file'), delete(fullFilePath); end

                for t = startFrame:endFrame
                    % 注意：TIF 输出保存的是处理后的像素，不包含图形叠加
                    frame = data.processed(:,:,z, t);

                    lowVal = prctile(frame(:), state.lowPerc);
                    highVal = prctile(frame(:), state.highPerc);
                    if lowVal < highVal
                        frame = imadjust(frame, [lowVal highVal], [0 1]);
                    end
                    if applyAmp
                        frame = frame * data.ampAll(t);
                    end

                    frame_to_save = im2uint16(frame);
                    if t == startFrame
                        imwrite(frame_to_save, fullFilePath, 'tif', 'WriteMode', 'overwrite', 'Compression', 'none');
                    else
                        imwrite(frame_to_save, fullFilePath, 'tif', 'WriteMode', 'append', 'Compression', 'none');
                    end

                    % 更新进度
                    curStep = curStep + 1;
                    progress = curStep / totalSteps;
                    elapsed  = toc;
                    remaining = elapsed * (1/progress - 1);
                    if ishandle(wb)
                        waitbar(progress, wb, sprintf('Exporting TIF... Z %d (%d/%d), Frame %d/%d | %.0f%% (%.1f s left)', ...
                            z, zi, numel(zList), t, endFrame, progress*100, remaining));
                    else
                        % 取消：删除当前未完成文件
                        if exist(fullFilePath, 'file'), delete(fullFilePath); end
                        return;
                    end
                end
            end

            if ishandle(wb), close(wb); end
            % msgbox('TIF export complete!', 'Success');
        catch ME
            if ishandle(wb), close(wb); end
            errordlg(['TIF export failed: ' ME.message], 'Error');
        end
    end
    
    function closeFig(~, ~)
        % This function will be called when the figure is closed
        delete(fig);
    end
end