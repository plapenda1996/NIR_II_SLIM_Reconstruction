function [tform_bkwd_ref,tform_bkwd_rel] = coordinate_transform(tform_bkwd_list, center_frame_idx)
% input: tform_bkwd_list (LF space -> 3D obj space)
% s.t.
% tform_bkwd_list(x) = tform_bkwd_rel(tform_bkwd_ref(x))
% i.e. LF space -> 3D ref obj space (depth invariant) -> 3D obj space
[nViews, nDepths] = size(tform_bkwd_list);
tform_bkwd_rel = repmat(affinetform2d(),nViews,nDepths);
tform_bkwd_ref = repmat(affinetform2d(),nViews,1);
for i = 1:nViews
    % Reference transformation for each view
    if nargin<2
        % option 1: avg.
        % fwd: 3D obj space (depth invariant) -> LF space
        A_ref_fwd = zeros(3,3);
        for j = 1:nDepths
            tform_tmp = invert(tform_bkwd_list(i,j));
            A_ref_fwd = A_ref_fwd + tform_tmp.A/nDepths;
        end
        % bkwd_ref: LF space -> 3D obj space (depth invariant)
        tform_bkwd_ref(i) = invert(affinetform2d(A_ref_fwd));
    else
        % % option 2: center
        tform_tmp = invert(tform_bkwd_list(i,center_frame_idx));
        A_ref_fwd = tform_tmp.A;
        % bkwd_ref: LF space -> 3D obj space (depth invariant)
        tform_bkwd_ref(i) = tform_bkwd_list(i,center_frame_idx);
    end
    
    % fit on reference tform
    % tform_bkwd_rel: 3D obj space (depth invariant) -> 3D obj space
    % equal to: 1. 3D obj space (depth invariant) -> LF space
    % 2. LF space -> 3D obj space (depth variant)
    for j = 1:nDepths
        A = tform_bkwd_list(i,j).A * A_ref_fwd;
        tform_bkwd_rel(i,j) = affinetform2d(A);
    end
end

% Extract all transformation parameters for all views and depths
tx_orig = zeros(nViews, nDepths);
ty_orig = zeros(nViews, nDepths);
a11_orig = zeros(nViews, nDepths);
a12_orig = zeros(nViews, nDepths);
a21_orig = zeros(nViews, nDepths);
a22_orig = zeros(nViews, nDepths);

for i = 1:nViews
    for j = 1:nDepths
        A = tform_bkwd_rel(i,j).A;
        tx_orig(i,j) = A(1,3);
        ty_orig(i,j) = A(2,3);
        a11_orig(i,j) = A(1,1);
        a12_orig(i,j) = A(1,2);
        a21_orig(i,j) = A(2,1);
        a22_orig(i,j) = A(2,2);
    end
end

% Store the original tx and ty for reference
tx_original = tx_orig;
ty_original = ty_orig;

% Create the figure for displaying fitting results
fig = figure('Name', 'Translation Parameter Fitting', 'Position', [50, 50, 1200, 800]);

% Calculate subplot grid dimensions
nRows = ceil(sqrt(nViews));
nCols = ceil(nViews / nRows);

% Define outlier detection options
outlier_options = {
    'No outlier removal', 
    'IQR method (1.5×IQR)',
    'Remove first point only',
    'Remove last point only',
    'Remove first & last points',
    'Remove first & last 10%',
    'Remove first & last 20%',
    'Mean ± 2×STD'
};

% Store the current polynomial order and outlier method for each view
view_orders = ones(1, nViews);
outlier_methods = ones(1, nViews);  % Default: No outlier removal

% Create dropdowns for each view
order_dropdowns = cell(1, nViews);
outlier_dropdowns = cell(1, nViews);

% Store subplot handles
subplots = cell(1, nViews);

for v = 1:nViews
    % Position the controls near the subplot
    row = ceil(v/nCols);
    col = mod(v-1, nCols) + 1;
    x_pos = (col-1)*(1200/nCols) + 20;
    y_pos = 800 - row*(800/nRows) + 30;
    
    % Order dropdown
    order_dropdowns{v} = uicontrol('Style', 'popupmenu', ...
        'String', {'Order 1', 'Order 2', 'Order 3'}, ...
        'Position', [x_pos, y_pos, 80, 25], ...
        'Value', 1, ...  % Default to 1st order
        'Tag', num2str(v));
    
    % Outlier method dropdown
    outlier_dropdowns{v} = uicontrol('Style', 'popupmenu', ...
        'String', outlier_options, ...
        'Position', [x_pos + 90, y_pos, 140, 25], ...
        'Value', 1, ...  % Default to no outlier removal
        'Tag', num2str(v));
    
    % Set callback functions
    set(order_dropdowns{v}, 'Callback', @(src,event) order_callback(src, event, v));
    set(outlier_dropdowns{v}, 'Callback', @(src,event) outlier_callback(src, event, v));
    
    % Create subplot
    subplots{v} = subplot(nRows, nCols, v);
end

% Initial plot with all views at default settings
update_all_plots();

% Set close request function
set(fig, 'CloseRequestFcn', @close_figure);

% Wait for user to close the figure
uiwait(fig);

    % Nested function: Update plots for all views
    function update_all_plots()
        for v = 1:nViews
            update_plot(v);
        end
    end

    % Nested function: Update a specific view's plot
    function update_plot(view_idx)
        % Get current order and outlier method for this view
        order = view_orders(view_idx);
        outlier_method = outlier_methods(view_idx);
        
        % Get data for this view
        tx_data = tx_original(view_idx,:);
        ty_data = ty_original(view_idx,:);
        
        % Initialize outlier masks (false = not an outlier)
        tx_outliers = false(1, nDepths);
        ty_outliers = false(1, nDepths);
        
        % Depth indices for x-axis
        depth_indices = 1:nDepths;
        
        % Apply selected outlier detection method
        switch outlier_method
            case 1  % No outlier removal
                % Do nothing - all points are used
                
            case 2  % IQR method
                % For tx
                q1_tx = prctile(tx_data, 25);
                q3_tx = prctile(tx_data, 75);
                iqr_tx = q3_tx - q1_tx;
                threshold_tx = 1.5 * iqr_tx;
                tx_outliers = (tx_data < q1_tx - threshold_tx) | (tx_data > q3_tx + threshold_tx);
                
                % For ty
                q1_ty = prctile(ty_data, 25);
                q3_ty = prctile(ty_data, 75);
                iqr_ty = q3_ty - q1_ty;
                threshold_ty = 1.5 * iqr_ty;
                ty_outliers = (ty_data < q1_ty - threshold_ty) | (ty_data > q3_ty + threshold_ty);
                
            case 3  % Remove first point only
                if nDepths > 1
                    tx_outliers(1) = true;
                    ty_outliers(1) = true;
                end
                
            case 4  % Remove last point only
                if nDepths > 1
                    tx_outliers(nDepths) = true;
                    ty_outliers(nDepths) = true;
                end
                
            case 5  % Remove first & last points
                if nDepths > 2
                    tx_outliers([1, nDepths]) = true;
                    ty_outliers([1, nDepths]) = true;
                end
                
            case 6  % Remove first & last 10%
                num_to_remove = max(1, round(nDepths * 0.1));
                tx_outliers([1:num_to_remove, (nDepths-num_to_remove+1):nDepths]) = true;
                ty_outliers([1:num_to_remove, (nDepths-num_to_remove+1):nDepths]) = true;
                
            case 7  % Remove first & last 20%
                num_to_remove = max(1, round(nDepths * 0.2));
                tx_outliers([1:num_to_remove, (nDepths-num_to_remove+1):nDepths]) = true;
                ty_outliers([1:num_to_remove, (nDepths-num_to_remove+1):nDepths]) = true;
                
            case 8  % Mean ± 2×STD
                % For tx
                mean_tx = mean(tx_data);
                std_tx = std(tx_data);
                tx_outliers = abs(tx_data - mean_tx) > 2 * std_tx;
                
                % For ty
                mean_ty = mean(ty_data);
                std_ty = std(ty_data);
                ty_outliers = abs(ty_data - mean_ty) > 2 * std_ty;
        end
        
        % Create indices for non-outlier data
        valid_tx_indices = find(~tx_outliers);
        valid_ty_indices = find(~ty_outliers);
        
        % Fit tx and ty with the specified polynomial order using only non-outlier data
        if ~isempty(valid_tx_indices)
            p_tx = polyfit(valid_tx_indices, tx_data(valid_tx_indices), order);
            tx_fit = polyval(p_tx, depth_indices);
        else
            p_tx = polyfit(depth_indices, tx_data, order);
            tx_fit = polyval(p_tx, depth_indices);
        end
        
        if ~isempty(valid_ty_indices)
            p_ty = polyfit(valid_ty_indices, ty_data(valid_ty_indices), order);
            ty_fit = polyval(p_ty, depth_indices);
        else
            p_ty = polyfit(depth_indices, ty_data, order);
            ty_fit = polyval(p_ty, depth_indices);
        end
        
        % Update tform_bkwd_rel with fitted values
        for j = 1:nDepths
            A = tform_bkwd_rel(view_idx,j).A;
            A(1,3) = tx_fit(j);
            A(2,3) = ty_fit(j);
            tform_bkwd_rel(view_idx,j) = affinetform2d(A);
        end
        
        % Delete the old subplot entirely and create a new one
        delete(subplots{view_idx});
        subplots{view_idx} = subplot(nRows, nCols, view_idx);
        
        % Arrays to store plot handles for legend
        plotHandles = [];
        legendLabels = {};
        
        % Left y-axis for tx and ty
        yyaxis(subplots{view_idx}, 'left');
        
        % Plot normal data points and store handles
        h1 = plot(depth_indices(~tx_outliers), tx_data(~tx_outliers), 'bo');
        hold(subplots{view_idx}, 'on');
        plotHandles = [plotHandles, h1];
        legendLabels{end+1} = 'Original tx';
        
        h2 = plot(depth_indices(~ty_outliers), ty_data(~ty_outliers), 'go');
        plotHandles = [plotHandles, h2];
        legendLabels{end+1} = 'Original ty';
        
        % Mark outliers with 'x' if detected
        if any(tx_outliers)
            h3 = plot(depth_indices(tx_outliers), tx_data(tx_outliers), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
            plotHandles = [plotHandles, h3];
            legendLabels{end+1} = 'tx outliers';
        end
        
        if any(ty_outliers)
            h4 = plot(depth_indices(ty_outliers), ty_data(ty_outliers), 'mx', 'MarkerSize', 10, 'LineWidth', 2);
            plotHandles = [plotHandles, h4];
            legendLabels{end+1} = 'ty outliers';
        end
        
        % Plot fitted lines
        h5 = plot(depth_indices, tx_fit, 'r-', 'LineWidth', 2);
        plotHandles = [plotHandles, h5];
        legendLabels{end+1} = 'Fitted tx';
        
        h6 = plot(depth_indices, ty_fit, 'm-', 'LineWidth', 2);
        plotHandles = [plotHandles, h6];
        legendLabels{end+1} = 'Fitted ty';
        
        hold(subplots{view_idx}, 'off');
        ylabel('Translation Parameters');
        
        % Right y-axis for other matrix elements
        yyaxis(subplots{view_idx}, 'right');
        hold(subplots{view_idx}, 'on');
        
        h7 = plot(depth_indices, a11_orig(view_idx,:), 'c.');
        plotHandles = [plotHandles, h7];
        legendLabels{end+1} = 'A_{11}';
        
        h8 = plot(depth_indices, a12_orig(view_idx,:), 'y.');
        plotHandles = [plotHandles, h8];
        legendLabels{end+1} = 'A_{12}';
        
        h9 = plot(depth_indices, a21_orig(view_idx,:), 'k.');
        plotHandles = [plotHandles, h9];
        legendLabels{end+1} = 'A_{21}';
        
        h10 = plot(depth_indices, a22_orig(view_idx,:), 'r.');
        plotHandles = [plotHandles, h10];
        legendLabels{end+1} = 'A_{22}';
        
        hold(subplots{view_idx}, 'off');
        ylabel('Other Matrix Elements');
        
        % General plot settings
        title_text = sprintf('View %d (Order %d, %s)', view_idx, order, outlier_options{outlier_method});
        title(subplots{view_idx}, title_text);
        
        xlabel(subplots{view_idx}, 'Depth Index');
        grid(subplots{view_idx}, 'on');
        
        % Create legend explicitly with the handles we collected
        legend(subplots{view_idx}, plotHandles, legendLabels, 'Location', 'bestoutside');
    end

    % Nested function: Dropdown callback for polynomial order
    function order_callback(src, ~, view_idx)
        selected_idx = get(src, 'Value');
        view_orders(view_idx) = selected_idx; % Update the order for this view
        update_plot(view_idx);
    end
    
    % Nested function: Dropdown callback for outlier method
    function outlier_callback(src, ~, view_idx)
        selected_idx = get(src, 'Value');
        outlier_methods(view_idx) = selected_idx; % Update outlier method
        update_plot(view_idx);
    end

    % Nested function: Handle figure close
    function close_figure(src, ~)
        % The figure close callback - no special action needed
        % as tform_bkwd_rel has already been updated
        delete(src);
    end
end