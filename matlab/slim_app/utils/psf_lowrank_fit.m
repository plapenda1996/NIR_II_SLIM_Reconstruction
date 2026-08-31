function [PSF_approx, basis_psfs, weights, fitted_weights] = psf_lowrank_fit(PSFs, K, poly_orders)
% Multi-view 3D PSF low-rank approximation with interactive polynomial fitting
% Inputs:
% PSFs: 4D PSF with dimensions [nx, ny, nz, nViews]
% K: Rank of the low-rank approximation
% poly_orders: Polynomial fitting orders for each weight, with dimensions [nViews, K], optional
%
% Outputs:
% PSF_approx: Fitted low-rank approximation PSF, dimensions [nx, ny, nz, nViews]
% basis_psfs: K basis PSFs, dimensions [nx, ny, K, nViews]
% weights: Original weight matrix, dimensions [nz, K, nViews]
% fitted_weights: Fitted weight matrix, dimensions [nz, K, nViews]

% Get dimensions
[nx, ny, nz, nViews] = size(PSFs);

% Initialize outputs
basis_psfs = zeros(nx, ny, K, nViews);
weights = zeros(nz, K, nViews);
fitted_weights = zeros(nz, K, nViews);
PSF_approx = zeros(nx, ny, nz, nViews);

% Check and adjust poly_orders
if nargin < 3 || isempty(poly_orders)
    poly_orders = 3 * ones(nViews, K);
elseif size(poly_orders, 1) == 1 && size(poly_orders, 2) == 1
    poly_orders = poly_orders * ones(nViews, K);
elseif size(poly_orders, 1) == 1 && size(poly_orders, 2) == K
    poly_orders = repmat(poly_orders, nViews, 1);
elseif size(poly_orders, 1) == nViews && size(poly_orders, 2) == 1
    poly_orders = repmat(poly_orders, 1, K);
end

% Process SVD decomposition for each view
for v = 1:nViews
    % Extract current 3D PSF from 4D PSF
    current_PSF = PSFs(:, :, :, v);
    
    % Reshape PSF to matrix: [nz, nx * ny]
    M = zeros(nz, nx * ny);
    for d = 1:nz
        M(d, :) = reshape(current_PSF(:, :, d), 1, []);
    end
    
    % Perform SVD
    [U, S, V] = svd(M, 'econ');
    
    % Truncate to rank K
    K_actual = min(K, min(size(M))); % Ensure K doesn't exceed rank of M
    U_k = U(:, 1:K_actual);
    S_k = diag(S(1:K_actual, 1:K_actual));
    V_k = V(:, 1:K_actual);
    
    % Extract weights and basis PSFs
    weights(:, 1:K_actual, v) = U_k * diag(S_k);
    
    % Basis PSFs - spatial patterns
    for k = 1:K_actual
        basis_psfs(:, :, k, v) = reshape(V_k(:, k), [nx, ny]);
    end
end

% Initialize interactive controls
outlier_methods = {'none', 'first', 'last', 'first10p', 'last10p', 'first20p', 'last20p', 'auto'};
current_method = 'none';
current_view = 1;
normalize_coefs = true;  % Default: normalize coefficients
apply_outliers_to_all = true;  % Default: apply same outliers to all coefficients

% Pre-compute fitted weights for all views using default parameters
z_indices = 1:nz;
for v = 1:nViews
    K_actual = min(K, min(size(weights(:,:,v))));
    for k = 1:K_actual
        coef = weights(:, k, v);
        [~, outlier_mask] = remove_outliers(z_indices, coef, current_method);
        
        % Get valid samples
        valid_indices = find(~outlier_mask);
        valid_z = z_indices(valid_indices);
        valid_coef = coef(valid_indices);
        
        % Ensure enough points for fitting
        if length(valid_z) > poly_orders(v, k)
            % Fit polynomial
            p = polyfit(valid_z, valid_coef, poly_orders(v, k));
            
            % Calculate fitted values
            fitted_weights(:, k, v) = polyval(p, z_indices);
        else
            % If not enough points, use original coefficients
            warning('Not enough points for %d-th order polynomial fit in view %d. Using original values.', poly_orders(v, k), v);
            fitted_weights(:, k, v) = coef;
        end
    end
    
    % Reconstruct PSF for this view
    current_PSF_approx = zeros(nx, ny, nz);
    for d = 1:nz
        for k = 1:K_actual
            current_PSF_approx(:, :, d) = current_PSF_approx(:, :, d) + ...
                fitted_weights(d, k, v) * basis_psfs(:, :, k, v);
        end
    end
    PSF_approx(:, :, :, v) = current_PSF_approx;
end

% Create interactive figure
fig = figure('Position', [100, 100, 1200, 900], 'Name', '3D PSF Low-Rank Approximation and Fitting', 'NumberTitle', 'off');

% Create UI panel for all controls
control_panel = uipanel('Position', [0, 0.9, 1, 0.1], 'BackgroundColor', [0.9, 0.9, 0.9]);

% Create UI controls - using relative positions
uicontrol(control_panel, 'Style', 'text', 'Position', [20, 50, 100, 20], 'String', 'View:');
view_popup = uicontrol(control_panel, 'Style', 'popup', 'Position', [120, 50, 100, 20], ...
    'String', arrayfun(@(x) ['View ' num2str(x)], 1:nViews, 'UniformOutput', false), ...
    'Value', 1, 'Callback', @change_view);

uicontrol(control_panel, 'Style', 'text', 'Position', [240, 50, 100, 20], 'String', 'Outlier Method:');
outlier_popup = uicontrol(control_panel, 'Style', 'popup', 'Position', [340, 50, 150, 20], ...
    'String', outlier_methods, 'Value', 1, 'Callback', @change_outlier_method);

% Add polynomial order controls
uicontrol(control_panel, 'Style', 'text', 'Position', [20, 15, 100, 20], 'String', 'Polynomial Order:');
order_edit = uicontrol(control_panel, 'Style', 'edit', 'Position', [120, 15, 50, 20], ...
    'String', num2str(poly_orders(1, 1)), 'Callback', @update_current_view_order);

% Add coefficient normalization toggle
norm_checkbox = uicontrol(control_panel, 'Style', 'checkbox', 'Position', [320, 15, 120, 20], ...
    'String', 'Normalize', 'Value', normalize_coefs, 'Callback', @toggle_normalization);

% Add outlier application method toggle
outlier_checkbox = uicontrol(control_panel, 'Style', 'checkbox', 'Position', [450, 15, 180, 20], ...
    'String', 'Apply outliers to all', 'Value', apply_outliers_to_all, 'Callback', @toggle_outliers_application);

% Initial update of plots
update_plots();

% Wait for user interaction until figure is closed
uiwait(fig);

% Nested callback function: change view
    function change_view(src, ~)
        current_view = src.Value;
        update_plots();
    end

% Nested callback function: change outlier method
    function change_outlier_method(src, ~)
        current_method = outlier_methods{src.Value};
        % Recalculate fits for current view only
        update_fitted_weights(current_view);
        update_plots();
    end

% Nested callback function: toggle coefficient normalization display
    function toggle_normalization(src, ~)
        normalize_coefs = src.Value;
        update_plots();
    end

% Nested callback function: toggle outlier application method
    function toggle_outliers_application(src, ~)
        apply_outliers_to_all = src.Value;
        % Recalculate fits for current view only
        update_fitted_weights(current_view);
        update_plots();
    end

% Nested callback function: update current view's fitting order
    function update_current_view_order(src, ~)
        % Get current view
        v = current_view;
        
        % Get new order
        new_order = str2double(src.String);
        
        % Validate input
        if isnan(new_order) || new_order < 1
            errordlg('Please enter a valid polynomial order (positive integer)', 'Input Error');
            src.String = num2str(poly_orders(v, 1)); % Restore original value
            return;
        end
        
        % Update all coefficients' polynomial orders for current view
        K_actual = min(K, min(size(weights(:,:,v))));
        poly_orders(v, 1:K_actual) = new_order;
        disp(['Updated view ' num2str(v) ' to order ' num2str(new_order)]);
        
        % Update fitted weights for this view only
        update_fitted_weights(v);
        
        % Update plots
        update_plots();
    end

% Nested function: update fitted weights for a specific view
    function update_fitted_weights(v)
        K_actual = min(K, min(size(weights(:,:,v))));
        z_indices = 1:nz;
        
        % Determine global outlier mask (if needed)
        global_outlier_mask = false(nz, 1);
        if apply_outliers_to_all && strcmp(current_method, 'auto')
            % First identify outliers in each coefficient
            all_masks = false(nz, K_actual);
            for k = 1:K_actual
                coef = weights(:, k, v);
                [~, mask] = remove_outliers(z_indices, coef, current_method);
                all_masks(:, k) = mask;
            end
            % If any coefficient marks a point as outlier, mark it in global mask
            global_outlier_mask = any(all_masks, 2);
        end
        
        % Update each coefficient
        for k = 1:K_actual
            coef = weights(:, k, v);
            
            % Apply outlier detection and removal
            if apply_outliers_to_all && ~strcmp(current_method, 'none')
                if strcmp(current_method, 'auto')
                    % Use global mask
                    outlier_mask = global_outlier_mask;
                else
                    % Use same outlier method but without considering auto detection results
                    [~, outlier_mask] = remove_outliers(z_indices, coef, current_method);
                end
            else
                % Detect outliers individually for each coefficient
                [~, outlier_mask] = remove_outliers(z_indices, coef, current_method);
            end
            
            % Get valid samples
            valid_indices = find(~outlier_mask);
            valid_coef = coef(valid_indices);
            valid_z = z_indices(valid_indices);
            
            % Ensure enough points for fitting
            if length(valid_z) > poly_orders(v, k)
                % Fit polynomial
                p = polyfit(valid_z, valid_coef, poly_orders(v, k));
                
                % Calculate fitted values
                fitted_weights(:, k, v) = polyval(p, z_indices);
            else
                % If not enough points, use original coefficients
                warning('Not enough points for %d-th order polynomial fit. Using original values.', poly_orders(v, k));
                fitted_weights(:, k, v) = coef;
            end
        end
        
        % Reconstruct PSF for this view
        current_PSF_approx = zeros(nx, ny, nz);
        for d = 1:nz
            for k = 1:K_actual
                current_PSF_approx(:, :, d) = current_PSF_approx(:, :, d) + ...
                    fitted_weights(d, k, v) * basis_psfs(:, :, k, v);
            end
        end
        PSF_approx(:, :, :, v) = current_PSF_approx;
    end

% Nested function: update all plots
    function update_plots()
        % Clear plot area but keep control panel
        clf;
        
        % Recreate UI panel for all controls
        control_panel = uipanel('Position', [0, 0.9, 1, 0.1], 'BackgroundColor', [0.9, 0.9, 0.9]);
        
        % Create UI controls - using relative positions
        uicontrol(control_panel, 'Style', 'text', 'Position', [20, 50, 100, 20], 'String', 'View:');
        view_popup = uicontrol(control_panel, 'Style', 'popup', 'Position', [120, 50, 100, 20], ...
            'String', arrayfun(@(x) ['View ' num2str(x)], 1:nViews, 'UniformOutput', false), ...
            'Value', current_view, 'Callback', @change_view);
        
        uicontrol(control_panel, 'Style', 'text', 'Position', [240, 50, 100, 20], 'String', 'Outlier Method:');
        outlier_popup = uicontrol(control_panel, 'Style', 'popup', 'Position', [340, 50, 150, 20], ...
            'String', outlier_methods, 'Value', find(strcmp(outlier_methods, current_method)), ...
            'Callback', @change_outlier_method);
        
        % Get current view's coefficient count
        v = current_view;
        K_actual = min(K, min(size(weights(:,:,v))));
        
        % Display SVD energy ratio in control panel
        current_PSF = PSFs(:, :, :, v);
        
        % Reshape PSF to matrix
        [nx, ny, nz] = size(current_PSF);
        M = zeros(nz, nx * ny);
        for d = 1:nz
            M(d, :) = reshape(current_PSF(:, :, d), 1, []);
        end
        
        % Calculate complete SVD
        [~, S, ~] = svd(M, 'econ');
        S_values = diag(S);
        total_energy = sum(S_values.^2);
        K_actual = min(K, length(S_values));
        K_energy = sum(S_values(1:K_actual).^2);
        energy_ratio = K_energy / total_energy * 100;
        
        % Display energy ratio and current settings
        uicontrol(control_panel, 'Style', 'text', 'Position', [510, 50, 350, 20], ...
            'String', ['SVD Energy: First ' num2str(K_actual) ' components: ' num2str(energy_ratio, '%.2f') '%'], ...
            'HorizontalAlignment', 'left');
            
        uicontrol(control_panel, 'Style', 'text', 'Position', [580, 15, 280, 20], ...
            'String', ['Current: View ' num2str(current_view) ', Outlier: ' current_method], ...
            'HorizontalAlignment', 'left');
        
        % Add polynomial order controls
        uicontrol(control_panel, 'Style', 'text', 'Position', [20, 15, 100, 20], 'String', 'Polynomial Order:');
        order_edit = uicontrol(control_panel, 'Style', 'edit', 'Position', [120, 15, 50, 20], ...
            'String', num2str(poly_orders(v, 1)), 'Callback', @update_current_view_order);

        % Add coefficient normalization toggle
        norm_checkbox = uicontrol(control_panel, 'Style', 'checkbox', 'Position', [320, 15, 120, 20], ...
            'String', 'Normalize', 'Value', normalize_coefs, 'Callback', @toggle_normalization);

        % Add outlier application method toggle
        outlier_checkbox = uicontrol(control_panel, 'Style', 'checkbox', 'Position', [450, 15, 180, 20], ...
            'String', 'Apply outliers to all', 'Value', apply_outliers_to_all, 'Callback', @toggle_outliers_application);
        
        % Adjust coefficient fitting plot
        subplot('Position', [0.07, 0.64, 0.40, 0.24]);
        hold on;
        
        z_indices = 1:nz;
        
        % Prepare data structures for normalized coefficients
        normalized_weights = zeros(size(weights(:, 1:K_actual, v)));
        
        % If normalization needed, calculate normalized values for each coefficient
        if normalize_coefs
            for k = 1:K_actual
                coef = weights(:, k, v);
                coef_min = min(coef);
                coef_max = max(coef);
                % Avoid division by zero (if all coefficients are equal)
                if coef_max > coef_min
                    normalized_weights(:, k) = (coef - coef_min) / (coef_max - coef_min);
                else
                    normalized_weights(:, k) = 0.5 * ones(size(coef)); % If all values are the same, set to 0.5
                end
            end
        else
            % If normalization not needed, use original weights
            normalized_weights = weights(:, 1:K_actual, v);
        end
        
        % Prepare normalized fitted weights for display
        normalized_fitted_weights = zeros(size(fitted_weights(:, 1:K_actual, v)));
        if normalize_coefs
            for k = 1:K_actual
                orig_coef = weights(:, k, v);  % Original coefficient
                coef_min = min(orig_coef);
                coef_max = max(orig_coef);
                % Avoid division by zero (if all coefficients are equal)
                if coef_max > coef_min
                    normalized_fitted_weights(:, k) = (fitted_weights(:, k, v) - coef_min) / (coef_max - coef_min);
                else
                    normalized_fitted_weights(:, k) = 0.5 * ones(size(fitted_weights(:, k, v))); % If all values are the same, set to 0.5
                end
            end
        else
            % If normalization not needed, use fitted weights directly
            normalized_fitted_weights = fitted_weights(:, 1:K_actual, v);
        end
        
        % Determine global outlier mask (if needed)
        global_outlier_mask = false(nz, 1);
        if apply_outliers_to_all && strcmp(current_method, 'auto')
            % First identify outliers in each coefficient
            all_masks = false(nz, K_actual);
            for k = 1:K_actual
                coef = weights(:, k, v);
                [~, mask] = remove_outliers(z_indices, coef, current_method);
                all_masks(:, k) = mask;
            end
            % If any coefficient marks a point as outlier, mark it in global mask
            global_outlier_mask = any(all_masks, 2);
        end
        
        % Use different colors for each coefficient and its fitting curve
        colors = lines(K_actual);
        legend_entries = cell(1, K_actual);
        
        % Store line handles for legend
        plot_handles = zeros(K_actual, 1);
        
        % Plot each coefficient and its fitting curve
        for k = 1:K_actual
            % Extract current coefficient
            coef = normalized_weights(:, k);
            fitted_values = normalized_fitted_weights(:, k);
            
            % Determine outlier mask
            if apply_outliers_to_all && ~strcmp(current_method, 'none')
                if strcmp(current_method, 'auto')
                    % Use global mask
                    outlier_mask = global_outlier_mask;
                else
                    % Use same outlier method but without considering auto detection results
                    [~, outlier_mask] = remove_outliers(z_indices, weights(:, k, v), current_method);
                end
            else
                % Detect outliers individually for each coefficient
                [~, outlier_mask] = remove_outliers(z_indices, weights(:, k, v), current_method);
            end
            
            % Plot original coefficient points
            scatter(z_indices, coef, 20, colors(k,:), 'filled');
            
            % Mark outliers (if any)
            if any(outlier_mask)
                scatter(z_indices(outlier_mask), coef(outlier_mask), 40, colors(k,:), 'x', 'LineWidth', 1.5);
            end
            
            % Plot fitting curve
            h_line = plot(z_indices, fitted_values, 'Color', colors(k,:), 'LineWidth', 1.5);
            plot_handles(k) = h_line;
            
            % Add legend entry
            legend_entries{k} = ['Coef ' num2str(k) ' (Order=' num2str(poly_orders(v, k)) ')'];
        end
        
        % Set title and labels
        title_str = sprintf('View %d - Coefficients and Fits', v);
        if normalize_coefs
            title_str = [title_str ' (Normalized)'];
            ylabel_str = 'Normalized Coefficient (0-1)';
        else
            ylabel_str = 'Coefficient Value';
        end
        
        title(title_str);
        xlabel('Z Position');
        ylabel(ylabel_str);
        
        % Only use fitting lines for legend
        if K_actual <= 10
            legend(plot_handles, legend_entries, 'Location', 'best');
        end
        
        % Display normalization status and order information
        if normalize_coefs
            text(0.02, 0.98, 'Normalized Display (0-1)', 'Units', 'normalized', 'VerticalAlignment', 'top', 'FontWeight', 'bold');
        end
        
        grid on;
        hold off;
        
        % Get current reconstructed PSF
        current_PSF_approx = PSF_approx(:, :, :, v);
        
        % Display original PSF's XZ slice
        subplot('Position', [0.53, 0.64, 0.40, 0.24]);
        mid_y = round(ny/2);
        original_xz_slice = squeeze(PSFs(:, mid_y, :, v))';
        imagesc(original_xz_slice);
        title(['Original View ' num2str(v) ' - XZ Slice (y=' num2str(mid_y) ')']);
        xlabel('X');
        ylabel('Z');
        colorbar;
        
        % Display original PSF's YZ slice
        subplot('Position', [0.07, 0.34, 0.40, 0.24]);
        mid_x = round(nx/2);
        original_yz_slice = squeeze(PSFs(mid_x, :, :, v))';
        imagesc(original_yz_slice);
        title(['Original View ' num2str(v) ' - YZ Slice (x=' num2str(mid_x) ')']);
        xlabel('Y');
        ylabel('Z');
        colorbar;
        
        % Display reconstructed PSF's XZ slice
        subplot('Position', [0.53, 0.34, 0.40, 0.24]);
        xz_slice = squeeze(current_PSF_approx(:, mid_y, :))';
        imagesc(xz_slice);
        title(['Reconstructed View ' num2str(v) ' - XZ Slice (y=' num2str(mid_y) ')']);
        xlabel('X');
        ylabel('Z');
        colorbar;
        
        % Display reconstructed PSF's YZ slice
        subplot('Position', [0.07, 0.04, 0.40, 0.24]);
        yz_slice = squeeze(current_PSF_approx(mid_x, :, :))';
        imagesc(yz_slice);
        title(['Reconstructed View ' num2str(v) ' - YZ Slice (x=' num2str(mid_x) ')']);
        xlabel('Y');
        ylabel('Z');
        colorbar;
        
        % Display reconstruction error XZ slice
        subplot('Position', [0.53, 0.04, 0.40, 0.24]);
        error_xz = abs(original_xz_slice - xz_slice);
        imagesc(error_xz);
        
        % Calculate RMSE and PSNR
        rmse_xz = sqrt(mean(error_xz(:).^2));
        max_val = max(original_xz_slice(:));
        if rmse_xz > 0 && max_val > 0
            psnr_xz = 20 * log10(max_val / rmse_xz);
        else
            psnr_xz = Inf;
        end
        
        title({['Error View ' num2str(v) ' - XZ Slice'], ...
              ['RMSE = ' num2str(rmse_xz, '%.3e') ', PSNR = ' num2str(psnr_xz, '%.2f') ' dB']});
        xlabel('X');
        ylabel('Z');
        colorbar;
    end

% Nested function: remove outliers
    function [valid_z, outlier_mask] = remove_outliers(z_values, coef, method)
        n = length(z_values);
        outlier_mask = false(n, 1);
        
        switch method
            case 'none'
                % Don't remove any points
            case 'first'
                % Remove first sample
                outlier_mask(1) = true;
            case 'last'
                % Remove last sample
                outlier_mask(end) = true;
            case 'first10p'
                % Remove first 10%
                num_to_remove = max(1, round(n * 0.1));
                outlier_mask(1:num_to_remove) = true;
            case 'last10p'
                % Remove last 10%
                num_to_remove = max(1, round(n * 0.1));
                outlier_mask(end-num_to_remove+1:end) = true;
            case 'first20p'
                % Remove first 20%
                num_to_remove = max(1, round(n * 0.2));
                outlier_mask(1:num_to_remove) = true;
            case 'last20p'
                % Remove last 20%
                num_to_remove = max(1, round(n * 0.2));
                outlier_mask(end-num_to_remove+1:end) = true;
            case 'auto'
                % Use interquartile range method to automatically detect outliers
                q1 = quantile(coef, 0.25);
                q3 = quantile(coef, 0.75);
                iqr = q3 - q1;
                % If IQR is close to zero, use standard deviation as alternative
                if iqr < eps * 1000
                    mu = mean(coef);
                    sigma = std(coef);
                    if sigma < eps * 1000
                        % If standard deviation is also very small, all values are nearly equal, don't remove any points
                    else
                        % Use 3x standard deviation as threshold
                        outlier_mask = abs(coef - mu) > 3 * sigma;
                    end
                else
                    % Use standard IQR method
                    lower_bound = q1 - 1.5 * iqr;
                    upper_bound = q3 + 1.5 * iqr;
                    outlier_mask = coef < lower_bound | coef > upper_bound;
                end
        end
        
        valid_z = z_values(~outlier_mask);
    end
end