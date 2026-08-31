function time_str = formatTime(seconds)
    % 格式化时间显示
    if seconds < 60
        time_str = sprintf('%.1f seconds', seconds);
    elseif seconds < 3600
        minutes = floor(seconds / 60);
        remaining_seconds = mod(seconds, 60);
        time_str = sprintf('%d min %.0f sec', minutes, remaining_seconds);
    else
        hours = floor(seconds / 3600);
        remaining_minutes = floor(mod(seconds, 3600) / 60);
        time_str = sprintf('%d h %d min', hours, remaining_minutes);
    end
end