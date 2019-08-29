savePath = fullfile('D:\','EddiesUV','uvelfull\');
%savePath = fullfile('D:\','EddiesUV');
baseDirectory = 'http://tds.hycom.org/thredds/dodsC/datasets/GOMl0.04/expt_02.2/';
baseYear = 1900;
variable = 'u';

for i = 101:102
    fprintf('Initializing download of year %d:\n', (baseYear + i));
    yearDirectory = strcat(baseDirectory, num2str(baseYear + i), '-1/z3d/022GOMl0.04-', num2str(baseYear + i), '_');
    if (leapyear(baseYear + i)); lastDay = 366; else lastDay = 365; end
    x = cell(lastDay, 1);
    for j = 1:lastDay
    %for j = 1:3
        try
            dayDirectory = strcat(yearDirectory, sprintf('%03d', j), '_00_', variable, '.nc');
            temp = ncread(dayDirectory, variable);
            x{j} = temp(:,:,:);
            fprintf('Done - Day %d\n', j);
        catch Exception
            fprintf('Failed - Day %d\n', j);
        end
    end
    save(strcat(savePath,variable,"velfull", num2str(baseYear + i), '.mat'), 'x', '-v7.3');
    fprintf('\n');
end
%D:\EddiesUV\uvelfull

function status = leapyear(year)
    if mod(year, 4) == 0
        if mod(year, 100) == 0
            if mod(year,400) == 0
                status = true;
            else
                status = false;
            end
        else
            status = true;
        end
    else
        status = false;
    end
end