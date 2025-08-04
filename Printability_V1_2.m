%Image julien
function gridAnalysisTool()
    % gridAnalysisTool - Outil d'analyse d'images de grilles imprimées en 3D
    % Permet d'analyser la printabilité, le taux d'étalement et de comparer 
    % la grille théorique à la grille réelle
    clear all;
    clc;
    close all;

    [filenames, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.bmp','Image Files (*.jpg,*.png,*.tif,*.bmp)'},...
        'Select an image','MultiSelect', 'on');
    if isequal(filenames, 0)
        disp('Selection deleted');
        return;
    end
    if ischar(filenames)
        filenames = {filenames};
    end
    fprintf('Selected %d image(s)\n', length(filenames));
    
    for i = 1:length(filenames)
        filename = filenames{i};
        filename = fullfile(pathname, filenames{i});
        img = imread(filename); % Lire l'image
        figure('Name', sprintf('Original image - %s',filenames{i}));   
        imshow(img); % Afficher l'image
    end
    
    % Image processing
    %img_sharp = imsharpen(img, 'Radius', 6, 'Amount', 8);
    grayImage = rgb2gray(img);
    grayImage = imadjust(grayImage);
    %img_filt = medfilt2(grayImage, [8 8]);
    %img_filt = imbinarize(img_filt, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.28);
    edges = edge(grayImage, 'Canny', 0.2);
    
    % Connect boudaries
    se = strel('disk', 3); %3
    dilatedEdges = imdilate(edges, se);
    % Fill holes to have closed regions
    filledEdges = imfill(dilatedEdges, 'holes');
    
    % Isolate holes that is interior region 
    holes = filledEdges & ~dilatedEdges;
    holes = bwareaopen(holes, 1000);
    figure;
    imshow(holes);
    cc = bwconncomp(holes);
    stats = regionprops(cc, 'Area', 'BoundingBox', 'Centroid', 'Solidity', 'Perimeter');

    % Calculate image center 
    imgCenter = [size(grayImage, 2)/2, size(grayImage, 1)/2];
    
    % Select element closer to the center 
    centroids = vertcat(stats.Centroid);
    distances = zeros(length(stats), 1);
    for i = 1:length(stats)
        distances(i) = norm(centroids(i,:) - imgCenter);
    end
    [~, centerIndex] = min(distances);
    
    % Create a mask for inside square/pore
    labelMatrix = labelmatrix(cc);
    squareMask = false(size(grayImage));
    squareMask(labelMatrix == centerIndex) = true;
    %squareMask = imcomplement(squareMask);
    
    figure;
    imshow(squareMask);
    figure('Name', 'Détection du carré intérieur');
    imshow(img);
    hold on;
    
    % Détection de l'échelle
    scale = input('What is the scale of all images (µm) : ');
    edges = edge(holes, 'Canny');
    [H, theta, rho] = hough(edges);
    peaks = houghpeaks(H, 5, 'threshold', ceil(0.6 * max(H(:))));
    lines = houghlines(edges, theta, rho, peaks, 'FillGap', 5, 'MinLength', 50);
    
    figure('Name', sprintf('Scale detection - %s', filenames{1}));
    imshow(img);
    hold on;

    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        if abs(xy(1,2) - xy(2,2)) < 10  
            plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
            lineLength = norm(xy(1,:) - xy(2,:));
            % scale
            pixel2microm = scale / lineLength;
            fprintf('1 pixel = %.2f µm\n', pixel2microm);
            break;
        end
    end
    
    % Principal menu
    while true
        menu_printability = input(['What do you want to do :\n' ...
            'Press 1 : Calculate Printability Pr\n' ...
            'Press 2 : Calculate Spreading Rate Sp\n' ...
            'Press 3 : Compare the theorical and real grid\n' ...
            'Press 4 : Exit\n' ...
            'Your choice, press 1, 2, 3 or 4 : ']);
        
        if menu_printability == 1
            analyzePrintability(squareMask, pixel2microm, img);
        elseif menu_printability == 2
            analyzeSp(filenames, pathname, pixel2microm);
        elseif menu_printability == 3
            analyzeCompareArea(filenames, pathname, pixel2microm, squareMask);
        elseif menu_printability == 4
            break;
        else
            fprintf("Choose a number between 1, 2, 3, and 4\n");
        end
    end
end

function printability = analyzePrintability(squareMask, pixel2microm, img)
    
    props = regionprops(squareMask, 'Area', 'Perimeter');
    airePixels = props.Area;
    perimetrePixels = props.Perimeter;
    aire_microns = airePixels * (pixel2microm^2);
    perimetre_microns = perimetrePixels * pixel2microm;
    fprintf('Interior square area: %f μm²\n', aire_microns);
    fprintf('Interior square perimeter: %f μm\n', perimetre_microns);
    
    printability = (perimetre_microns^2) / (16*aire_microns);
    printability = round(printability, 3);
    fprintf('Printability = %.3f\n', printability);
    
    if printability == 1
        fprintf("Pr = 1, Good Gelification\n");
    elseif printability < 1
        fprintf("Pr < 1, Under Gelification\n");
    elseif printability > 1
        fprintf("Pr > 1, Over Gelification\n");
    end
    
    figure;
    imshow(img);
    hold on;
    % Draw boundaries 
    boundaries = bwboundaries(squareMask, 'holes');
    plot(boundaries{1}(:,2), boundaries{1}(:,1), 'r', 'LineWidth', 2);
    title(sprintf('Printability, Pr = %.3f', printability));
end

function filament_diameter = measure_filament_by_two_lines(img, pixel2microm)
    figure; imshow(img); title('Click 4 points (2 per line)'); hold on;
    fprintf('Click 4 points (2 per line) to measure filament thickness\n');

    % Collect 4 points
    points = zeros(4, 2);
    for i = 1:4
        [x, y] = ginput(1);
        plot(x, y, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
        points(i, :) = [x, y];
    end

    line1 = points(1:2, :);
    line2 = points(3:4, :);
    plot(line1(:,1), line1(:,2), 'g-', 'LineWidth', 2);
    plot(line2(:,1), line2(:,2), 'b-', 'LineWidth', 2);

    % Calculate distance
    d1 = norm(line1(2,:) - line1(1,:)) * pixel2microm;
    d2 = norm(line2(2,:) - line2(1,:)) * pixel2microm;

    filament_diameter = mean([d1, d2]);

    fprintf('Diamètre ligne 1 : %.2f µm\n', d1);
    fprintf('Diamètre ligne 2 : %.2f µm\n', d2);
    fprintf('Diamètre moyen du filament : %.2f µm\n', filament_diameter);
end

function analyzeSp(filenames, pathname, pixel2microm)
% analyzeSp - Analyzes the spreading rate
% Measures the diameter of the filaments relative to the needle diameter
   
    final_table = [];
    for file_idx = 1:length(filenames)
        filename = filenames{file_idx};
        fprintf('\nImage processing %d/%d : %s\n', file_idx, length(filenames), filename);
 
        img = imread(fullfile(pathname, filename));        
        needle_diameter = input('What is the intern diameter of the needle (µm) : ');
      
        list_sp = [];
        list_filament_diameter = [];
        fprintf('Click on filament edges to measure diameter (click outside image to finish)\n');
        drawing = true;

        while drawing
            [x, y, button] = ginput(1);
            if button ~= 1 
                drawing = false;
                break;
            end
            filament_diameter = measure_filament_by_two_lines(img, pixel2microm);
            sp = filament_diameter / needle_diameter;
            list_sp = [list_sp; sp];
            list_filament_diameter = [list_filament_diameter; filament_diameter];
            fprintf('Filament diameter : %.2f µm | ', filament_diameter);
            fprintf('Sp : %.2f\n', sp);
        
end

        fprintf('List Sp\n');
        disp(list_sp);
        mean_sp = mean(list_sp);
        fprintf('Mean Sp : %.2f\n', mean_sp);
        std_sp = std(list_sp);
        fprintf('Standard deviation Sp : %.2f\n', std_sp);
        
        % Save on an Excel file
        fprintf('\nSave results on a file\n');
        image_name_row = table(string(filename), 'VariableNames', {'Image_Name'});
        filaments_sp = (1:length(list_sp))'; 
        results_table1 = table(filaments_sp, round(list_sp(:),2), 'VariableNames', {'Filament number','Sp'});
        results_table2 = table(round(mean_sp,2), round(std_sp,2), 'VariableNames', {'Mean Sp', 'Standard deviation Sp'});
        
        max_rows = max(height(results_table1), height(results_table2));
        results_table1 = [results_table1; array2table(nan(max_rows - height(results_table1), width(results_table1)), ...
            'VariableNames', results_table1.Properties.VariableNames)];
        
        results_table2 = [results_table2; array2table(nan(max_rows - height(results_table2), width(results_table2)), ...
            'VariableNames', results_table2.Properties.VariableNames)];
        
        image_name_row = [image_name_row; array2table(repmat({nan}, max_rows - height(image_name_row), 1), ...
            'VariableNames', image_name_row.Properties.VariableNames)];
        
        % Combine table
        combined_table = [image_name_row, results_table1, results_table2];
        final_table = [final_table; combined_table];
    end
    
    % Save table
    [save_file, save_path] = uiputfile('*.xlsx', 'Save all images results as');
    if ~isequal(save_file, 0)
        writetable(final_table, fullfile(save_path, save_file));
        fprintf('Results saved to %s\n', fullfile(save_path, save_file));
    end
end

function analyzeCompareArea(filenames, pathname, pixel2microm, squareMask)
    % analyzeCompareArea - Compares theoretical area to actual area of squares

    final_table = [];
    for file_idx = 1:length(filenames)
        filename = filenames{file_idx};
        
        fprintf('\nImage processing %d/%d : %s\n', file_idx, length(filenames), filename);
        img = imread(fullfile(pathname, filename));  
        theorical_real_square_area = input('What is the theorical square area (mm²) : ');
        
        % Calculate real area
        props = regionprops(squareMask, 'Area', 'Perimeter');
        airePixels = props.Area;
        aire_microns = airePixels * (pixel2microm^2);
        real_square_area = aire_microns / 1000000; % Converr in mm²
        fprintf('Square area: %.2f mm²\n', real_square_area);
        
        areas = [theorical_real_square_area, real_square_area];
        std_areas = std(areas);
        fprintf('Standard deviation area: %.2f\n', std_areas);

        figure;
        imshow(img);
        hold on;
        % Draw boudaries of inide square
        boundaries = bwboundaries(squareMask, 'holes');
        plot(boundaries{1}(:,2), boundaries{1}(:,1), 'r', 'LineWidth', 2);
        
        % Save on an Excel file
        fprintf('\nSave results on a file\n');
        image_name_row = table(string(filename), 'VariableNames', {'Image_Name'});
        results_table2 = table(theorical_real_square_area, round(real_square_area, 2), std_areas, ...
            'VariableNames', {'Theorical square area', 'Real Square Area', 'Standard deviation area'});
       
        max_rows = height(results_table2);
        image_name_row = [image_name_row; array2table(repmat({nan}, max_rows - height(image_name_row), 1), ...
            'VariableNames', image_name_row.Properties.VariableNames)];
        combined_table = [image_name_row, results_table2];
        final_table = [final_table; combined_table];
    end
    
    % Save
    [save_file, save_path] = uiputfile('*.xlsx', 'Save all images results as');
    if ~isequal(save_file, 0)
        writetable(final_table, fullfile(save_path, save_file));
        fprintf('Results saved to %s\n', fullfile(save_path, save_file));
    end
end