% ADD PUIS DELETE

clear all;
clc;
close all;

%____________________________________________________
%___________________FUNCTIONS________________________
%____________________________________________________

%Select a pore électionner un pore 
function [new_diameters,new_areas, new_centroids] = clicPore(binary_img, pixel2microm, DistanceKnow)
    new_diameters = [];
    new_areas = [];
    new_centroids = [];
    keepGoing = true;
    while keepGoing
        [x, y, button] = ginput(1);
        if button ~= 1
            keepGoing = false;
            break;
        end
    
        row = round(y);
        col = round(x);
        labeledImage = bwlabel(binary_img);
        poreLabel = labeledImage(row, col);
        
        if poreLabel == 0
            fprintf('No pore detected at this location. Please click inside a pore.\n');
            continue;
        end
        porePixels = labeledImage == poreLabel;
        props = regionprops(porePixels, 'MajorAxisLength', 'Centroid', 'Area');
        
        if isempty(props)
            continue;
        end
        
        diameter = props.MajorAxisLength;
        centroid = props.Centroid;
        area = props.Area;
        exact_diameter = diameter * pixel2microm ; % Method better to find pore diameter
        exact_area = area * pixel2microm ;
        new_diameters = [new_diameters, exact_diameter];
        new_areas = [new_areas, exact_area];
        new_centroids = [new_centroids; centroid];
        
        hold on;
        viscircles(centroid, diameter/2, 'EdgeColor', 'r'); 
        text(centroid(1), centroid(2), sprintf('%.f µm', exact_diameter), ...
             'Color', 'red', 'FontSize', 12, 'HorizontalAlignment', 'center');
        hold off;

        fprintf('Pore size: %.f µm   (%.f pixels)\n', exact_diameter, diameter); %±
        fprintf('Area size: %.f µm   (%.f pixels)\n', exact_area, area);
    end
    hold off;
end

%Deselect pore
function delete_indices = deselectPore(valid_diameters, valid_areas, valid_centroids, pixel2microm)
    fprintf('Click on the red circle to delete it (right-click to finish).\n');

    delete_indices = [];
    keepGoing = true;

    while keepGoing
        [x, y, button] = ginput(1);
        if button ~= 1
            keepGoing = false;
            break;
        end

        maxDistance = 25;
        closest_dist = inf;
        closest_idx = 0;
        
        for j = 1:size(valid_centroids, 1)
            if ismember(j, delete_indices)
                continue;
            end
            
            % Calculate distance of click from the center 
            dist_to_center = sqrt((valid_centroids(j,1) - x)^2 + (valid_centroids(j,2) - y)^2);
            % Calculate radius in pixels
            radius_pixels = valid_diameters(j) / (2 * pixel2microm);
            
            tolerance = 15;
            distance_to_border = abs(dist_to_center - radius_pixels);
            
            if (distance_to_border < tolerance || (dist_to_center > radius_pixels && dist_to_center < radius_pixels + tolerance)) && distance_to_border < closest_dist
                closest_dist = distance_to_border;
                closest_idx = j;
            end
        end
        
        if closest_idx > 0
            % Add to list of deleted pores
            delete_indices = [delete_indices, closest_idx];
            
            % Delete circle 
            allObjects = findall(gca, 'Type', 'line');
            textObjects = findall(gca, 'Type', 'text');
            deletedSomething = false;

            for i = 1:length(allObjects)
                obj = allObjects(i);
                xData = get(obj, 'XData');
                yData = get(obj, 'YData');
                
                if ~isempty(xData) && ~isempty(yData) && length(xData)  
                distances = sqrt((xData - x).^2 + (yData - y).^2);
                if min(distances) < maxDistance
                    delete(obj);
                end

                end
            end
            
            fprintf('Pore %d deleted (diameter: %.2f µm)\n', closest_idx, valid_diameters(closest_idx));
        else
            fprintf('No pore found near this location.\n');
        end
    end
    
    delete_indices = unique(delete_indices);
    delete_indices = sort(delete_indices, 'descend');
end

function [binary_separated] = separateConnectedPores(binary_img)

    % Calculate transform distance
    dist_transform = bwdist(~binary_img);
    dist_smooth = imgaussfilt(dist_transform, 4);
   
    % Calculate local maxima to identify potential pore centers
    max_local = imregionalmax(dist_smooth);
    
    %Markers for watershed
    markers = bwlabel(max_local);
    
    % Calculate the inverted distance transform for watershed segmentation
    % (The watershed algorithm segments based on valleys, not peaks)
    dist_neg = -dist_smooth;
    
    % Apply watershed
    watershed_result = watershed(dist_neg, 18);
    
    % Create a mask for the boundaries between objects
    watershed_lines = watershed_result == 0;
    
    % Apply the watershed lines to the original binary image
    binary_separated = binary_img & ~watershed_lines;
    
    % Remove small fragments generated by the watershed 
    binary_separated = bwareaopen(binary_separated, 50);
    
    % Reconstruct pores to make them more uniform
    se = strel('disk', 3);
    binary_separated = imclose(binary_separated, se);

end

function [new_centroid, new_diameter_pixels] = removeNestedCircles(centroid, diameter_pixels)
    indices_to_keep = true(1, size(centroid, 1));
   
    for i = 1:size(centroid, 1)
        % Check if this circle is inside another circle
        for j = 1:size(centroid, 1)
            if i == j
                continue; % Ignore the same circle
            end
            
            % Calculate the distance between the centers
            dist = norm(centroid(i,:) - centroid(j,:));
            total_radius = (diameter_pixels(i) + diameter_pixels(j)) / 2;
            
            % Check if one circle is entirely contained within another
            if dist + min(diameter_pixels(i), diameter_pixels(j))/2 <= max(diameter_pixels(i), diameter_pixels(j))/2 || ...    
                    (dist < total_radius)
                 
                    % Remove the smaller circle
                if diameter_pixels(i) < diameter_pixels(j)
                    indices_to_keep(i) = false;
                else
                    indices_to_keep(j) = false;filenames
                end
            end
        end
    end

    new_centroid = centroid(indices_to_keep, :);
    new_diameter_pixels = diameter_pixels(indices_to_keep);
end
     

%____________________________________________________
%_______READ AND PREPARE IMAGE TO BE ANALYSED________
%____________________________________________________

[filenames, pathname] = uigetfile({'*.jpg;*.jpg;*.png;*.tif;*.bmp','Image Files (*.jpg,*.png,*.tif,*.bmp)'},...
    'Select an image','MultiSelect', 'on');
if isequal(filenames, 0)
    disp('Selection deleted');
    return;
end

if ischar(filenames)
    filenames = {filenames};
end

fprintf('Opening image...\n');

for i = 1:length(filenames)
    filename = filenames{i};
    filename = fullfile(pathname, filenames{i});
    img = imread(filename); % Read image
    figure('Name', sprintf('Original image - %s',filenames{i}));
    imshow(img); % Plot image

    info = imfinfo(filename); % Informations about the picture
    %disp(info); % To show informations
    Igray = rgb2gray(img);  % Convert in grayscale
   
end

%__________________________________________
%___________________MENU __________________
%__________________________________________

user_choice = input(sprintf('Are the pores lighter - 1 or darker - 2 than the background ? (1/2) : '));

DistanceKnow = input('What is the scale (µm) : ');
lineLength = input('What is the length scale (pixels) : ');
fprintf('\nChoose the range for the pore size \n');
min_diameter_microm = input('What is the minimum pore diameter you want (µm) : ');
max_diameter_microm = input('What is the maximum pore diameter you want (µm) : ');
global pore_counter;
if ~exist('pore_counter', 'var') || isempty(pore_counter)
    pore_counter = 0;
end

% Beginning of loop for multiple images
% Loop to analyse each image 
all_diameters = []; % To stock all diameters
all_areas = [];
image_info = cell(length(filenames), 4); % To stock name and diameters 

first_image_name = erase(filenames{1}, '.png');
excel_filename = sprintf('%s_Results.xlsx', first_image_name);
[save_file, save_path] = uiputfile(excel_filename, 'Save results as');

if isequal(save_file, 0)
    disp('Save cancelled');
    return;
end

for file_idx = 1:length(filenames)
    filename = filenames{file_idx};
    fprintf('\nImage processing %d/%d : %s\n', file_idx, length(filenames), filename);
    I = imread(fullfile(pathname, filename));
    imshow(I)
    
    if user_choice == 1 % Light pores 
        I_gray = rgb2gray(I);
        I_enhanced = adapthisteq(I_gray);
        I_filtered = imgaussfilt(I_enhanced, 2);%2
        Im = imsharpen(I_filtered, 'Radius', 2, 'Amount', 4); % TO BE MODIFIED : 3 et 2
        grayImage = imadjust(Im, [0.5 0.8], []); 
        binary_img = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.45); % TO BE MODIFIED 
    else % Dark pores
        I_gray = rgb2gray(I);
        I_enhanced = adapthisteq(I_gray);
        I_filtered = imgaussfilt(I_enhanced, 1.5);
        Im = imsharpen(I_filtered, 'Radius', 10, 'Amount', 2); % TO BE MODIFIED 

        grayImage = imadjust(Im, [0.5 0.9], [0.2 1]);
        binary_img = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.9); % TO BE MODIFIED: 0.8
        binary_img = imcomplement(binary_img);
    end
      
    se = strel('disk', 4); % TO BE MODIFIED: 2
    binary_img = imopen(binary_img, se);
    binary_img = imclose(binary_img, se);
    binary_img = bwareafilt(binary_img, [200, inf]);
    binary_img = bwareaopen(binary_img, 200);
    binary_img = medfilt2(binary_img, [3 3]);
    %imshow(binary_img); % To see image

    figure('Name', sprintf('Pore analysis - %s', filename));
    imshow(I);
    fprintf('Scale detected 1...\n');

%___________________________________________________________
%_______________AUTOMATIC PORE SIZE DETECTION_______________
%___________________________________________________________

        
fprintf('Automatic detection processing...\n')
fprintf('Wait...\n')
pixel2microm = DistanceKnow / lineLength;
min_diameter_pixels = min_diameter_microm / pixel2microm;
max_diameter_pixels = max_diameter_microm / pixel2microm;
min_radius_pixels = round(min_diameter_pixels / 2);
max_radius_pixels = round(max_diameter_pixels / 2);
            
if user_choice == 1 
    [centroid,radius_pixels] = imfindcircles(Im,[min_radius_pixels max_radius_pixels], ObjectPolarity="bright", Sensitivity=0.91,EdgeThreshold=0.085); % TO BE MODIFIED : sensitivity 0.85
else 
    %[centroid,radius_pixels] = imfindcircles(Im,[min_radius_pixels max_radius_pixels],ObjectPolarity="dark", Sensitivity=0.85, EdgeThreshold=0.085); % TO BE MODIFIED : sensitivity 0.95 or 0.9 EdgeThreshold 0.08 or 0.085
    [centroid,radius_pixels] = imfindcircles(Im,[min_radius_pixels max_radius_pixels],ObjectPolarity="dark", Sensitivity=0.9, EdgeThreshold=0.085); 

end 
diameter_pixels = radius_pixels*2;
diameter_microm = diameter_pixels*pixel2microm;
area_microm = (pi*(diameter_microm.*diameter_microm)) / 4;
valid_diameters = diameter_microm;
valid_diameters = valid_diameters';
valid_areas = area_microm;
valid_centroids = centroid;

% Calculate the indices of the diameters to retain
indices_a_conserver = ismember(diameter_microm, diameter_pixels*pixel2microm);
% Filtered diameters of the picture 
diameter_microm_filtered = diameter_microm(indices_a_conserver);
area_microm_filtered = area_microm(indices_a_conserver);

% MODIFICATION: Stocker diameters of the picture
image_info{file_idx, 1} = filename;
image_info{file_idx, 2} = diameter_microm_filtered;
image_info{file_idx, 3} = area_microm_filtered;

% Add filtered diameters in the global table
all_diameters = [all_diameters; diameter_microm_filtered];
all_areas = [all_areas; area_microm_filtered];

circle_handles = zeros(length(diameter_pixels), 1);
text_handles = zeros(1, length(diameter_pixels));
for k = 1:length(diameter_microm_filtered) 
    circle_handles(k) = viscircles(centroid(k,:), diameter_pixels(k)/2, 'EdgeColor', 'r','LineWidth', 1);
    % If you want to name each particules by a number
    %text_handles(k) = text(centroid(k,1), centroid(k,2), num2str(pore_counter+k), 'Color', 'red','FontWeight','bold', 'FontSize',12);  
end

pore_counter = pore_counter + length(diameter_microm_filtered);

 fprintf('Number of pores detected in this picture : %d \n', length(diameter_microm_filtered));
 fprintf('Pores detection :\n');
 fprintf('Pore number #\tDiameter (µm)\n');
 for k = 1:length(diameter_microm_filtered)
    fprintf('%d\t%.2f\n', pore_counter - length(diameter_microm_filtered) + k, diameter_microm_filtered(k) );
 end

%_______________________________________________
%_______________MORE MODIFICATION_______________
%_______________________________________________

modif_detection = input(sprintf('Do you want to add (1) or delete (2) some detected particules or do nothing (3) ? (1/2/3) : '));
fprintf('Number of pores before modification : %d \n', length(diameter_microm_filtered));

% Locals variables for modifications
current_diameters = diameter_microm_filtered;
current_areas = area_microm_filtered;

if modif_detection == 1 % Add pores
    % Call the function clicPore and take new diameters 
    [new_diameters,new_areas, new_centroids] = clicPore(binary_img, pixel2microm, DistanceKnow);
    
    % Add new pores on existing table
    valid_diameters = [valid_diameters, new_diameters];
    %valid_areas = [valid_areas; new_areas];
    valid_centroids = [valid_centroids; new_centroids];
    
    % Add to local table
    new_diameters = new_diameters(:);
    current_diameters = [current_diameters; new_diameters];
    new_areas = new_areas(:);
    current_areas = [current_areas; new_areas];
    
    % Add to global table
    all_diameters = [all_diameters; new_diameters];
    all_areas = [all_areas; new_areas];
    
    modif_again = input(sprintf('Do you want to delete (1) or do nothing (2) ? (1/2) : '));

    if modif_again == 1 
        delete_indices = deselectPore(valid_diameters, valid_areas, valid_centroids,pixel2microm);
        valid_diameters(delete_indices) = []; % Delete elements 
        valid_areas(delete_indices, :) = [];
        valid_centroids(delete_indices, :) = [];
        
        % Update tables
        indices_to_remove = delete_indices(delete_indices <= length(current_diameters));
        current_diameters(indices_to_remove) = [];
        current_areas(indices_to_remove) = [];
        
        % Update all_diameters
        all_diameters = [];
        all_areas = [];
        for i = 1:length(filenames)
            if i < file_idx
                all_diameters = [all_diameters; image_info{i, 2}];
                all_areas = [all_areas; image_info{i, 3}];
            elseif i == file_idx
                all_diameters = [all_diameters; current_diameters];
                all_areas = [all_areas; current_areas];
            end
        end
    end
    
elseif modif_detection == 2 % Delete pores
    % Call deselectPore 
    delete_indices = deselectPore(valid_diameters, valid_areas,valid_centroids,pixel2microm);
    valid_diameters(delete_indices) = [] ;
    valid_areas(delete_indices, :) = [];
    valid_centroids(delete_indices, :) = [];
    
    % Update local tables
    indices_to_remove = delete_indices(delete_indices <= length(current_diameters));
    current_diameters(indices_to_remove) = [];
    current_areas(indices_to_remove) = [];
    
    % Update all_diameters
    all_diameters = [];
    all_areas = [];
    for i = 1:length(filenames)
        if i < file_idx
            all_diameters = [all_diameters; image_info{i, 2}];
            all_areas = [all_areas; image_info{i, 3}];
        elseif i == file_idx
            all_diameters = [all_diameters; current_diameters];
            all_areas = [all_areas; current_areas];
        end
    end
    
    modif_again = input(sprintf('Do you want to add (1) or do nothing (2) ? (1/2) : '));
    
    if modif_again == 1
       [new_diameters,new_areas, new_centroids] = clicPore(binary_img, pixel2microm, DistanceKnow);
       valid_diameters = [valid_diameters, new_diameters];
       valid_areas = [valid_areas; new_areas'];
       valid_centroids = [valid_centroids; new_centroids]; 
       
       % Updates tables
       current_diameters = [current_diameters; new_diameters'];
       all_diameters = [all_diameters; new_diameters'];

       current_areas = [current_areas; new_areas'];
       all_areas = [all_areas; new_areas'];
    end
    fprintf('Number of pores after modification : %d \n', length(current_diameters));
    
    % Update info about image
    image_info{file_idx, 2} = current_diameters;
    image_info{file_idx, 3} = current_areas;

else % Continue the code as usual
    fprintf('No modification made.\n');
    image_info{file_idx, 2} = current_diameters;
    image_info{file_idx, 3} = current_areas;
    
    % Save image detection
    image_name_clean = erase(filenames{file_idx}, '.png');
    detection_filename = fullfile(save_path, sprintf('%s_Detection.png', image_name_clean));
    saveas(gcf, detection_filename);
    fprintf('Detection saved to %s\n', detection_filename);
    % Stocker image detection to be Add on Excel 
    image_info{file_idx, 4} = detection_filename; 
end
    fprintf('\nTotal number of detected pores : %d\n', length(all_diameters));

end

%____________________________________________
%_______________STATISTIQUES_________________
%____________________________________________

mean_diameter = mean(all_diameters);
std_diameter = std(all_diameters);
median_diameter = median(all_diameters);
cv_diameter = (std_diameter/mean_diameter) * 100 ;

[ Height , Width, ~] = size(I);
fprintf('Taille : %d x %d pixels\n', Width,  Height );
total_image_surface = Width *  Height ;
total_image_surface_microm = total_image_surface *pixel2microm * pixel2microm ;

black_threshold = 20; % Pixels < 20 consider black
mask_analyze = Igray > black_threshold;
usable_area = sum(mask_analyze(:));
usable_area_microm = usable_area*pixel2microm*pixel2microm;
surface_totale = numel(mask_analyze);
surface_totale_microm = surface_totale*pixel2microm*pixel2microm;
density = ((length(all_diameters)) / (usable_area_microm));

fprintf('\nPore Analysis Results (All Images):\n');
fprintf('Number of valid pores detected: %d\n', length(all_diameters));
fprintf('Minimum pore size : %.2f µm\n', min(all_diameters));
fprintf('Maximum pore size : %.2f µm\n', max(all_diameters));
fprintf('Mean pore diameter: %.2f µm\n', mean_diameter);
fprintf('Median pore diameter: %.2f µm\n', median_diameter);
fprintf('Standard deviation: %.2f µm\n', std_diameter);
fprintf('Coefficient of variation: %.2f%%\n', cv_diameter);
porosity = (sum(all_areas) / usable_area_microm) * 100;

fprintf('Porosity: %.1f%%\n', porosity);
fprintf('Density: %f pores/mm²\n', density*1000000);


final_table = [];
image_name_row = table(string(filename), 'VariableNames', {'Image_Name'});
Pore_numbers = (1:length(all_diameters))'; 
results_table1 = table(Pore_numbers,round(all_diameters,2),'VariableNames', {'Pore_number','Diameters_microm'});
results_table3 = table(round(all_areas,2),'VariableNames', {'Areas_microm'});
results_table2 = table(round(min(all_diameters),2), round(max(all_diameters),2),round(mean_diameter,2),round(median_diameter,2) ,round(std_diameter,2),round(cv_diameter,2), 'VariableNames', {'Minimum diameter','Maximum diameter','Mean','Median','Standard deviation','C.V'});
results_table4 = table(round(porosity,2), round(density*1000000,2),'VariableNames', {'Porosity (%)', 'Density pores/mm²'});
results_table5 = table(min_diameter_microm, max_diameter_microm,'VariableNames', {'Chosen mininum diameter', 'Chosen maximum diameter'});

% Find the maximul size of rows 
max_rows = max(height(results_table1), height(results_table2));

results_table1 = [results_table1; array2table(nan(max_rows - height(results_table1), width(results_table1)), ...
    'VariableNames', results_table1.Properties.VariableNames)];

results_table2 = [results_table2; array2table(nan(max_rows - height(results_table2), width(results_table2)), ...
    'VariableNames', results_table2.Properties.VariableNames)];
results_table3 = [results_table3; array2table(nan(max_rows - height(results_table3), width(results_table3)), ...
    'VariableNames', results_table3.Properties.VariableNames)];
results_table4 = [results_table4; array2table(nan(max_rows - height(results_table4), width(results_table4)), ...
    'VariableNames', results_table4.Properties.VariableNames)];
results_table5 = [results_table5; array2table(nan(max_rows - height(results_table5), width(results_table5)), ...
    'VariableNames', results_table5.Properties.VariableNames)];

image_name_row = [image_name_row; array2table(repmat({nan}, max_rows - height(image_name_row), 1), ...
    'VariableNames', image_name_row.Properties.VariableNames)];

% Combine tables
combined_table = [image_name_row,results_table1, results_table3, results_table2, results_table4,results_table5];
final_table = [final_table; combined_table];

% Exporte results
if ~isequal(save_file, 0)
    writetable(final_table, fullfile(save_path, save_file));
    %writetable(results_table);
    fprintf('Results saved to %s\n', fullfile(save_path, save_file));
end

%__________________________________________
%_______________HISTOGRAM__________________
%__________________________________________

all_diameters = [];
all_areas = [];
for img_idx = 1:length(filenames)
    all_diameters = [all_diameters; image_info{img_idx, 2}];
    all_areas = [all_areas; image_info{img_idx, 3}];
end

fprintf('Total number of pores : %d\n', length(all_diameters));
figure('Name', 'Pore Size Distribution');
h = histogram(all_diameters,'BinWidth', 5,'Normalization','probability');
xlabel('Diameter (µm)');
ylabel('Frequency (%)'); 
ax = gca;
yTicks = ax.YTick;
ax.YTickLabel = arrayfun(@(x) sprintf('%.0f%', x*100), yTicks, 'UniformOutput', false);
grid on;
xlim([0 inf]);
currentYLim = ylim;
ylim([currentYLim(1), currentYLim(2) ]);
set(gca, 'XMinorTick', 'on', 'YMinorTick', 'on');
% Save histogramm
histogram_filename = fullfile(save_path, 'Histogram.png');
saveas(gcf, histogram_filename);
fprintf('Histogram saved to %s\n', histogram_filename);

%__________________________________________
%_________________GRAPHIC__________________
%__________________________________________

figure('Name', 'Distribution Curve');
binEdges = h.BinEdges;
binCounts = h.Values * 100; 
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
binCounts = [0, binCounts];
binCenters = [0, binCenters];
x_dense = linspace(min(binCenters), max(binCenters), 400);
y_smooth = pchip(binCenters, binCounts, x_dense);
y_smooth(x_dense < min(binCenters(binCounts > 0))) = 0;
y_smooth = max(y_smooth, 0);
window_size = 5; 
kernel = ones(window_size, 1) / window_size;
y_smooth = filter(kernel, 1, y_smooth);
plot(x_dense, y_smooth);
hold on
xlim([0 inf]);
xlabel('Pore Diameter (µm)');
currentYLim = ylim; 
ylim([currentYLim(1), currentYLim(2) + 0.05]);
ylabel('Frequency (%)'); 
grid on;
set(gca, 'FontSize', 11);
set(gca, 'XMinorTick', 'on', 'YMinorTick', 'on');

distribution_filename = fullfile(save_path, 'DistributionCurve.png');
saveas(gcf, distribution_filename);
fprintf('Distribution Curve saved to %s\n', distribution_filename);

excelApp = actxserver('Excel.Application');
excelApp.Visible = true;
workbook = excelApp.Workbooks.Open(fullfile(save_path, save_file));
sheet = workbook.Sheets.Item(1);
sheet.Shapes.AddPicture(histogram_filename, 0, 1, 300, 50, 300, 200); 
sheet.Shapes.AddPicture(distribution_filename, 0, 1, 650, 50, 300, 200);

start_x = 1000;
for img_idx = 1:length(filenames)
    if length(image_info) >= img_idx && length(image_info{img_idx}) >= 4
        detection_file = image_info{img_idx, 4};
        if exist(detection_file, 'file')
            sheet.Shapes.AddPicture(detection_file, 0, 1, start_x, 50 + (img_idx-1)*260, 300, 250);
        end
    end
end
workbook.Save;
fprintf('\nSave results on a file\n');

  


