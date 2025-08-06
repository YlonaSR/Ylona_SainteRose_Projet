%CODE WORKS - ANALYSE DE SLURRIES 

%Clean Workspace and the command window
clear all;
clc;
close all;


%____________________________________________________
%___________________FUNCTIONS________________________
%____________________________________________________

% Segment slurries
function [binary_img] = segmentSlurries(grayImage)
    % Apply gaussien filter
    grayImage = imgaussfilt(grayImage, 3); %ou 6 ou 3
    %grayImage = medfilt2(grayImage, [3 3]);
    %grayImage = adapthisteq(grayImage);
    
    grayImage = adapthisteq(grayImage, 'ClipLimit', 0.02); % Improve contrast and control intensity 
    %grayImage = medfilt2(grayImage, [1 1]); % elimnate noise, doesn't work
    grayImage = imsharpen(grayImage, 'Radius', 2, 'Amount', 1); % Improve sharpness

    thresh = multithresh(grayImage, 3); 
    disp(thresh);
    binary_img = grayImage > thresh(2);
    %BIG PORES
    %binary_img = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.5, 'ForegroundPolarity', 'dark'); %TO BE MODIFIED : 0.6 
    % FOR FITC IMAGES
    %binary_img = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.99, 'ForegroundPolarity', 'bright');
    % FOR ALL PORES 
    binary_img = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.75,'ForegroundPolarity', 'dark'); %TO ME MODIFEID : 0.6 / 0.85 / 0.63 / 0.75
 
    % Morphological to improve segmentation
    se1 = strel('disk', 1); % TO BE MODIFIED 13 / 7 for big pores / 1 for all
    %For FITC se1 = strel('disk', 1);
    binary_img = imopen(binary_img, se1);

    % Delete very small pores that can be noise
    binary_img = bwareaopen(binary_img, 10); % BIG PORES : 6, All 10
    %For FITC binary_img = bwpropfilt(binary_img, 'Area', [5, Inf]); % Garde uniquement les objets plus grands que 5 pixels
    %binary_img = imcomplement (binary_img);
    imshow(binary_img);
end

% Calculate slurry statistics 
function [stats_table] = calculateSlurryStats(props, pixel2microm)
    % Extract properties
    areas_pixels = [props.Area];
    %perimeters_pixels = [props.Perimeter];
    diameters_pixels = [props.EquivDiameter];
    
    % Convert
    areas_microm2 = areas_pixels * (pixel2microm^2);
    %perimeters_microm = perimeters_pixels * pixel2microm;
    diameters_microm = diameters_pixels * pixel2microm;
    
    % Create a table
    stats_table = table(areas_microm2', diameters_microm',  ...
                      'VariableNames', {'Area_microm2', 'Diameter_microm'});
end

%To add slurry
function [new_areas, new_perimeters, new_centroids, new_diameters] = clicSlurry(binary_img, pixel2microm)
    new_areas = [];
    new_perimeters = [];
    new_centroids = [];
    new_diameters = [];
    keepGoing = true;
    
    while keepGoing
        [x, y, button] = ginput(1);
        
        if button ~= 3
                title('Drawing mode: click to set points, double-click to finish.');
                
                % Collect points to draw 
                drawing_points = [];
                drawing = true;
                
                while drawing
                    [xd, yd, btn] = ginput(1);
                    
                    if btn == 1 % Left click to add a point
                        drawing_points = [drawing_points; round(xd) round(yd)];
                        
                        % Display point and link it to the previous one
                        hold on;
                        plot(xd, yd, 'b.', 'MarkerSize', 10);
                        if size(drawing_points, 1) > 1
                            plot([drawing_points(end-1,1), drawing_points(end,1)], ...
                                 [drawing_points(end-1,2), drawing_points(end,2)], 'r-');
                        end
                        hold off;
                    elseif btn == 3 % Right click to end
                        % Close the draw 
                        if size(drawing_points, 1) > 2
                            hold on;
                            %plot([drawing_points(end,1), drawing_points(1,1)], ...
                                 %[drawing_points(end,2), drawing_points(1,2)], 'r-');
                            hold off;
                            
                            % Ceate a mask 
                            mask = poly2mask(drawing_points(:,1), drawing_points(:,2), size(binary_img,1), size(binary_img,2));
                            % Calculate properties
                            props = regionprops(mask, 'Area', 'Perimeter', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
                            
                            if ~isempty(props)
                                props = props(1);
                                diameter_pixels = props.EquivDiameter;
                                area_pixels =  prosp.Area;
                                centroid = props.Centroid;
                                
                                diameter_microm = diameter_pixels * pixel2microm;
                                area_microm = area_pixels * pixel2microm;
                                
                                new_perimeters = [new_perimeters, perimeter_microm];
                                new_diameters = [new_diameters, diameter_microm];
                                
                                % Plot info
                                hold on;
                                
                                % Tracer les axes pour visualiser le diamètre
                                theta = linspace(0, 2*pi, 50);
                                a = props.MajorAxisLength/2;
                                b = props.MinorAxisLength/2;
                                phi = deg2rad(0); 
                                if isfield(props, 'Orientation')
                                    phi = deg2rad(props.Orientation);
                                end
                                hold off;
                                fprintf('  Diameter: %.1f µm (%.1f pixels)\n', diameter_microm, diameter_pixels);
                            end
                        end
                        drawing = false;
                    end
                end
                
                continue;
            else
                keepGoing = false;
                break;
        end
        
        % Normal treatment 
        row = round(y);
        col = round(x);
        labeledImage = bwlabel(binary_img);
        slurryLabel = labeledImage(row, col);
        
        if slurryLabel == 0
            fprintf('No particle detected at this location. Please click inside a particle.\n');
            continue;
        end
        
        slurryPixels = labeledImage == slurryLabel;
        props = regionprops(slurryPixels, 'Area','EquivDiameter', 'Perimeter', 'Centroid', 'MajorAxisLength', 'MinorAxisLength');
        
        if isempty(props)
            continue;
        end
        
        % Make sure that props is a single element, not an array.
        props = props(1);
        
        diameter_pixels = props.EquivDiameter; 
        area_pixels = props.Area;
        centroid = props.Centroid;
       
        diameter_microm = diameter_pixels * pixel2microm;
        % Conversion en micromètres carrés et micromètres
        %area_microm2 = area_pixels * (pixel2microm^2);
        perimeter_microm = perimeter_pixels * pixel2microm;
        
        % Stock new data
        %new_areas = [new_areas, area_microm2];
        %new_perimeters = [new_perimeters, perimeter_microm];
        new_centroids = [new_centroids; centroid];
        new_diameters = [new_diameters, diameter_microm];
        
        % Visualize particule
        hold on;
        boundary = bwboundaries(slurryPixels);
        plot(boundary{1}(:,2), boundary{1}(:,1), 'b', 'Linelargeur', 2);
        
        theta = linspace(0, 2*pi, 50);
        a = props.MajorAxisLength/2;
        b = props.MinorAxisLength/2;
        phi = deg2rad(0); 
        if isfield(props, 'Orientation')
            phi = deg2rad(props.Orientation);
        end
        
        x = centroid(1) + a*cos(theta)*cos(phi) - b*sin(theta)*sin(phi);
        y = centroid(2) + a*cos(theta)*sin(phi) + b*sin(theta)*cos(phi);
        plot(x, y, 'b-', 'Linelargeur', 1);
        
        line_x = centroid(1) + [-a*cos(phi), a*cos(phi)];
        line_y = centroid(2) + [-a*sin(phi), a*sin(phi)];
        plot(line_x, line_y, 'r-', 'Linelargeur', 1.5);
        text(centroid(1), centroid(2), sprintf('P=%.1f µm', diameter_microm), 'Color', 'blue', 'FontSize', 10, 'HorizontalAlignment', 'center');
        hold off;
        
        fprintf('Particule selected:\n');
        %fprintf('  Aire: %.0f µm² (%.0f pixels)\n', area_microm2, area_pixels);
        %fprintf('  Périmètre: %.1f µm (%.1f pixels)\n', perimeter_microm, perimeter_pixels);
        fprintf('  Diameter: %.1f µm (%.1f pixels)\n', diameter_microm, diameter_pixels);
    end
    
    hold off;
end


function delete_indices = deselectedPore(valid_diameters, valid_centroids)
    fprintf('Click on the red circle to delete it (right-click to finish).\n');

    delete_indices = [];
    keepGoing = true;
    %closest_idx = 0;

    while keepGoing
        [x, y, button] = ginput(1);
        if button ~= 1
            keepGoing = false;
            break;
        end

        maxDistance = 5;
        allObjects = findall(gca, 'Type', 'line'); 
        textObjects = findall(gca, 'Type', 'text');
        % Parcourir tous les objets et supprimer ceux qui sont proches du clic
        deletedSomething = false;
        
        for i = 1:length(allObjects)
            obj = allObjects(i);
            xData = get(obj, 'XData');
            yData = get(obj, 'YData');
                
            if ~isempty(xData) && ~isempty(yData)
                distances = sqrt((xData - x).^2 + (yData - y).^2);
                if min(distances) < maxDistance
                    delete(obj);
                    deletedSomething = true;
                    % Find pore 
                    closest_dist = inf;
                    closest_idx = 0;
                    
                    for j = 1:size(valid_centroids,1)
                        dist_to_centroid = sqrt((valid_centroids(j,1) - x)^2 + (valid_centroids(j,2) - y)^2);
                        if dist_to_centroid < closest_dist
                            closest_dist = dist_to_centroid;
                            closest_idx = j;
                            %delete_indices = [delete_indices, j];
                            
                        end
                    end
                 %if closest_dist < maxDistance %&& closest_idx > 0
                    delete_indices = [delete_indices, closest_idx];
                %end
                end
            end
        end

        for i = 1:length(textObjects)
            text_obj = textObjects(i);
            position = get(text_obj, 'Position');
            
            % Check if this text is near the centroid we want to delete
            dist_to_text = sqrt((position(1) - valid_centroids(closest_idx,1))^2 + (position(2) - valid_centroids(closest_idx,2))^2);
            if dist_to_text < maxDistance
                delete(text_obj);
            end
        end       

        if deletedSomething
            fprintf('Elements deleted.\n');
        else
            fprintf('No element found next to this point.\n');
        end
    end
     delete_indices = unique(delete_indices);
     delete_indices = sort(delete_indices);
end

%____________________________________________________
%_______READ AND PREPARE IMAGE TO BE ANALYSED________
%____________________________________________________

% SELECT FILES
[filenames, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.tif;*.bmp','Image Files (*.jpg,*.png,*.tif,*.bmp)'},...
    'Select images','MultiSelect', 'on');

if isequal(filenames, 0)
    disp('Selection canceled');
    return;
end

if ischar(filenames)
    filenames = {filenames};
end

% Analyze each picture

for k = 1:length(filenames)
    fprintf('\n=== Traitement de l''image %d/%d ===\n', k, length(filenames));
    
    fullPath = fullfile(pathname, filenames{k});
    info = imfinfo(fullPath);
    %disp(info);
    
    % Read and display image
    imageData = imread(fullPath);
    figure;
    imshow(imageData);
    title(filenames{k}, 'Interpreter', 'none');

    fprintf('Setting scale...\n');
    scale_value = input('What is the scale of all images (µm) : ');
    scale_length = input('What is the lenght of the scale (pixels) : ');
    pixel2microm = scale_value / scale_length;
    %pixel2microm = 2.13;
    fprintf('Scale: 1 pixel = %.3f µm\n', pixel2microm);
    diameter_min = input('Enter the minimum diameter to detect" (µm) : ');
    diameter_max = input('Enter the maximum diameter to detect (µm) : ');
    
    % Calculate surface
    [hauteur, largeur, ~] = size(imageData);
    total_image_surface = (hauteur * largeur)*pixel2microm*pixel2microm;
    
    % INTERACTIV MASK MULTIPLE
    masque_interactif = true(hauteur, largeur);
    
    % Instructions pour l'utilisateur
    fprintf('\n=== Select zones to be masked ===\n');
    fprintf('Instructions :\n');
    fprintf('- Click and drag to select an area to mask.\n');
    fprintf('- Repeat as many times as needed.\n');
    fprintf('- Right-click to complete the selection.\n\n');
    
    zone_count = 0;
    continuer = true;
    
    while continuer
        try
            fprintf('Select area %d to mask (or right-click to finish).)\n', zone_count + 1);
            
            % Use ginput to detect the type of click.
            [x, y, button] = ginput(1);
            
            % Verify if its a right click (button = 3)
            if button == 3
                fprintf('Selection end. %d zone(s) selected.\n', zone_count);
                continuer = false;
                continue;
            end
            
            % If its a left click
            if button == 1
                rect = getrect;
                if rect(3) > 0 && rect(4) > 0
                    zone_count = zone_count + 1;
                    x_rect = max(1, round(rect(1))); 
                    y_rect = max(1, round(rect(2))); 
                    w_rect = min(round(rect(3)), largeur - x_rect + 1); 
                    h_rect = min(round(rect(4)), hauteur - y_rect + 1);
                    
                    if w_rect > 0 && h_rect > 0
                        masque_interactif(y_rect:y_rect+h_rect-1, x_rect:x_rect+w_rect-1) = false;
                        fprintf('Area %d successfully masked.(%.0f x %.0f pixels)\n', zone_count, w_rect, h_rect);
                    else
                        fprintf('Warning: Area %d is invalid and has been ignored.\n', zone_count);
                        zone_count = zone_count - 1;
                    end
                else
                    fprintf('Selection cancel\n');
                end
            end
            
        catch ME
            % If error or interruption like Ctrl+C
            fprintf('Selection interrupted. %d area(s) selected..\n', zone_count);
            continuer = false;
        end
    end
    
    if zone_count == 0
        fprintf('No zone selected. Use the entire image.\n');
        masque_interactif = true(hauteur, largeur);
    end
    
    % USE INTERACTIVE MASK
    img_masked_interactive = imageData;
    
    % APPLY MASK
    for canal = 1:size(img_masked_interactive, 3)
        temp = img_masked_interactive(:, :, canal);
        temp(~masque_interactif) = 0;  % Selected zones in black
        img_masked_interactive(:, :, canal) = temp;
    end
    
    % Display image with mask 
    figure;
    imshow(img_masked_interactive);
    title(sprintf('Image with %d zone(s) masked', zone_count));
        
    %% CALCULATE SURFACES
    % Convert in gray if picture in color
    if size(img_masked_interactive, 3) == 3
        I_gray = rgb2gray(img_masked_interactive);
    else
        I_gray = img_masked_interactive;
    end
    
    % Count black pixels (valeur = 0)
    black_pixels = sum(I_gray(:) == 0);
    masked_surface = black_pixels;
    masked_surface_microm = masked_surface * pixel2microm * pixel2microm;
    
    % Calculate real surface (total - masked)
    analyze_surface = total_image_surface - masked_surface_microm;
end

fprintf('\n=== Processing finished ===\n');
fprintf('Ready to process %d image(s)\nWait...\n', length(filenames));

%__________________________________________
%___________________MENU __________________
%__________________________________________

% To stock variables
all_areas = [];
all_diameters = [];
image_info = cell(length(filenames), 3); % Name, area, perimeter

% Analyze each image
for file_idx = 1:length(filenames)
    filename = filenames{file_idx};
    fprintf('\nProcessing image %d/%d: %s\n', file_idx, length(filenames), filename);
    
    % Read image
    I = imread(fullfile(pathname, filename));
    
    % Convert in gray
    if size(I, 3) == 3
        grayImage = rgb2gray(I);
    else
        grayImage = I;
    end
    % Display original image
    figure('Name', sprintf('Original image - %s', filename));
    imshow(I);
    
    % Image segmentation
    binary_img = segmentSlurries(I_gray);
    fprintf('Wait....\n')

    CC = bwconncomp(binary_img);
    props = regionprops(CC, 'Area', 'Perimeter', 'Centroid', 'BoundingBox', 'Eccentricity', 'EquivDiameter');
    all_diameters_pixels = [props.EquivDiameter];
    all_diameters_microm = all_diameters_pixels * pixel2microm;
    valid_indices = find(all_diameters_microm >= diameter_min & all_diameters_microm <= diameter_max & [props.Area] >= 0);
    valid_props = props(valid_indices);
    
    % Calculate statistics of slurries 
    if ~isempty(valid_props)
        stats_table = calculateSlurryStats(valid_props, pixel2microm);
        
        % Display results 
        figure('Name', sprintf('Automatic Analysis - %s', filename));
        imshow(I);
        hold on;
        
        % Draw boundaries 
        for i = 1:length(valid_props)
            % Get slurries pixel
            temp_binary = false(size(binary_img));
            pixels = CC.PixelIdxList{valid_indices(i)};
            temp_binary(pixels) = true;
            
            % Find and draw boundaries
            boundary = bwboundaries(temp_binary);
            if ~isempty(boundary)
                plot(boundary{1}(:,2), boundary{1}(:,1), 'b');
                centroid = valid_props(i).Centroid;
                for k = 1:length(centroid)
                end
            end
        end
        hold off;
        for k = 1:length(valid_props)
            fprintf('%d\t%.2f\n', k,  stats_table.Diameter_microm(k));
        end

        % Add data to global table 
        areas_this_image = stats_table.Area_microm2;
        diameters_this_image = stats_table.Diameter_microm;

        % Save informations 
        image_info{file_idx, 1} = filename;
        image_info{file_idx, 2} = areas_this_image;
        image_info{file_idx, 3} = diameters_this_image;

         type_modif = input(sprintf(['Do you want to add pores (Press 1) or delete (Press 2) or do nothing (Press 3) ?\n' ...
        'Press 1, 2 or 3 : ']));

 if type_modif == 1 %Add element
     [~, diametres, ~, ~] = clicSlurry(I, pixel2microm);
     [~, areaz, ~, ~] = clicSlurry(I, pixel2microm);
   
     % Stock 
     all_areas = [all_areas, areas_this_image];
     all_diameters = [all_diameters, diameters_this_image];
    
     % Save
     image_info{file_idx, 1} = filename;
     image_info{file_idx, 2} = areas_this_image;
     image_info{file_idx, 3} = diameters_this_image;

    if ~isempty(diametres)
        
        all_diameters = [all_diameters; diametres]';
        all_areas = [all_areas; areaz]';
        
        % Update informations 
        image_info{file_idx, 3} = [diameters_this_image; diametres'];
        image_info{file_idx, 2} = [areas_this_image; areaz'];
        
        fprintf('Added %d new pores manually\n', length(diametres));
    end

     more_modif = input(sprintf(['Do you want to delete (Press 1) or do nothing (Press 2) ?\n' ...
        'Press 1 or 2 : ']));
    if more_modif == 1 
        valid_centroids = zeros(length(valid_props), 2);
        valid_diameters = diameters_this_image;
        valid_areas = areas_this_image;

        for i = 1:length(valid_props)
            valid_centroids(i,:) = valid_props(i).Centroid;
        end

        delete_indices = deselectedPore(valid_diameters, valid_centroids);
    
        % Delete elements 
        diameters_this_image(delete_indices) = [];
        areas_this_image(delete_indices) = [];
        
        % Update (no effect to all_diameters)
        valid_diameters(delete_indices) = [];
        valid_centroids(delete_indices, :) = [];
        valid_areas(delete_indices, :) = [];
        
        % Update info
        image_info{file_idx, 3} = diameters_this_image;
        image_info{file_idx, 2} = areas_this_image;

        %all_diameters = [];
        all_diameters(delete_indices) = [];
        all_areas(delete_indices) = [];
        %all_diameters = [all_diameters; diametres];
        %all_diameters = [all_diameters; diameters_this_image];
        fprintf('Removed %d slurries\n', length(delete_indices));
    end

 elseif type_modif == 2 %Delete element

        valid_centroids = zeros(length(valid_props), 2);
        valid_diameters = diameters_this_image;
        valid_areas = areas_this_image;
        for i = 1:length(valid_props)
            valid_centroids(i,:) = valid_props(i).Centroid;
        end

        delete_indices = deselectedPore(valid_diameters, valid_centroids);
        diameters_this_image(delete_indices) = [];
        areas_this_image(delete_indices) = [];

        valid_diameters(delete_indices) = [];
        valid_centroids(delete_indices, :) = [];
        valid_areas(delete_indices, :) = [];
        
        image_info{file_idx, 3} = diameters_this_image;
        image_info{file_idx, 2} = areas_this_image;

        fprintf('Removed %d slurries\n', length(delete_indices));
        
        all_diameters = [all_diameters; diameters_this_image];
        all_areas = [all_areas; areas_this_image];

       %image_info{file_idx, 3} = diameters_this_image;
       %image_info{file_idx, 2} = areas_this_image;
        
        more_modif = input(sprintf(['Do you want to add (Press 1) or do nothing (Press 2) ?\n' ...
        'Press 1 or 2 : ']));
        if more_modif == 1 
            %[areas_this_image, diameters_this_image, ~] = clicSlurry(binary_img, pixel2microm);
             %all_diameters = [all_diameters; diameters_this_image];
             [~, diametres, ~, ~] = clicSlurry(I, pixel2microm);
             %[aires, diametres, centroides, diametres] = clicSlurry(I, pixel2microm);
            
            % Stock
            %all_areas = [all_areas, areas_this_image];
            %all_diameters = [all_diameters, diameters_this_image];
            
            % Sauvegarder les informations de cette image
            image_info{file_idx, 1} = filename;
            image_info{file_idx, 2} = areas_this_image;
            image_info{file_idx, 3} = diameters_this_image;
        
            if ~isempty(diametres)
                all_diameters = [all_diameters; diametres'];
                
                % Update info
                image_info{file_idx, 3} = [diameters_this_image; diametres'];
                image_info{file_idx, 2} = [areas_this_image; areaz'];
                
                fprintf('Added %d new pores manually\n', length(diametres));
            end

         end
    
         else
            size(all_diameters)
            size(diameters_this_image)
            all_diameters = [all_diameters, diameters_this_image];
            all_areas = [all_areas, areas_this_image];
            fprintf('No modification made');
         end
 %all_diameters = [all_diameters, diameters_this_image];

 %[save_file, save_path] = uiputfile('*.xlsx', 'Save results as');
        % Generate a file name 
        image_name = erase(filename, '.png'); 
        excel_filename = sprintf('%s_Detection.xlsx', image_name);
        
        % Choose a folder for the save 
        [save_file, save_path] = uiputfile(excel_filename, 'Save results as');
        
        % Create a column for file name
        detection_filename = fullfile(save_path, 'Dectection.png');
        saveas(gcf, detection_filename);
        fprintf('Detection saved to %s\n', detection_filename);
        fprintf('Number of slurries detected in this image: %d\n', length(all_diameters));
            
        % Update info
        image_info{file_idx, 2} = areas_this_image;
        image_info{file_idx, 3} = diameters_this_image;
            
        %fprintf('Number of slurries after modifications: %d\n', length(diameters_this_image));
        else
            fprintf('No valid slurries detected with the current criteria.\n');
    end
end

%N = round(0.03 * numel(all_diameters));
%all_diameters_sort = sort(all_diameters,'descend');
%big_pores = all_diameters_sort(1:N);
%all_diameters(ismember(all_diameters, big_pores)) = [];

%____________________________________________
%_______________STATISTICS_________________
%____________________________________________

[hauteur, largeur, ~] = size(I);
fprintf('Size : %d x %d pixels\n', largeur, hauteur);
total_image_surface = largeur * hauteur;
total_image_surface_microm = total_image_surface *pixel2microm * pixel2microm ;
fprintf('The total image surface is : %.f pixels^2\n', total_image_surface);

%DIAMETER STATS
min_diameter = min(all_diameters);
max_diameter = max(all_diameters);
mean_diameter = mean(all_diameters);
std_diameter = std(all_diameters);
median_diameter = median(all_diameters);
cv_diameter = (std_diameter/mean_diameter) * 100 ;
density = ((length(all_diameters)) / total_image_surface_microm)*1000000;

%AREA STATS
mean_area = mean(all_areas);
%mean_area2 = (pi*(mean_diameter*mean_diameter))/4;
min_area = min(all_areas);
max_area = max(all_areas);
std_area = std(all_areas);
median_area = median(all_areas);
cv_area = (std_area/mean_area) * 100 ;

fprintf('\nSlurry Analysis Results (All Images):\n');
fprintf('Number of valid slurries detected after modification : %d\n', length(all_diameters));

fprintf('\nStatistics:\n');
fprintf('Min diameter: %.1f µm\n', min_diameter);
fprintf('Max diameter: %.1f µm\n', max_diameter);
fprintf('Mean diameter: %.1f µm\n', mean_diameter);
fprintf('Median slurry diameter: %.1f µm\n', median_diameter);
fprintf('Standard deviation: %.1f µm\n', std_diameter);
fprintf('Coefficient of variation of diameter: %.1f%%\n', cv_diameter);

fprintf('Min area: %.1f µm\n', min_area);
fprintf('Max area: %.1f µm\n', max_area);
fprintf('Mean area: %.1f µm^2\n', mean_area);
fprintf('Standard deviation: %.1f µm\n', std_area);
fprintf('Coefficient of variation of area: %.1f%%\n', cv_area);

porosity = (sum(all_areas) / total_image_surface_microm) * 100;

fprintf('Porosity: %.1f%%\n', porosity);
fprintf('Density: %.1f pores/mm²\n', density);

% Save results in a file
fprintf('\nSave results on a file\n');

final_table = [];
image_name_row = table(string(filename), 'VariableNames', {'Image_Name'});
Pore_numbers = (1:length(all_diameters))'; 
results_table1 = table(Pore_numbers,round(all_diameters,2),'VariableNames', {'Pore_number','Diameters_microm'});
results_table2 = table(round(all_areas,2),'VariableNames', {'Areas_microm'});
results_table3 = table(round(mean_diameter,2),round(median_diameter,2) ,round(std_diameter,2), round(cv_diameter,2), 'VariableNames', {'mean_diameter','median_diameter','std_diameter','C.V'});
results_table4 = table(round(mean_area,2),round(median_area,2) ,round(std_area,2), 'VariableNames', {'mean_area','median_area','std_area'});
results_table5 = table(round(porosity,2), round(density,2),'VariableNames', {'Porosity (%)', 'Density pores/mm²'});
fprintf('\nSave results on a file2\n');

max_rows = max([height(results_table1), height(results_table2), height(results_table3),height(results_table4)]);
%max_rows = max([hauteur(results_table1), hauteur(results_table2), hauteur(results_table3),hauteur(results_table4)]);
fprintf('\nSave results on a file3\n');


% Complete table to have the same hauteurs
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
combined_table = [image_name_row, results_table1, results_table2, results_table3, results_table4, results_table5];
final_table = [final_table; combined_table];

% Export results
if ~isequal(save_file, 0)
    writetable(final_table, fullfile(save_path, save_file));
    %writetable(results_table);
    fprintf('Results saved to %s\n', fullfile(save_path, save_file));
end

%__________________________________________
%_______________HISTOGRAM__________________
%__________________________________________

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
% Save histogram
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
sheet.Shapes.AddPicture(detection_filename, 0, 1, 1000, 50, 300, 250);
workbook.Save;
fprintf('\nSave results on a file\n');





