% YOLOv2 Tumor Detection Training Script

% Load the YOLOv2 dataset
data = load("YOLOv2_dataset.mat");
tumorDataset = data.T;

tumorDataset.imageFilename = fullfile("archive/", tumorDataset.imageFilename);

% Replace backslashes with forward slashes in imageFilename
tumorDataset.imageFilename = replace(tumorDataset.imageFilename, "\", "/");

tumorDataset(1:4,:);

% Shuffle indices and split into 70%-15%-15% partitions
rng(3);
shuffledIndices = randperm(height(tumorDataset));
N = height(tumorDataset);

% Calculate split indices
trainEnd = floor(0.7 * N);
valEnd = trainEnd + floor(0.15 * N);

trainingIdx = 1:trainEnd;
validationIdx = (trainEnd+1):valEnd;
testIdx = (valEnd+1):N;

% Create partitioned tables
trainingDataTbl = tumorDataset(shuffledIndices(trainingIdx),:);
validationDataTbl = tumorDataset(shuffledIndices(validationIdx),:);
testDataTbl = tumorDataset(shuffledIndices(testIdx),:);

% Create image and box label datastores
imdsTrain = imageDatastore(trainingDataTbl{:,"imageFilename"});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,"tumor"));

imdsValidation = imageDatastore(validationDataTbl{:,"imageFilename"});
bldsValidation = boxLabelDatastore(validationDataTbl(:,"tumor"));

imdsTest = imageDatastore(testDataTbl{:,"imageFilename"});
bldsTest = boxLabelDatastore(testDataTbl(:,"tumor"));

% Define inputSize before transforming the datastores
inputSize = [224 224 3];

% Combine datastores
cdsTrain = combine(imdsTrain, bldsTrain);
cdsValidation = combine(imdsValidation, bldsValidation);
cdsTest = combine(imdsTest, bldsTest);

% Function to preprocess data: resize image, convert to RGB, and resize bounding boxes
function data = preprocessData(data, targetSize)
    % Resize image
    img = data{1};
    scale = targetSize(1:2)./size(img,[1 2]);
    img = imresize(img,targetSize(1:2));

    % Convert to RGB if grayscale
    if size(img,3) == 1
        img = cat(3, img, img, img);
    end
    data{1} = img;

    boxData = data{2};
    if isempty(boxData)
        data{2} = zeros(0,4);
        if numel(data) >=3
             data{3} = {};
        end
        return;
    end

    % Ensure boxData is numeric and has 4 columns
    if ~isnumeric(boxData) || size(boxData,2) ~= 4
        if iscell(boxData) && all(cellfun(@isnumeric, boxData)) && all(cellfun(@(x) numel(x)==4 || isempty(x), boxData))
            % Handle cases where some cells might be empty bounding boxes
            validBoxes = ~cellfun(@isempty, boxData);
            if ~any(validBoxes) % All boxes are empty
                 data{2} = zeros(0,4);
                 if numel(data) >=3, data{3} = {}; end
                 return;
            end
            boxData = cell2mat(boxData(validBoxes));
             % Adjust labels if some boxes were empty - this part can be complex
             % For simplicity, assuming labels correspond to valid boxes or are handled accordingly
        else
            warning('Invalid bounding box data encountered.');
            data{2} = zeros(0,4);
            if numel(data) >=3, data{3} = {}; end
            return;
        end
    end
    
    if ~isempty(boxData)
        validBoxIndices = all(boxData > 0, 2) & (boxData(:,3) > 0) & (boxData(:,4) > 0);
        boxData = boxData(validBoxIndices, :);
        if numel(data) >=3 && iscell(data{3})
            labels = data{3};
            if ~isempty(labels) && numel(labels) == numel(validBoxIndices)
                data{3} = labels(validBoxIndices);
            elseif ~isempty(labels) && numel(labels) ~= size(boxData,1)
                warning('Label count mismatch after filtering invalid boxes. Review labels.');
            end
        end
    end

    if isempty(boxData)
        data{2} = zeros(0,4);
        if numel(data) >=3, data{3} = {}; end
        return;
    end

    data{2} = bboxresize(boxData,scale);

    % Handle labels
    if numel(data) >= 3
        labels = data{3};
        if isempty(labels)
             data{3} = {}; 
        elseif ~iscell(labels) && ~iscategorical(labels)
            if ischar(labels) || isstring(labels)
                data{3} = {char(labels)};
            else
                warning('Invalid label data encountered.');
                data{3} = {}; 
            end
        elseif iscell(labels) && ~all(cellfun(@(x) ischar(x) || isstring(x) || iscategorical(x), labels))
             warning('Invalid label data within cell array.');
             data{3} = {};
        end
    else
        data{3} = {};
    end
end

trainingData = transform(cdsTrain, @(data)preprocessData(data, inputSize));
validationData = transform(cdsValidation, @(data)preprocessData(data, inputSize));
testData = transform(cdsTest, @(data)preprocessData(data, inputSize));

data = read(trainingData);
I = data{1}; 
bbox = data{2};
annotatedImage = insertShape(I,"rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

classes = "tumor";

numAnchors = 7;
[anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData, numAnchors);

net = mobilenetv2;
featureLayer = 'block_12_add'; 

lgraph = yolov2Layers(inputSize, numel(classes), anchorBoxes, net, featureLayer);

options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'ValidationData',validationData, ...
    'ValidationFrequency',50, ...
    'Plots','training-progress');

% [detector, info] = trainYOLOv2ObjectDetector(trainingData, lgraph, options);

save('trainedYOLOv2TumorDetector.mat', 'detector');
disp('Eğitim tamamlandı ve model kaydedildi: trainedYOLOv2TumorDetector.mat');

[anchorBoxes,meanIoU] = estimateAnchorBoxes(testData, numAnchors);


% Evaluate the trained YOLO model on test data
detector = load('trainedYOLOv2TumorDetector.mat').detector;

% Initialize evaluation metrics
totalTP = 0;
totalFP = 0;
totalFN = 0;

% Reset the datastore to the beginning
reset(testData);

% Calculate number of test images more safely
numTestImages = 0;
while hasdata(testData)
    read(testData);
    numTestImages = numTestImages + 1;
end
disp(['Number of test images: ' num2str(numTestImages)]);

% Reset the datastore again to start processing
reset(testData);

% Display progress
disp('Evaluating detector performance on test data...');

% Create a figure to store results
figure('Name', 'Detection Results', 'Position', [100, 100, 800, 600]);

% Create a counter for the subplot
plotIndex = 1;

% Process the test data
for i = 1:min(numTestImages, 10)  % Limit to 10 images for visualization
    if ~hasdata(testData)
        break;
    end

    % Get the next test data
    data = read(testData);
    img = data{1};
    gtBoxes = data{2};

    % Run detection
    [bboxes, scores, labels] = detect(detector, img, 'Threshold', 0.3);

    % Calculate IoU between detections and ground truth
    tp = 0;
    fp = size(bboxes, 1);
    fn = 0;

    if ~isempty(bboxes) && ~isempty(gtBoxes)
        % Calculate IoU for each detection with each ground truth box
        ious = bboxOverlapRatio(bboxes, gtBoxes);

        if ~isempty(ious)
            % For each ground truth box, find the detection with max IoU
            [maxIoU, ~] = max(ious, [], 1);

            % Count true positives (IoU > 0.5)
            tp = sum(maxIoU > 0.5);
            fp = size(bboxes, 1) - tp;

            % Count false negatives
            fn = sum(maxIoU <= 0.5);
        else
            % If IoU calculation failed, count all detections as false positives
            fp = size(bboxes, 1);
            fn = size(gtBoxes, 1);
        end
    elseif isempty(bboxes) && ~isempty(gtBoxes)
        % No detections but there are ground truth boxes - all are false negatives
        fn = size(gtBoxes, 1);
    elseif ~isempty(bboxes) && isempty(gtBoxes)
        % Detections but no ground truth - all are false positives
        fp = size(bboxes, 1);
    end

    % Accumulate metrics
    totalTP = totalTP + tp;
    totalFP = totalFP + fp;
    totalFN = totalFN + fn;

    % Display results for this image
    fprintf('Image %d: TP=%d, FP=%d, FN=%d\n', i, tp, fp, fn);

    % Create a subplot for this image
    if plotIndex <= 12  % Only display up to 12 images in the subplot
        subplot(3, 4, plotIndex);
        plotIndex = plotIndex + 1;

        % Display the image with annotations
        imshow(img);
        hold on;

        % Draw ground truth boxes in green
        if ~isempty(gtBoxes)
            for j = 1:size(gtBoxes, 1)
                rectangle('Position', gtBoxes(j, :), 'EdgeColor', 'g', 'LineWidth', 2);
            end
        end

        % Draw detection boxes in red
        if ~isempty(bboxes)
            for j = 1:size(bboxes, 1)
                rectangle('Position', bboxes(j, :), 'EdgeColor', 'r', 'LineWidth', 2);
                text(bboxes(j, 1), bboxes(j, 2)-5, sprintf('%.2f', scores(j)), ...
                    'Color', 'red', 'FontWeight', 'bold');
            end
        end

        title(sprintf('Image %d', i));
        hold off;
    end
end

% Calculate overall metrics
if (totalTP + totalFP) > 0
    precision = totalTP / (totalTP + totalFP);
else
    precision = 0;
end

if (totalTP + totalFN) > 0
    recall = totalTP / (totalTP + totalFN);
else
    recall = 0;
end

% Handle division by zero
if precision + recall > 0
    f1Score = 2 * (precision * recall) / (precision + recall);
else
    f1Score = 0;
end

% Display evaluation results
fprintf('\n==== Tumor Detector Evaluation Results ====\n');
fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1Score);

% Create a summary figure with metrics
figure('Name', 'Evaluation Metrics', 'Position', [100, 100, 400, 300]);
metrics = [precision, recall, f1Score];
bar(metrics);
set(gca, 'XTickLabel', {'Precision', 'Recall', 'F1 Score'});
ylim([0 1]);
grid on;
title('YOLOv2 Tumor Detector Performance');

% Create a results table to display
resultTable = table(precision, recall, f1Score, ...
    'VariableNames', {'Precision', 'Recall', 'F1_Score'});
disp(resultTable);
