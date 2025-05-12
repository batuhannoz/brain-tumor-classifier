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

numAnchors = 1;
[anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData, numAnchors);

net = mobilenetv2;
featureLayer = 'block_12_add'; 

lgraph = yolov2Layers(inputSize, numel(classes), anchorBoxes, net, featureLayer);

options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',35, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'ValidationData',validationData, ...
    'ValidationFrequency',50, ...
    'Plots','training-progress');

 [detector, info] = trainYOLOv2ObjectDetector(trainingData, lgraph, options);

 save('trainedYOLOv2TumorDetector.mat', 'detector');

disp('Eğitim tamamlandı ve model kaydedildi: trainedYOLOv2TumorDetector.mat');

[anchorBoxes,meanIoU] = estimateAnchorBoxes(testData, numAnchors);

% Evaluate the trained YOLO model on test data
disp('Loading trained YOLOv2 detector...');
detector = load('trainedYOLOv2TumorDetector.mat').detector;

% Set IoU threshold for evaluation
iouThreshold = 0.44;
confidenceThreshold = 0.3;

% Reset the datastore to the beginning
reset(testData);

% Count test images and collect all ground truth boxes
allTestImages = cell(0);
allGroundTruth = cell(0);
numTestImages = 0;

disp('Counting test images and collecting ground truth...');
while hasdata(testData)
    data = read(testData);
    numTestImages = numTestImages + 1;

    % Store the image and ground truth boxes
    allTestImages{numTestImages} = data{1};
    allGroundTruth{numTestImages} = data{2};
end

disp(['Test dataset contains ' num2str(numTestImages) ' images']);

% Initialize arrays to store detection results
allDetections = cell(numTestImages, 1);
allScores = cell(numTestImages, 1);
allTP = zeros(numTestImages, 1);
allFP = zeros(numTestImages, 1);
allFN = zeros(numTestImages, 1);

% Process all test images - only keeping best detection per image
disp('Evaluating detector on test images (best detection only)...');
for i = 1:numTestImages
    img = allTestImages{i};
    gtBoxes = allGroundTruth{i};

    % Apply the detector to the image
    [bboxes, scores, labels] = detect(detector, img, 'Threshold', confidenceThreshold);

    % Keep only the best detection (highest confidence)
    if ~isempty(bboxes) && ~isempty(scores)
        [maxScore, maxIdx] = max(scores);
        bestBbox = bboxes(maxIdx, :);
        bestScore = scores(maxIdx);
        if numel(labels) >= maxIdx
            bestLabel = labels(maxIdx);
        else
            bestLabel = categorical("tumor");
        end

        % Store only the best detection
        allDetections{i} = bestBbox;
        allScores{i} = bestScore;
    else
        % No detections for this image
        allDetections{i} = [];
        allScores{i} = [];
    end

    % Calculate metrics for this image with the best detection only
    [tp, fp, fn, matchedGT, matchedDet] = evaluateDetectionForImage(allDetections{i}, gtBoxes, iouThreshold);

    allTP(i) = tp;
    allFP(i) = fp;
    allFN(i) = fn;

    % Display progress
    if mod(i, 10) == 0 || i == numTestImages
        fprintf('Processed %d/%d test images\n', i, numTestImages);
    end
end

% Flatten detections and scores for precision-recall curve
% Now each image has at most one detection (the best one)
allFlatDetections = [];
allFlatScores = [];
allFlatIsTP = [];
totalGT = sum(cellfun(@(x) size(x, 1), allGroundTruth));

for i = 1:numTestImages
    if ~isempty(allDetections{i}) && ~isempty(allScores{i})
        % For best detection approach, each image has at most one detection
        allFlatDetections = [allFlatDetections; allDetections{i}];
        allFlatScores = [allFlatScores; allScores{i}];

        % Determine if this detection is a true positive
        isTP = 0;
        if ~isempty(allGroundTruth{i})
            ious = bboxOverlapRatio(allDetections{i}, allGroundTruth{i});
            if any(ious >= iouThreshold)
                isTP = 1;
            end
        end
        allFlatIsTP = [allFlatIsTP; isTP];
    end
end

% Sort detections by confidence score (descending)
[sortedScores, sortIdx] = sort(allFlatScores, 'descend');
sortedDetections = allFlatDetections(sortIdx,:); % Keep sorted detections if needed later
sortedIsTP = allFlatIsTP(sortIdx);

% Calculate cumulative TP and FP
cumTP = cumsum(sortedIsTP);
cumFP = cumsum(~sortedIsTP); % ~sortedIsTP gives FPs among sorted detections

% Calculate precision and recall at each threshold
precision = cumTP ./ (cumTP + cumFP);
% Handle cases where (cumTP + cumFP) is zero to avoid NaN
precision(isnan(precision)) = 0;
recall = cumTP / totalGT;
if totalGT == 0 % Avoid division by zero if no ground truths
    recall(:) = 0;
end


% Calculate AP using all points
ap = 0;
for t = 0:0.01:1
    if any(recall >= t)
        p = max(precision(recall >= t));
        ap = ap + p * 0.01;
    else
        ap = ap + 0 * 0.01; % if no recall >= t, precision is 0 for that segment
    end
end

% Calculate overall metrics
totalTP = sum(allTP);
totalFP = sum(allFP);
totalFN = sum(allFN);

if totalTP + totalFP > 0
    overallPrecision = totalTP / (totalTP + totalFP);
else
    overallPrecision = 0;
end

if totalTP + totalFN > 0
    overallRecall = totalTP / (totalTP + totalFN);
else
    overallRecall = 0;
end

if overallPrecision + overallRecall > 0
    f1Score = 2 * (overallPrecision * overallRecall) / (overallPrecision + overallRecall);
else
    f1Score = 0;
end

% Display evaluation results
fprintf('\n==== YOLOv2 Tumor Detector Evaluation Results (Best Detection Only) ====\n');
fprintf('Evaluation Method: Best Detection Only (Highest Confidence)\n');
fprintf('Confidence Threshold: %.2f, IoU Threshold: %.2f\n', confidenceThreshold, iouThreshold);
fprintf('Test Images: %d\n', numTestImages);
fprintf('Total Ground Truth Boxes: %d\n', totalGT);
fprintf('Total Detections: %d\n', length(allFlatScores)); % Total actual detections made
fprintf('True Positives (evaluated per image): %d\n', totalTP);
fprintf('False Positives (evaluated per image): %d\n', totalFP);
fprintf('False Negatives (evaluated per image): %d\n', totalFN);
fprintf('Average Precision (AP): %.4f\n', ap);
fprintf('Overall Precision (based on per-image TP/FP): %.4f\n', overallPrecision);
fprintf('Overall Recall (based on per-image TP/FN): %.4f\n', overallRecall);
fprintf('Overall F1 Score (based on per-image metrics): %.4f\n', f1Score);

% Create visualization of detection results
figure('Name', 'Detection Results', 'Position', [100, 100, 1000, 800]);
% title('Sample Detection Results'); % Add a main title to the figure

% Visualize sample images with detection results
% subplot(2, 3, [1, 2, 4, 5]); % This subplot command might be for a different layout
numToDisplay = min(16, numTestImages);
if numTestImages == 0
    disp('No test images to display.');
    return; % Exit if no images
end
rows = ceil(sqrt(numToDisplay));
cols = ceil(numToDisplay/rows);

% Random sample of images to display
sampleIndices = randperm(numTestImages, numToDisplay);

for i = 1:numToDisplay
    idx = sampleIndices(i);

    % Get image and detection data
    img = allTestImages{idx};
    gtBoxes = allGroundTruth{idx};
    detBoxes = allDetections{idx}; % Best detection for this image
    detScores = allScores{idx};   % Score of the best detection

    % Create subplot
    subplot(rows, cols, i);

    % Display image
    imshow(img);
    hold on;

    % Draw ground truth boxes in green
    if ~isempty(gtBoxes)
        for j = 1:size(gtBoxes, 1)
            rectangle('Position', gtBoxes(j,:), 'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '--');
            text(gtBoxes(j,1), gtBoxes(j,2)-10, 'GT', 'Color', 'g', 'FontSize', 8);
        end
    end

    % Draw detection box in red (or blue if strong match)
    if ~isempty(detBoxes) && ~isempty(detScores)
        isMatch = false;
        boxColor = 'r'; % Default to red (FP)

        % Check if this detection matches a ground truth (for visualization color)
        if ~isempty(gtBoxes)
            ious_viz = bboxOverlapRatio(detBoxes, gtBoxes);
            if any(ious_viz >= iouThreshold)
                isMatch = true;
                boxColor = 'b'; % Blue for TP in visualization
            end
        end

        rectangle('Position', detBoxes, 'EdgeColor', boxColor, 'LineWidth', 2);
        text(detBoxes(1), detBoxes(2)-5, sprintf('Det: %.2f', detScores), ...
             'Color', boxColor, 'FontWeight', 'bold', 'FontSize', 8, 'BackgroundColor', 'w');
    end

    imageTitle = sprintf('Img %d: TP=%d, FP=%d, FN=%d', idx, allTP(idx), allFP(idx), allFN(idx));
    if isempty(detBoxes) && ~isempty(gtBoxes)
        imageTitle = [imageTitle ' (Missed)'];
    elseif ~isempty(detBoxes) && isempty(gtBoxes)
        imageTitle = [imageTitle ' (False Alarm)'];
    end
    title(imageTitle, 'FontSize', 8);
    hold off;
end
sgtitle('Sample Detection Results with Ground Truth (Green) and Detections (Red/Blue)', 'FontSize', 12);


% Create PR curve
figure('Name', 'Precision-Recall Curve and Metrics', 'Position', [100, 100, 1200, 500]);

% Plot PR curve
subplot(1, 3, 1);
if ~isempty(recall) && ~isempty(precision)
    plot(recall, precision, 'b-', 'LineWidth', 2);
    hold on;
    scatter(overallRecall, overallPrecision, 100, 'ro', 'filled'); % Using overall P/R from per-image counts
    xlabel('Recall');
    ylabel('Precision');
    title(sprintf('Precision-Recall Curve (AP=%.4f)', ap));
    grid on;
    legend('PR Curve', 'Overall P/R Point', 'Location', 'southwest');
    axis([0 1 0 1]);
else
    text(0.5, 0.5, 'Not enough data for PR curve', 'HorizontalAlignment', 'center');
    title('Precision-Recall Curve');
    axis([0 1 0 1]);
    grid on;
end


% Create metrics visualization (Bar Chart)
subplot(1, 3, 2);
metricNames = {'Precision', 'Recall', 'F1 Score', 'AP'};
metricValues = [overallPrecision, overallRecall, f1Score, ap];
barHandle = bar(metricValues);
set(gca, 'XTickLabel', metricNames);
ylim([0 1.05]); % Extend y-limit slightly for text
grid on;
title('Overall Performance Metrics');
% Add text labels on top of bars
for k=1:length(metricValues)
    text(k, metricValues(k), sprintf('%.3f', metricValues(k)),...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom');
end

% NEW: Confidence Score Distribution for TPs and FPs
subplot(1, 3, 3);
tp_scores = sortedScores(sortedIsTP == 1);
fp_scores = sortedScores(sortedIsTP == 0);

if ~isempty(tp_scores) || ~isempty(fp_scores)
    hold on; % Hold on before plotting histograms
    if ~isempty(fp_scores)
        histogram(fp_scores, 'FaceColor', 'r', 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'DisplayName', sprintf('FP Scores (N=%d)', numel(fp_scores)));
    end
    if ~isempty(tp_scores)
        histogram(tp_scores, 'FaceColor', 'b', 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'DisplayName', sprintf('TP Scores (N=%d)', numel(tp_scores)));
    end
    hold off; % Release hold after plotting
    xlabel('Confidence Score');
    ylabel('Number of Detections');
    title('Confidence Score Distribution (TP vs FP)');
    legend('show', 'Location', 'northeast');
    grid on;
else
    text(0.5, 0.5, 'No detections to show score distribution', 'HorizontalAlignment', 'center');
    title('Confidence Score Distribution');
    grid on;
end
sgtitle('Model Evaluation Metrics and Score Analysis', 'FontSize', 14);


% Create a results table to display
resultTable = table(overallPrecision, overallRecall, f1Score, ap, ...
    totalTP, totalFP, totalFN, totalGT, length(allFlatScores), ...
    'VariableNames', {'OverallPrecision', 'OverallRecall', 'F1_Score', 'AP', ...
    'TotalTP_perImage', 'TotalFP_perImage', 'TotalFN_perImage', 'TotalGroundTruth', 'TotalDetectionsMade'});
disp(resultTable);

% Save evaluation results
save('yolov2_tumor_detector_evaluation_best_only.mat', 'resultTable', 'ap', 'precision', 'recall', ...
    'overallPrecision', 'overallRecall', 'f1Score', 'sortedScores', 'sortedIsTP', 'confidenceThreshold', 'iouThreshold');

% Function to evaluate detections for a single image - best detection only approach
function [tp, fp, fn, matchedGT, matchedDet] = evaluateDetectionForImage(detections, groundTruth, iouThreshold)
    % Initialize metrics
    tp = 0;
    fp = 0;
    fn = 0;

    % Initialize arrays to track which boxes are matched
    numGT = size(groundTruth, 1);
    matchedGT = false(numGT, 1);
    matchedDet = []; % Will be a scalar boolean for single detection

    % Case 1: No detections, No ground truth
    if isempty(detections) && isempty(groundTruth)
        return; % tp=0, fp=0, fn=0
    end

    % Case 2: No detections, but ground truth exists
    if isempty(detections) && ~isempty(groundTruth)
        fn = 1; % At least one ground truth was missed for the image
        return;
    end

    % Case 3: Detections, but no ground truth
    if ~isempty(detections) && isempty(groundTruth)
        fp = 1; % The detection is a false positive for the image
        matchedDet = false;
        return;
    end

    % Case 4: Both detections and ground truth exist
    % For the best detection approach, we only have one detection to evaluate
    % Assert that detections is a single row if not empty
    % if ~isempty(detections) && size(detections,1) ~=1
    %     error('Expected only one best detection per image for this evaluation mode.');
    % end

    detection = reshape(detections, 1, 4); % Ensure it's a row vector

    % Calculate IoU with all ground truth boxes
    ious = bboxOverlapRatio(detection, groundTruth);

    % Find the best matching ground truth box
    [maxIoU, maxIdxGT] = max(ious);

    if maxIoU >= iouThreshold
        tp = 1;
        matchedGT(maxIdxGT) = true; % Mark this GT as matched (though only one detection)
        matchedDet = true;
    else
        fp = 1;
        matchedDet = false;
        % If no detection matched, and there was ground truth, it's also a FN context for the image.
        % The fn here counts if *any* ground truth on the image was not detected by this *single best* detection.
        % This fn is tricky in "best detection only". If TP=1, FN=0 for this detection.
        % If FP=1, it means this detection didn't match. If there were GTs, they are effectively FN for this image.
        % The current script calculates overall FN by sum(allFN(i)) where allFN(i) is based on this function.
        % Let's refine fn definition for this function:
        % If we have a detection, and it's an FP, it means it didn't match any GT.
        % If there *were* GTs on the image, those GTs are now effectively missed *by this detection process for the image*.
        % So, if fp=1 (detection is false positive) AND ground truth exists, then fn should be 1 for the image.
    end

    % Final check for False Negative for the image:
    % An image-level FN occurs if there was ground truth, but no true positive was found for that image.
    % If tp=0 (meaning the single best detection was not a TP) AND there was ground truth.
    if tp == 0 && numGT > 0
        fn = 1;
    end
    % Note: if tp=1, fn must be 0 for the image in "best detection only", because the single detection successfully found a tumor.
    % The 'evaluateDetectionForImage' aims to give image-level TP/FP/FN for the "best detection" strategy.
end