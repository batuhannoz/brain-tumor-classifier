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

% Analyze Network Architecture (Opens a new window)
disp('Analyzing network architecture...');
analyzeNetwork(lgraph);
disp('Network analysis window opened.');

% Open Network in Deep Network Designer (Opens the App)
disp('Opening Deep Network Designer...');
try
    deepNetworkDesigner(lgraph);
    disp('Deep Network Designer launched with the defined layers.');
    disp('Use the "Analyze" button within the app for further details.');
catch ME
    warning('Could not automatically open Deep Network Designer. Error: %s', ME.message);
    disp('You can still manually open Deep Network Designer from the MATLAB Apps tab and import the `layers` variable.');
end

options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',35, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'ValidationData',validationData, ...
    'ValidationFrequency',50, ...
    'Plots','training-progress');

% Uncomment these lines to train the model
% [detector, info] = trainYOLOv2ObjectDetector(trainingData, lgraph, options);
% 
% % Save the trained detector
% save('trainedYOLOv2TumorDetector.mat', 'detector');

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


% Generate clear, concise visualizations of model performance metrics using our functions
visualizeModelMetrics(precision, recall, ap, overallPrecision, overallRecall, f1Score);
visualizeConfidenceDistribution(sortedScores, sortedIsTP);


% Create a results table to display
resultTable = table(overallPrecision, overallRecall, f1Score, ap, ...
    totalTP, totalFP, totalFN, totalGT, length(allFlatScores), ...
    'VariableNames', {'OverallPrecision', 'OverallRecall', 'F1_Score', 'AP', ...
    'TotalTP_perImage', 'TotalFP_perImage', 'TotalFN_perImage', 'TotalGroundTruth', 'TotalDetectionsMade'});
disp(resultTable);

% Generate concise visualizations of model performance metrics
fprintf('\nOluşturuluyor: Ağ eğitim grafikleri ve model performans metrikleri...\n');

% If we have training info, visualize the training progress
if exist('info', 'var') && isstruct(info)
    visualizeTrainingProgress(info);
end

% Visualize model evaluation metrics
visualizeModelMetrics(precision, recall, ap, overallPrecision, overallRecall, f1Score);

% Visualize confidence score distributions
visualizeConfidenceDistribution(sortedScores, sortedIsTP);

% Generate comprehensive dashboard with all metrics in one view
createMetricsDashboard(ap, precision, recall, overallPrecision, overallRecall, f1Score, ...
                       sortedScores, sortedIsTP, totalTP, totalFP, totalFN, totalGT);

fprintf('Görselleştirmeler tamamlandı. Grafikler ayrı pencerede gösteriliyor.\n');

% Save evaluation results
save('yolov2_tumor_detector_evaluation_best_only.mat', 'resultTable', 'ap', 'precision', 'recall', ...
    'overallPrecision', 'overallRecall', 'f1Score', 'sortedScores', 'sortedIsTP', 'confidenceThreshold', 'iouThreshold');

% Function to visualize network training information and metrics
function visualizeTrainingProgress(trainInfo)
    % Create a figure with two subplots for network training graphs
    figure('Name', 'Ağ Eğitim Grafikleri', 'Position', [100, 100, 900, 400]);
    
    % Loss values over iterations
    subplot(1, 2, 1);
    plot(trainInfo.TrainingLoss, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(trainInfo.ValidationLoss, 'r-', 'LineWidth', 1.5);
    xlabel('İterasyon');
    ylabel('Kayıp Değeri (Loss)');
    title('Eğitim ve Validasyon Kayıp Değerleri');
    legend('Eğitim Kaybı', 'Validasyon Kaybı');
    grid on;
    
    % Training accuracy over iterations
    subplot(1, 2, 2);
    plot(trainInfo.TrainingAccuracy, 'b-', 'LineWidth', 1.5);
    hold on;
    if isfield(trainInfo, 'ValidationAccuracy')
        plot(trainInfo.ValidationAccuracy, 'r-', 'LineWidth', 1.5);
        legend('Eğitim Doğruluğu', 'Validasyon Doğruluğu');
    else
        legend('Eğitim Doğruluğu');
    end
    xlabel('İterasyon');
    ylabel('Doğruluk (%)');
    title('Eğitim Doğruluk Değerleri');
    grid on;
end

% Function to visualize model performance metrics
function visualizeModelMetrics(precision, recall, ap, overallPrecision, overallRecall, f1Score)
    % Create a figure for model performance metrics
    figure('Name', 'Model Performans Metrikleri', 'Position', [100, 100, 1000, 400]);
    
    % Plot PR curve
    subplot(1, 2, 1);
    plot(recall, precision, 'b-', 'LineWidth', 2);
    hold on;
    scatter(overallRecall, overallPrecision, 100, 'ro', 'filled');
    xlabel('Recall');
    ylabel('Precision');
    title(sprintf('Precision-Recall Eğrisi (AP=%.4f)', ap));
    grid on;
    legend('PR Eğrisi', 'Ortalama P/R Noktası', 'Location', 'southwest');
    axis([0 1 0 1]);
    
    % Create bar chart for metrics
    subplot(1, 2, 2);
    metricNames = {'Precision', 'Recall', 'F1 Score', 'AP'};
    metricValues = [overallPrecision, overallRecall, f1Score, ap];
    barHandle = bar(metricValues);
    barHandle.FaceColor = 'flat';
    barHandle.CData(1,:) = [0.2 0.6 0.8];
    barHandle.CData(2,:) = [0.8 0.4 0.2];
    barHandle.CData(3,:) = [0.2 0.8 0.4];
    barHandle.CData(4,:) = [0.6 0.2 0.8];
    set(gca, 'XTickLabel', metricNames);
    ylim([0 1.05]);
    grid on;
    title('Genel Performans Metrikleri');
    % Add value labels on top of bars
    for k=1:length(metricValues)
        text(k, metricValues(k)+0.02, sprintf('%.3f', metricValues(k)),...
            'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontWeight', 'bold');
    end
end

% Function to visualize confidence score distributions and confusion matrix for detections
function visualizeConfidenceDistribution(sortedScores, sortedIsTP)
    % Create figure for confidence score distribution
    figure('Name', 'Deteksiyon Güven Skoru Dağılımı', 'Position', [100, 100, 800, 400]);
    
    % Extract TP and FP scores
    tp_scores = sortedScores(sortedIsTP == 1);
    fp_scores = sortedScores(sortedIsTP == 0);
    
    % Plot score distributions as histograms
    if ~isempty(tp_scores) || ~isempty(fp_scores)
        hold on;
        if ~isempty(fp_scores)
            histogram(fp_scores, 10, 'FaceColor', 'r', 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'DisplayName', sprintf('FP Skorları (N=%d)', numel(fp_scores)));
        end
        if ~isempty(tp_scores)
            histogram(tp_scores, 10, 'FaceColor', 'b', 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'DisplayName', sprintf('TP Skorları (N=%d)', numel(tp_scores)));
        end
        hold off;
        xlabel('Güven Skoru');
        ylabel('Tespit Sayısı');
        title('Güven Skoru Dağılımı (TP vs FP)');
        legend('show', 'Location', 'northeast');
        grid on;
        xlim([0 1]);
    else
        text(0.5, 0.5, 'Gösterilecek tespit skoru bulunamadı', 'HorizontalAlignment', 'center');
        title('Güven Skoru Dağılımı');
        grid on;
    end
end

% Function to create a comprehensive metrics dashboard
function createMetricsDashboard(ap, precision, recall, overallPrecision, overallRecall, f1Score, sortedScores, sortedIsTP, totalTP, totalFP, totalFN, totalGT)
    % Create a single figure with multiple subplots for a comprehensive dashboard
    figure('Name', 'Tümör Detektörü Performans Metrikleri', 'Position', [50, 50, 1200, 700]);
    
    % 1. Precision-Recall curve
    subplot(2, 3, 1);
    plot(recall, precision, 'b-', 'LineWidth', 2);
    hold on;
    scatter(overallRecall, overallPrecision, 100, 'ro', 'filled');
    xlabel('Recall');
    ylabel('Precision');
    title(sprintf('Precision-Recall Eğrisi (AP=%.4f)', ap));
    grid on;
    legend('PR Eğrisi', 'Ortalama P/R', 'Location', 'southwest');
    axis([0 1 0 1]);
    
    % 2. Performance metrics bar chart
    subplot(2, 3, 2);
    metricNames = {'Precision', 'Recall', 'F1 Score', 'AP'};
    metricValues = [overallPrecision, overallRecall, f1Score, ap];
    barHandle = bar(metricValues);
    barHandle.FaceColor = 'flat';
    barHandle.CData(1,:) = [0.2 0.6 0.8]; % Custom colors
    barHandle.CData(2,:) = [0.8 0.4 0.2];
    barHandle.CData(3,:) = [0.2 0.8 0.4];
    barHandle.CData(4,:) = [0.6 0.2 0.8];
    set(gca, 'XTickLabel', metricNames);
    ylim([0 1.05]);
    grid on;
    title('Performans Metrikleri');
    for k=1:length(metricValues)
        text(k, metricValues(k)+0.02, sprintf('%.3f', metricValues(k)),...
            'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontWeight', 'bold');
    end
    
    % 3. Confidence score distribution
    subplot(2, 3, 3);
    tp_scores = sortedScores(sortedIsTP == 1);
    fp_scores = sortedScores(sortedIsTP == 0);
    
    if ~isempty(tp_scores) || ~isempty(fp_scores)
        hold on;
        if ~isempty(fp_scores)
            histogram(fp_scores, 10, 'FaceColor', 'r', 'FaceAlpha', 0.7, 'EdgeColor', 'none', ...
                'DisplayName', sprintf('Yanlış Pozitifler (N=%d)', numel(fp_scores)));
        end
        if ~isempty(tp_scores)
            histogram(tp_scores, 10, 'FaceColor', 'b', 'FaceAlpha', 0.7, 'EdgeColor', 'none', ...
                'DisplayName', sprintf('Doğru Pozitifler (N=%d)', numel(tp_scores)));
        end
        hold off;
        xlabel('Güven Skoru');
        ylabel('Tespit Sayısı');
        title('Güven Skoru Dağılımı');
        legend('show', 'Location', 'northeast');
        grid on;
        xlim([0 1]);
    else
        text(0.5, 0.5, 'Gösterilecek tespit skoru bulunamadı', 'HorizontalAlignment', 'center');
        title('Güven Skoru Dağılımı');
        grid on;
    end
    
    % 4. Confusion metrics pie chart (TP, FP, FN)
    subplot(2, 3, 4);
    confusionData = [totalTP, totalFP, totalFN];
    labels = {sprintf('TP: %d', totalTP), sprintf('FP: %d', totalFP), sprintf('FN: %d', totalFN)};
    explode = [0.1 0.1 0.1];
    pie(confusionData, explode, labels);
    title('Tespit Sonuçları Dağılımı');
    colormap([0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250]); % Blue, Orange, Yellow
    
    % 5. Recall by confidence threshold simulation
    subplot(2, 3, 5);
    thresholds = 0:0.1:1;
    recalls = zeros(size(thresholds));
    
    % Simulate different confidence thresholds and calculate recall
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        detections_above_thresh = sum(sortedScores >= thresh);
        
        if detections_above_thresh > 0
            tp_above_thresh = sum(sortedIsTP(sortedScores >= thresh));
            recalls(i) = tp_above_thresh / totalGT;
        else
            recalls(i) = 0;
        end
    end
    
    plot(thresholds, recalls, 'g-o', 'LineWidth', 2, 'MarkerFaceColor', 'g');
    xlabel('Güven Eşiği');
    ylabel('Recall');
    title('Güven Eşiği vs. Recall');
    grid on;
    ylim([0 1]);
    
    % 6. Precision by confidence threshold simulation
    subplot(2, 3, 6);
    precisions = zeros(size(thresholds));
    
    % Simulate different confidence thresholds and calculate precision
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        detections_above_thresh = sum(sortedScores >= thresh);
        
        if detections_above_thresh > 0
            tp_above_thresh = sum(sortedIsTP(sortedScores >= thresh));
            precisions(i) = tp_above_thresh / detections_above_thresh;
        else
            precisions(i) = 1; % No detections means no false positives
        end
    end
    
    plot(thresholds, precisions, 'm-o', 'LineWidth', 2, 'MarkerFaceColor', 'm');
    xlabel('Güven Eşiği');
    ylabel('Precision');
    title('Güven Eşiği vs. Precision');
    grid on;
    ylim([0 1]);
    
    % Add overall title
    sgtitle('Tümör Detektörü MAP ve F1 Değerlendirme Sonuçları', 'FontSize', 14, 'FontWeight', 'bold');
end

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