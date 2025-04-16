%% Brain Tumor Classification with MobileNetV2 Transfer Learning in MATLAB
% This script demonstrates how to use transfer learning with MobileNetV2
% to classify brain tumor X-rays
% The dataset is organized in two folders:
% - /archive/no: X-rays without tumors
% - /archive/yes: X-rays with tumors

%% 1. Set up the environment and paths
clear all;
close all;
clc;

% Define paths
positiveFolder = fullfile('archive', 'yes');
negativeFolder = fullfile('archive', 'no');

% Check if folders exist
if ~exist(positiveFolder, 'dir') || ~exist(negativeFolder, 'dir')
    error('Dataset folders not found. Please check the paths.');
end

%% 2. Load and preprocess images
% Create an imageDatastore to manage the image data
imds = imageDatastore({positiveFolder, negativeFolder}, ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true);

% Count number of images in each category
labelCount = countEachLabel(imds);
disp('Dataset distribution:');
disp(labelCount);

% Check for class imbalance
totalImages = sum(labelCount.Count);
fprintf('Total images: %d\n', totalImages);
fprintf('Percentage of tumor images: %.2f%%\n', 100*labelCount.Count(labelCount.Label == 'yes')/totalImages);
fprintf('Percentage of non-tumor images: %.2f%%\n', 100*labelCount.Count(labelCount.Label == 'no')/totalImages);

% Read one image to get some information
sampleImg = readimage(imds, 1);
fprintf('Sample image is %d x %d pixels\n', size(sampleImg, 1), size(sampleImg, 2));
fprintf('Sample image has %d channels\n', size(sampleImg, 3));

% Display a sample image
figure;
imshow(sampleImg);
title('Sample image from dataset');

% Add code to check and process the image format
% If images are grayscale, make sure they are properly converted to RGB
if size(sampleImg, 3) == 1
    fprintf('Images are grayscale, will be converted to RGB for MobileNetV2\n');
end

%% 3. Load MobileNetV2 for transfer learning
% Ensure Deep Learning Toolbox Model for MobileNetV2 Network is installed
% You may need to run: 
% >> addpath(fullfile(matlabroot,'examples','nnet','main'));
% >> vgg16

try
    % Load pre-trained MobileNetV2
    net = mobilenetv2;
    fprintf('MobileNetV2 loaded successfully\n');
catch ME
    error(['Error loading MobileNetV2. Make sure Deep Learning Toolbox ', ...
           'and MobileNetV2 support package are installed.\n', ...
           'Error details: %s'], ME.message);
end

% Examine the network architecture 
% Comment this out if it causes issues - it's just for visualization
try
    analyzeNetwork(net);
catch
    fprintf('Unable to analyze network, but continuing with the process\n');
end

% For MobileNetV2, the input size must be 224x224x3
inputSize = [224, 224, 3]; % Height, width, channels (3 for RGB)

%% 4. Split the data into training, validation, and testing sets (70%, 15%, 15%)
% Shuffle the data first
imds = shuffle(imds);

% Get the number of observations
numObservations = numel(imds.Files);
numTrain = floor(0.7 * numObservations);
numVal = floor(0.15 * numObservations);
numTest = numObservations - numTrain - numVal;

% Create indices for splitting
indices = 1:numObservations;
trainIndices = indices(1:numTrain);
valIndices = indices(numTrain+1:numTrain+numVal);
testIndices = indices(numTrain+numVal+1:end);

% Create the respective datastores
trainingDS = subset(imds, trainIndices);
validationDS = subset(imds, valIndices);
testingDS = subset(imds, testIndices);

% Create augmented datastores for each split with resizing and pixel range adjustments
% Define preprocessing function to ensure images are properly formatted for MobileNetV2
imageAugmenter = imageDataAugmenter('RandXReflection', true);

augTrainingDS = augmentedImageDatastore(inputSize(1:2), trainingDS, ...
    'ColorPreprocessing', 'gray2rgb', ...
    'DataAugmentation', imageAugmenter);
augValidationDS = augmentedImageDatastore(inputSize(1:2), validationDS, ...
    'ColorPreprocessing', 'gray2rgb');
augTestingDS = augmentedImageDatastore(inputSize(1:2), testingDS, ...
    'ColorPreprocessing', 'gray2rgb');

% Display the split information
fprintf('Training set: %d images\n', numel(trainingDS.Files));
fprintf('Validation set: %d images\n', numel(validationDS.Files));
fprintf('Testing set: %d images\n', numel(testingDS.Files));

%% 5. Modify MobileNetV2 for our classification task
% Get the number of classes
numClasses = numel(categories(trainingDS.Labels));

% Create a simpler approach to modify the network
% Get the layer graph from the network
lgraph = layerGraph(net);

% Find the feature layer
featureLayer = 'global_average_pooling2d_1';

% Get input image size for the network
inputSize = net.Layers(1).InputSize;

% Check network layers
net_layers = net.Layers;
layerNames = {net_layers.Name};

% Create new layers for the classification task
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')
];

% Remove the layers after the feature layer
lgraph = removeLayers(lgraph, layerNames(end-2:end));

% Add the new layers
lgraph = addLayers(lgraph, newLayers);

% Connect the feature extraction layer to the new layers
lgraph = connectLayers(lgraph, featureLayer, 'new_fc');

% Display the modified network
figure;
plot(lgraph);
title('Modified MobileNetV2 for Brain Tumor Classification');

%% 6. Define training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 10, ... % Reduced epochs for transfer learning
    'MiniBatchSize', 8, ... % Reduced batch size to fit in memory
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidationDS, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu', ... % Force CPU usage to avoid possible GPU memory issues
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 5, ...
    'OutputFcn', @(info)disp(['Epoch: ', num2str(info.Epoch), ', Iteration: ', num2str(info.Iteration)]));

%% 7. Train the model using transfer learning
fprintf('Starting model training with MobileNetV2 transfer learning...\n');

% Add a try-catch block to provide better error handling
try
    fprintf('Verifying data format before training...\n');
    % Get a mini-batch from the augmented data store
    miniBatch = preview(augTrainingDS);
    fprintf('Mini-batch data dimensions: %dx%dx%dx%d\n', size(miniBatch.input));
    
    % Verify the input data is in the correct format
    if size(miniBatch.input, 4) > 0
        fprintf('Input data format is valid. Starting training...\n');
        tic;
        net = trainNetwork(augTrainingDS, lgraph, options);
        trainingTime = toc;
        fprintf('Training completed in %.2f seconds (%.2f minutes)\n', trainingTime, trainingTime/60);
    else
        error('Input data format is invalid. Please check your data preprocessing.');
    end
catch ME
    fprintf('Error during training: %s\n', ME.message);
    fprintf('Error details:\n');
    fprintf('%s\n', getReport(ME));
    error('Training failed. Please check input data format and network configuration.');
end

%% 8. Evaluate the model on test data
% Make predictions on the test set
predictions = classify(net, augTestingDS);
actualLabels = testingDS.Labels;

% Calculate accuracy
accuracy = sum(predictions == actualLabels) / numel(actualLabels);
fprintf('Test accuracy: %.2f%%\n', accuracy * 100);

% Create and display confusion matrix
confMat = confusionmat(actualLabels, predictions);
figure;
confusionchart(confMat, categories(actualLabels), 'Title', 'Brain Tumor Classification Results');

% Calculate more performance metrics
tp = confMat(1,1); % True positives (assuming 'yes' is first class)
fp = confMat(2,1); % False positives
fn = confMat(1,2); % False negatives
tn = confMat(2,2); % True negatives

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1Score = 2 * (precision * recall) / (precision + recall);
specificity = tn / (tn + fp);

fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1Score);
fprintf('Specificity: %.4f\n', specificity);

%% 9. Save the trained model
% Create a timestamp for the filename
dateString = datestr(now, 'yyyymmdd_HHMMSS');
modelFilename = sprintf('brain_tumor_mobilenetv2_%s.mat', dateString);

% Save the model and important metadata
save(modelFilename, 'net', 'inputSize', 'accuracy', 'trainingTime', 'precision', 'recall', 'f1Score');
fprintf('Model saved as: %s\n', modelFilename);

%% 10. Visualize some sample predictions
% Display some sample predictions from the test set
numSamplesToDisplay = min(8, numel(testingDS.Files));
figure('Name', 'Sample Test Predictions', 'Position', [100 100 900 700]);

for i = 1:numSamplesToDisplay
    % Read and resize the image
    img = readimage(testingDS, i);
    if size(img, 3) ~= 3
        img = repmat(img, [1 1 3]); % Convert grayscale to RGB for display
    end
    
    % Get the prediction
    resizedImg = imresize(img, inputSize(1:2));
    if size(resizedImg, 3) ~= 3
        resizedImg = repmat(resizedImg, [1 1 3]); % Convert grayscale to RGB for prediction
    end
    pred = classify(net, augmentedImageDatastore(inputSize(1:2), resizedImg, 'ColorPreprocessing', 'gray2rgb'));
    
    % Display the image with prediction
    subplot(2, 4, i);
    imshow(img);
    actualLabel = testingDS.Labels(i);
    isCorrect = pred == actualLabel;
    
    % Create title with color coding (green for correct, red for incorrect)
    if isCorrect
        titleColor = '[0 0.7 0]'; % Green
    else
        titleColor = '[0.7 0 0]'; % Red
    end
    
    title(sprintf('Pred: %s, Actual: %s', string(pred), string(actualLabel)), ...
        'Color', titleColor, 'FontWeight', 'bold');
end

%% 11. Additional: Feature Visualization (optional)
% Visualize activations of a specific layer
if exist('deepDreamImage', 'file')
    try
        layer = 'conv_pw_13_relu'; % Choose an intermediate layer from MobileNetV2
        channels = 1:4; % Choose a few channels to visualize
        
        figure('Name', 'MobileNetV2 Feature Visualization', 'Position', [100 100 900 700]);
        for i = 1:length(channels)
            channel = channels(i);
            img = deepDreamImage(net, layer, channel, 'PyramidLevels', 1);
            
            subplot(2, 2, i);
            imshow(img);
            title(sprintf('Layer: %s, Channel: %d', layer, channel));
        end
    catch
        fprintf('Feature visualization not available for this model configuration.\n');
    end
end

fprintf('Brain tumor classification using MobileNetV2 transfer learning complete. Final test accuracy: %.2f%%\n', accuracy * 100);