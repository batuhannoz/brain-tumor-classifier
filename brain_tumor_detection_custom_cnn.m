%% Brain Tumor Classification with CNN in MATLAB - Improved Version
clear all; close all; clc;

%% 1. Set up environment and check data distribution
positiveFolder = fullfile('archive', 'yes');
negativeFolder = fullfile('archive', 'no');

% Create imageDatastore
imds = imageDatastore({positiveFolder, negativeFolder}, ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true);

% Display class distribution
labelCount = countEachLabel(imds);
disp('Original class distribution:');
disp(labelCount);

%% 2. Handle class imbalance using data augmentation
% Calculate class weights for loss function
totalImages = sum(labelCount.Count);
classWeights = totalImages./(2*labelCount.Count);

% Create augmented datastore with balanced augmentation
augmenter = imageDataAugmenter(...
    'RandRotation', [-20 20], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXScale', [0.8 1.2], ...
    'RandYScale', [0.8 1.2]);

inputSize = [224 224 3];

%% 3. Stratified data splitting
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

% Create augmented datastores with different processing
augTrainingDS = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augValidationDS = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

augTestingDS = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

% Get proper class order from training set
classNames = categories(imdsTrain.Labels);

% Calculate weights based on training set distribution
labelCountTrain = countEachLabel(imdsTrain);
classWeights = totalImages./(2*labelCountTrain.Count);

%% 4. Revised CNN architecture with regularization
layers = [
    imageInputLayer(inputSize)
    
    % First convolutional block
    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    
    % Second convolutional block
    convolution2dLayer(3, 64, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    
    % Third convolutional block
    convolution2dLayer(3, 128, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    
    % Fully connected layers
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.6)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer('ClassWeights', classWeights, 'Classes', classNames)
];

%% 5. Modified training options with early stopping
options = trainingOptions('adam',...
    'InitialLearnRate', 0.0001,...
    'MaxEpochs', 10,...
    'MiniBatchSize', 32,...
    'Shuffle', 'every-epoch',...
    'ValidationData', augValidationDS,...
    'ValidationFrequency', 50,...
    'Verbose', true,...
    'Plots', 'training-progress',...
    'ExecutionEnvironment', 'auto',...
    'OutputFcn', @(info)stopIfAccuracyNotImproving(info, 4));

%% 6. Train the model
[net, trainInfo] = trainNetwork(augTrainingDS, layers, options);

%% 7. Evaluate on test set
predictions = classify(net, augTestingDS);
actualLabels = imdsTest.Labels;

% Calculate metrics
confMat = confusionmat(actualLabels, predictions);
accuracy = sum(diag(confMat))/sum(confMat(:));

% Display detailed results
fprintf('Test Accuracy: %.2f%%\n', accuracy*100);
disp('Confusion Matrix:');
disp(confMat);

precision = confMat(1,1)/(confMat(1,1)+confMat(2,1));
recall = confMat(1,1)/(confMat(1,1)+confMat(1,2));
f1Score = 2*(precision*recall)/(precision+recall);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1Score);

%% Helper function for early stopping
function stop = stopIfAccuracyNotImproving(info, patience)
stop = false;
persistent bestLoss
persistent triggerTimes

if info.State == "start"
    bestLoss = inf;
    triggerTimes = 0;
elseif ~isempty(info.ValidationLoss)
    if info.ValidationLoss < bestLoss
        bestLoss = info.ValidationLoss;
        triggerTimes = 0;
    else
        triggerTimes = triggerTimes + 1;
    end
    
    if triggerTimes >= patience
        stop = true;
    end
end
end

%% Modeli Kaydetme
% Model ve gerekli parametreleri kaydet
modelName = 'brain_tumor_model.mat';
save(modelName, 'net', 'inputSize', 'classNames');
fprintf('Model "%s" kaydedildi.\n', modelName);

%% Kaydedilmiş Modeli Yükleme
loadedModel = load(modelName);
net = loadedModel.net;
inputSize = loadedModel.inputSize;
classNames = loadedModel.classNames;

%% Tüm Test Görsellerini Yükleme ve Hazırlama
augTestDS = augmentedImageDatastore(inputSize(1:2), imdsTest,...
    'ColorPreprocessing', 'gray2rgb');

%% Tüm Görseller Üzerinde Tahmin Yapma
fprintf('Tüm test görselleri üzerinde tahmin yapılıyor...\n');
tic;
[predictions, scores] = classify(net, augTestDS);
toc;

%% Accuracy Hesaplama
actualLabels = imdsTest.Labels;
accuracy = mean(predictions == actualLabels);
fprintf('Genel Doğruluk (Accuracy): %.2f%%\n', accuracy*100);

%% Detaylı Performans Metrikleri
% Confusion Matrix
[confMat, order] = confusionmat(actualLabels, predictions);
figure;
confusionchart(confMat, classNames,...
    'Title', 'Detaylı Karışıklık Matrisi',...
    'RowSummary', 'row-normalized',...
    'ColumnSummary', 'column-normalized');

% Precision, Recall, F1 Hesaplama
TP = confMat(1,1);
FP = confMat(2,1);
FN = confMat(1,2);
TN = confMat(2,2);

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1Score = 2*(precision*recall)/(precision + recall);

fprintf('Precision (Tümör Tespit): %.2f\n', precision);
fprintf('Recall (Tümör Tespit): %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1Score);

%% Sınıf Bazında Accuracy
classAccuracy = diag(confMat)./sum(confMat,2);
for i=1:length(classNames)
    fprintf('%s Sınıfı Accuracy: %.2f%%\n', classNames{i}, classAccuracy(i)*100);
end

%% Örnek Görselleştirme
figure('Name', 'Model Tahmin Örnekleri', 'Position', [100 100 1200 800]);
numSamples = min(12, numel(imdsTest.Files));
randIndices = randperm(numel(imdsTest.Files), numSamples);

for i=1:numSamples
    idx = randIndices(i);
    img = readimage(imdsTest, idx);
    
    subplot(3,4,i);
    imshow(img);
    
    % Başlık renk kodlaması
    if predictions(idx) == actualLabels(idx)
        color = 'g';
    else
        color = 'r';
    end
    
    title(sprintf('Tahmin: %s\nGerçek: %s',...
        string(predictions(idx)),...
        string(actualLabels(idx))),...
        'Color', color, 'FontSize', 9);
end

%% ROC Eğrisi (İsteğe Bağlı)
figure;
[rocX, rocY, ~, AUC] = perfcurve(actualLabels, scores(:,1), classNames{1});
plot(rocX, rocY);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Eğrisi (AUC = %.2f)', AUC));
grid on;

%% Sonuçları Kaydetme
results = struct(...
    'Accuracy', accuracy,...
    'Precision', precision,...
    'Recall', recall,...
    'F1_Score', f1Score,...
    'ConfusionMatrix', confMat,...
    'ROC_AUC', AUC);

save('model_performance_results.mat', 'results');
fprintf('Performans metrikleri kaydedildi.\n');