positiveFolder = fullfile('archive', 'yes');
negativeFolder = fullfile('archive', 'no');

% Create imageDatastore
imds = imageDatastore({positiveFolder, negativeFolder}, ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true);

% Veriyi eğitim ve validasyon setlerine ayırma (70%-15%-15%)
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

% Data Augmentation tanımlama
augmenter = imageDataAugmenter(...
    'RandRotation',[-100 100],...
    'RandXReflection',true,...
    'RandYReflection',true,...
    'RandXScale',[0.8 1.2],...
    'RandYScale',[0.8 1.2]);

% Görüntü boyutlarını 224x224'e ayarlama ve augmentation uygulama
inputSize = [224 224 3]; % RGB formatı için 3 kanal

% Eğitim verisi için augmented image datastore
augmentedImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain,...
    'DataAugmentation',augmenter,...
    'ColorPreprocessing','gray2rgb'); % Gri görüntüleri RGB'ye çevirme

% Validasyon verisi için preprocessing (augmentation yok)
augmentedImdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation,...
    'ColorPreprocessing','gray2rgb');

classNames = categories(imdsTrain.Labels);

layers = [
    imageInputLayer(inputSize) % 224x224x3 giriş

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer('Classes', classNames)
   ];

% Eğitim seçenekleri
options = trainingOptions('adam',...
    'InitialLearnRate',0.0001,...
    'MaxEpochs', 10,...
    'Shuffle','every-epoch',...
    'ValidationData',augmentedImdsValidation,...
    'ValidationFrequency',50,...
    'Verbose',true,...
    'Plots','training-progress',...
    'MiniBatchSize', 16);

% Modeli eğitme
[net, trainInfo] = trainNetwork(augmentedImdsTrain, layers, options);

save('brain_tumor_model.mat', 'net', 'inputSize', 'classNames');

% Validasyon setinde test etme
YPred = classify(trainedNet,augmentedImdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
disp(['Validation Accuracy: ', num2str(accuracy*100), '%'])