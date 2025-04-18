positiveFolder = fullfile('archive', 'yes');
negativeFolder = fullfile('archive', 'no');

% Create imageDatastore
imds = imageDatastore({positiveFolder, negativeFolder}, ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true);

% Veriyi eğitim, validasyon ve test setlerine ayırma
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

% Data Augmentation tanımlama
augmenter = imageDataAugmenter(...
    'RandRotation',[-100 100],...
    'RandXReflection',true,...
    'RandYReflection',true,...
    'RandXScale',[0.8 1.2],...
    'RandYScale',[0.8 1.2]);

% Görüntü boyutları
inputSize = [224 224 3];

% Augmented image datastore'ları oluşturma
augmentedImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain,...
    'DataAugmentation',augmenter,...
    'ColorPreprocessing','gray2rgb');

augmentedImdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation,...
    'ColorPreprocessing','gray2rgb');

% MobileNetV2 modelini yükleme
net = mobilenetv2;

% Layer graph oluşturma
lgraph = layerGraph(net);

% Orijinal sınıflandırma katmanlarını kaldırma
lgraph = removeLayers(lgraph, {'Logits', 'Logits_softmax', 'ClassificationLayer_Logits'});

% Yeni katmanları ekleme
numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc')
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classification')
];

lgraph = addLayers(lgraph, newLayers);

% Katmanları bağlama
lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'new_fc');

% Eğitim seçenekleri
options = trainingOptions('adam',...
    'InitialLearnRate', 0.0001,...
    'MaxEpochs', 10,...
    'Shuffle','every-epoch',...
    'ValidationData',augmentedImdsValidation,...
    'ValidationFrequency',50,...
    'Verbose',true,...
    'Plots','training-progress',...
    'MiniBatchSize', 16);

% Modeli eğitme
[net, trainInfo] = trainNetwork(augmentedImdsTrain, lgraph, options);

% Modeli kaydetme
save('brain_tumor_model_mobilenetv2.mat', 'net', 'inputSize', 'classNames');

% Validasyon setinde değerlendirme
YPred = classify(net, augmentedImdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
disp(['Validation Accuracy: ', num2str(accuracy*100), '%']);