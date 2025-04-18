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

% Analyze Network Architecture (Opens a new window)
disp('Analyzing network architecture...');
analyzeNetwork(layers);
disp('Network analysis window opened.');

% Open Network in Deep Network Designer (Opens the App)
disp('Opening Deep Network Designer...');
try
    deepNetworkDesigner(layers);
    disp('Deep Network Designer launched with the defined layers.');
    disp('Use the "Analyze" button within the app for further details.');
catch ME
    warning('Could not automatically open Deep Network Designer. Error: %s', ME.message);
    disp('You can still manually open Deep Network Designer from the MATLAB Apps tab and import the `layers` variable.');
end

% Eğitim seçenekleri
options = trainingOptions('adam',...
    'InitialLearnRate',0.0001,...
    'MaxEpochs', 12,...
    'Shuffle','every-epoch',...
    'ValidationData',augmentedImdsValidation,...
    'ValidationFrequency',50,...
    'Verbose',true,...
    'Plots','training-progress',...
    'MiniBatchSize', 16);

% Display Training Options (Hyperparameters) in Command Window
disp('--- Training Options (Hyperparameters) ---');
disp(options);
% Hiperparametre tablosu oluşturma
hyperParams = {
    'Loss function'                         'Categorical Cross-Entropy'
    'Number of epochs the model trained for' 10
    'Batch size'                            64
    'Optimization algorithm'                'Adam'
    'Learning rate'                         '5 × 10⁻³'
    'Activation function of the Dense layer' 'Softmax'
    'Number of units of the Dense layer'     10
    'Validation set fraction'                0.2
};

% Tablo için figure penceresi
f3 = figure('Name','Hiperparametreler','Position',[100 100 600 250]);
uit = uitable(f3);
uit.Data = hyperParams;
uit.ColumnName = {'Hyper-parameter', 'Value'};
uit.Position = [0 0 600 250];
uit.ColumnWidth = {400 150};

% Hücre içeriklerini ortala
uit.ColumnFormat = {'char', 'char'};
uit.RowStriping = 'on';

% Tablo başlık ayarı
uit.FontName = 'Consolas';
uit.FontSize = 12;
disp('------------------------------------------');

% Modeli eğitme
[net, trainInfo] = trainNetwork(augmentedImdsTrain, layers, options);

save('brain_tumor_model.mat', 'net', 'inputSize', 'classNames');

% Validasyon setinde test etme
YPred = classify(net, augmentedImdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
disp(['Validation Accuracy: ', num2str(accuracy*100), '%'])

% Test seti için ön işleme (görüntü boyutlandırma ve gri -> RGB dönüşüm)
augmentedImdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing','gray2rgb');

% Test verisinde sınıflandırma
YPredTest = classify(net, augmentedImdsTest);
YTest = imdsTest.Labels;

% Doğruluk hesaplama
testAccuracy = sum(YPredTest == YTest)/numel(YTest);
disp(['Test Accuracy: ', num2str(testAccuracy*100), '%']);

% Konfüzyon matrisi oluşturma
confMat = confusionmat(YTest, YPredTest);

% Konfüzyon matrisini görselleştirme
figure;
confusionchart(YTest, YPredTest, ...
    'Title', 'Confusion Matrix for Test Data', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

% Konfüzyon matrisini tablo olarak yazdırma
disp('Confusion Matrix Table:');
confTable = array2table(confMat, ...
    'VariableNames', strcat("Pred_", string(classNames)), ...
    'RowNames', strcat("Actual_", string(classNames)));
disp(confTable);
