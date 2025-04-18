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

% Eğitim seçenekleri
options = trainingOptions('adam',...
    'InitialLearnRate', 0.0001,...
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
    'Number of epochs the model trained for' 12
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
[net, trainInfo] = trainNetwork(augmentedImdsTrain, lgraph, options);

% trainInfo
disp('--- trainInfo ---');
disp(trainInfo);
disp('------------------------------------------');

% Modeli kaydetme
save('brain_tumor_model_mobilenetv2.mat', 'net', 'inputSize', 'classNames');

% Validasyon setinde değerlendirme
YPred = classify(net, augmentedImdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
disp(['Validation Accuracy: ', num2str(accuracy*100), '%']);

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
