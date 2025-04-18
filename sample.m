% Load the model with the correct variable name (replace 'net' if necessary)
% loadedModel = load('brain_tumor_model.mat');
loadedModel = load('brain_tumor_model_mobilenetv2.mat');
trainedNet = loadedModel.net; % Adjust based on the actual variable name

% Tum verileri yükleme
imdsAll = imageDatastore({'archive/yes','archive/no'},...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% Preprocessing ayarları
inputSize = [224 224 3];
augmentedImdsAll = augmentedImageDatastore(inputSize(1:2), imdsAll,...
    'ColorPreprocessing','gray2rgb');

% Tahminleri yapma
YPredAll = classify(trainedNet, augmentedImdsAll);

% ... (rest of your code remains the same)

YTrueAll = imdsAll.Labels;

% Confusion Matrix oluşturma
figure;
cm = confusionchart(YTrueAll, YPredAll,...
    'Title','Tüm Veri Seti için Confusion Matrix',...
    'RowSummary','row-normalized',...
    'ColumnSummary','column-normalized');

% Accuracy hesaplama
accuracyAll = sum(YPredAll == YTrueAll)/numel(YTrueAll);
disp(['Tüm Veri Seti Accuracy: ', num2str(accuracyAll*100), '%'])

% Precision, Recall ve F1-Score hesaplama
[confMat,order] = confusionmat(YTrueAll, YPredAll);

precision = diag(confMat)./sum(confMat,2)';
recall = diag(confMat)./sum(confMat,1)';
f1Scores = 2*(precision.*recall)./(precision+recall);


% Tüm resimler için tahminleri ve isimleri listeleyelim
disp('=== Resimler ve Tahminleri ===');
for i = 1:numel(imdsAll.Files)
    % Resim dosya adını al
    [~, fileName] = fileparts(imdsAll.Files{i});
    
    % Tahmini ve gerçek etiketi al
    predictedLabel = char(YPredAll(i));
    
    % Konsola yazdır
    fprintf('%d. Resim: %-25s - Tahmin: %s\n', i, fileName, predictedLabel);
end


% Her iki modeli de yükle
modelFiles = {'brain_tumor_model.mat', 'brain_tumor_model_mobilenetv2.mat'};
modelNames = {'Original Model', 'MobileNetV2 Model'};
numModels = numel(modelFiles);

% Veri kümesini yükle
imdsAll = imageDatastore({'archive/yes','archive/no'},...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

inputSize = [224 224 3];
augmentedImdsAll = augmentedImageDatastore(inputSize(1:2), imdsAll,...
    'ColorPreprocessing','gray2rgb');

% Sonuçları saklamak için yapı
results = struct();

for i = 1:numModels
    % Modeli yükle
    loadedModel = load(modelFiles{i});
    currentNet = loadedModel.net;
    
    % Tahminleri yap
    YPredAll = classify(currentNet, augmentedImdsAll);
    YTrueAll = imdsAll.Labels;
    
    % Accuracy hesapla
    accuracy = sum(YPredAll == YTrueAll)/numel(YTrueAll);
    
    % Confusion matrix
    [confMat, order] = confusionmat(YTrueAll, YPredAll);
    
    % Precision, Recall ve F1 Hesapla (DÜZELTİLMİŞ)
    precision = diag(confMat)./sum(confMat,1)'; % TP/(TP+FP)
    recall = diag(confMat)./sum(confMat,2);     % TP/(TP+FN)
    f1Scores = 2*(precision.*recall)./(precision + recall);
    
    % Metrikleri ata (sınıf sıralaması: 'no', 'yes')
    results(i).Model = modelNames{i};
    results(i).Accuracy = accuracy;
    results(i).Precision_Yes = precision(2);
    results(i).Precision_No = precision(1);
    results(i).Recall_Yes = recall(2);
    results(i).Recall_No = recall(1);
    results(i).F1_Yes = f1Scores(2);
    results(i).F1_No = f1Scores(1);
    results(i).Macro_Precision = mean(precision);
    results(i).Macro_Recall = mean(recall);
    results(i).Macro_F1 = mean(f1Scores);
    
    % Confusion Matrix görselleştirme
    figure;
    confusionchart(YTrueAll, YPredAll,...
        'Title',[modelNames{i} ' Confusion Matrix'],...
        'RowSummary','row-normalized',...
        'ColumnSummary','column-normalized');
end