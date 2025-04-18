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