classdef BrainTumorClassifierApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure            matlab.ui.Figure
        ModelDropDownLabel  matlab.ui.control.Label
        ModelDropDown       matlab.ui.control.DropDown
        LoadImageButton     matlab.ui.control.Button
        ClassifyButton      matlab.ui.control.Button
        UIAxes              matlab.ui.control.UIAxes
        ResultLabel         matlab.ui.control.Label
        ConfidenceLabel    matlab.ui.control.Label
    end

    properties (Access = private)
        currentModel % Seçilen model
        imageData    % Yüklenen görüntü verisi
        modelNames   % Mevcut model listesi
        modelPath = pwd % Model dosyalarının konumu
    end

    methods (Access = private)
        
        function detectAvailableModels(app)
            % Brain_tumor ile başlayan .mat dosyalarını bul
            files = dir(fullfile(app.modelPath, 'brain_tumor*.mat'));
            app.modelNames = {files.name};
            
            if isempty(app.modelNames)
                uialert(app.UIFigure, 'Model bulunamadı!', 'Hata');
                app.ModelDropDown.Items = {'No models available'};
            else
                app.ModelDropDown.Items = app.modelNames;
                app.ModelDropDown.Value = app.modelNames{1};
                app.loadSelectedModel();
            end
        end
        
        function loadSelectedModel(app)
            % Seçilen modeli yükle
            selectedModel = app.ModelDropDown.Value;
            try
                loadedData = load(fullfile(app.modelPath, selectedModel));
                if isfield(loadedData, 'net')
                    app.currentModel = loadedData.net;
                elseif isfield(loadedData, 'lgraph')
                    app.currentModel = loadedData.lgraph;
                else
                    error('Geçersiz model formatı');
                end
                disp(['Model yüklendi: ' selectedModel]);
            catch ME
                uialert(app.UIFigure, ME.message, 'Model Yükleme Hatası');
            end
        end
        
        function preprocessImage(app)
            % Görüntüyü model giriş boyutuna getir
            targetSize = [224 224 3]; % Varsayılan boyut
            
            if isa(app.currentModel, 'SeriesNetwork') || isa(app.currentModel, 'DAGNetwork')
                inputLayer = app.currentModel.Layers(1);
                targetSize = inputLayer.InputSize;
            end
            
            % Görüntüyü yeniden boyutlandır ve normalize et
            resizedImg = imresize(app.imageData, targetSize(1:2));
            if size(resizedImg,3) == 1 % Grayscale ise RGB'ye çevir
                resizedImg = repmat(resizedImg, 1, 1, 3);
            end
            app.imageData = im2single(resizedImg); % Normalizasyon
        end
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            % Modelleri tara
            detectAvailableModels(app);
            
            % GUI elementlerini ayarla
            app.UIFigure.Name = 'Beyin Tümörü Sınıflandırıcı';
            app.ResultLabel.Text = 'Sonuç: -';
            app.ConfidenceLabel.Text = 'Güven: -';
        end

        % Dropdown değiştiğinde çağrılır
        function ModelDropDownValueChanged(app, event)
            app.loadSelectedModel();
        end

        % Görüntü yükleme butonu
        function LoadImageButtonPushed(app, event)
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'});
            if file
                fullpath = fullfile(path, file);
                try
                    app.imageData = imread(fullpath);
                    imshow(app.imageData, 'Parent', app.UIAxes);
                    app.ResultLabel.Text = 'Sonuç: -';
                    app.ConfidenceLabel.Text = 'Güven: -';
                catch ME
                    uialert(app.UIFigure, ME.message, 'Görüntü Yükleme Hatası');
                end
            end
        end

        % Sınıflandırma butonu
        function ClassifyButtonPushed(app, event)
            if isempty(app.imageData)
                uialert(app.UIFigure, 'Önce görüntü yükleyin!', 'Uyarı');
                return;
            end
            
            if isempty(app.currentModel)
                uialert(app.UIFigure, 'Önce model seçin!', 'Uyarı');
                return;
            end
            
            try
                % Ön işleme
                preprocessImage(app);
                
                % Tahmin yap
                [label, scores] = classify(app.currentModel, app.imageData);
                [~, idx] = max(scores);
                confidence = scores(idx);
                
                % Sonuçları göster
                app.ResultLabel.Text = ['Sonuç: ' char(label)];
                app.ConfidenceLabel.Text = sprintf('Güven: %.2f%%', confidence*100);
                
                % Renk kodlama
                if strcmpi(char(label), 'yes')
                    app.ResultLabel.FontColor = [1 0 0]; % Kırmızı
                else
                    app.ResultLabel.FontColor = [0 0.7 0]; % Yeşil
                end
                
            catch ME
                uialert(app.UIFigure, ME.message, 'Sınıflandırma Hatası');
            end
        end
    end

    % App initialization and construction
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure
            app.UIFigure = uifigure;
            app.UIFigure.Position = [100 100 800 600];
            app.UIFigure.Color = [0.95 0.95 0.95];

            % Create ModelDropDownLabel
            app.ModelDropDownLabel = uilabel(app.UIFigure);
            app.ModelDropDownLabel.HorizontalAlignment = 'right';
            app.ModelDropDownLabel.Position = [50 550 100 22];
            app.ModelDropDownLabel.Text = 'Model Seçin:';

            % Create ModelDropDown
            app.ModelDropDown = uidropdown(app.UIFigure);
            app.ModelDropDown.Position = [165 550 200 22];
            app.ModelDropDown.ValueChangedFcn = createCallbackFcn(app, @ModelDropDownValueChanged, true);

            % Create LoadImageButton
            app.LoadImageButton = uibutton(app.UIFigure, 'push');
            app.LoadImageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadImageButtonPushed, true);
            app.LoadImageButton.Position = [400 545 100 30];
            app.LoadImageButton.Text = 'Görüntü Yükle';

            % Create ClassifyButton
            app.ClassifyButton = uibutton(app.UIFigure, 'push');
            app.ClassifyButton.ButtonPushedFcn = createCallbackFcn(app, @ClassifyButtonPushed, true);
            app.ClassifyButton.Position = [520 545 100 30];
            app.ClassifyButton.Text = 'Sınıflandır';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            app.UIAxes.Position = [50 150 700 380];

            % Create ResultLabel
            app.ResultLabel = uilabel(app.UIFigure);
            app.ResultLabel.FontSize = 16;
            app.ResultLabel.Position = [50 100 300 30];
            app.ResultLabel.Text = 'Sonuç: -';

            % Create ConfidenceLabel
            app.ConfidenceLabel = uilabel(app.UIFigure);
            app.ConfidenceLabel.FontSize = 16;
            app.ConfidenceLabel.Position = [50 70 300 30];
            app.ConfidenceLabel.Text = 'Güven: -';
        end
    end

    methods (Access = public)

        % Construct app
        function app = BrainTumorClassifierApp
            % Create and configure components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end
    end
end