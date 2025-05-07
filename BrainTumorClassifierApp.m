classdef BrainTumorClassifierApp < matlab.apps.AppBase

    properties (Access = public)
        UIFigure            matlab.ui.Figure
        ModelDropDownLabel  matlab.ui.control.Label
        ModelDropDown       matlab.ui.control.DropDown
        LoadImageButton     matlab.ui.control.Button
        ClassifyButton      matlab.ui.control.Button
        UIAxes              matlab.ui.control.UIAxes
        ResultLabel         matlab.ui.control.Label
        ConfidenceLabel     matlab.ui.control.Label
    end

    properties (Access = private)
        currentModel
        originalImage   
        processedImage  
        modelNames
        modelPath = pwd
    end

    methods (Access = private)
        
        function detectAvailableModels(app)
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
            selectedModel = app.ModelDropDown.Value;
            try
                loadedData = load(fullfile(app.modelPath, selectedModel));
                if isfield(loadedData, 'trainedNet') % Eğitim kodundaki değişken adı
                    app.currentModel = loadedData.trainedNet;
                elseif isfield(loadedData, 'net')
                    app.currentModel = loadedData.net;
                else
                    error('Geçersiz model formatı');
                end
                disp(['Model yüklendi: ' selectedModel]);
            catch ME
                uialert(app.UIFigure, ME.message, 'Model Yükleme Hatası');
            end
        end
        
        function preprocessImage(app)
            if isempty(app.currentModel) || ~isprop(app.currentModel, 'Layers')
                uialert(app.UIFigure, 'Model not loaded or invalid.', 'Error');
                app.processedImage = []; 
                return;
            end

             if isempty(app.originalImage)
                 uialert(app.UIFigure, 'Load an image first.', 'Error');
                 app.processedImage = []; 
                 return;
             end

            try
                inputSize = app.currentModel.Layers(1).InputSize;

                if numel(inputSize) < 2
                   error('Invalid model input size detected.');
                end

                targetSize = inputSize(1:2);

                resizedImage = imresize(app.originalImage, targetSize);

                if size(resizedImage, 3) == 1
                    app.processedImage = cat(3, resizedImage, resizedImage, resizedImage);
                elseif size(resizedImage, 3) == 3
                    app.processedImage = resizedImage;
                else
                    error('Unsupported image format: Image must be grayscale or RGB.');
                end

            catch ME
                uialert(app.UIFigure, ['Image preprocessing failed: ' ME.message], 'Preprocessing Error');
                app.processedImage = []; % Clear processed image on error to prevent issues
            end
        end
    end

    methods (Access = private)

        function startupFcn(app)
            detectAvailableModels(app);
            app.UIFigure.Name = 'Beyin Tümörü Sınıflandırıcı';
            app.ResultLabel.Text = 'Sonuç: -';
            app.ConfidenceLabel.Text = 'Güven: -';
        end

        function ModelDropDownValueChanged(app, event)
            app.loadSelectedModel();
        end

        function LoadImageButtonPushed(app, event)
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg;*.tif;*.tiff', 'Image Files'});
            if file
                fullpath = fullfile(path, file);
                try
                    app.originalImage = imread(fullpath);
                    imshow(app.originalImage, 'Parent', app.UIAxes);
                    app.ResultLabel.Text = 'Result: -';
                    app.ConfidenceLabel.Text = 'Confidence: -';
                catch ME
                    uialert(app.UIFigure, ME.message, 'Image Load Error');
                end
            end
        end

        function ClassifyButtonPushed(app, event)
            if isempty(app.originalImage)
                uialert(app.UIFigure, 'Load an image first!', 'Warning');
                return;
            end

            try
                % Preprocess and classify
                preprocessImage(app);
                [label, scores] = classify(app.currentModel, app.processedImage);
                confidence = max(scores);

                % Display results
                app.ResultLabel.Text = ['Result: ' char(label)];
                app.ConfidenceLabel.Text = sprintf('Confidence: %.2f%%', confidence*100);
                imshow(app.originalImage, 'Parent', app.UIAxes); % Show original image
            catch ME
                uialert(app.UIFigure, ME.message, 'Classification Error');
            end
        end
    end

    % App initialization and construction
    methods (Access = private)

        function createComponents(app)
            app.UIFigure = uifigure;
            app.UIFigure.Position = [100 100 800 600];
            app.UIFigure.Color = [0.95 0.95 0.95];

            app.ModelDropDownLabel = uilabel(app.UIFigure);
            app.ModelDropDownLabel.HorizontalAlignment = 'right';
            app.ModelDropDownLabel.Position = [50 550 100 22];
            app.ModelDropDownLabel.Text = 'Model Seçin:';

            app.ModelDropDown = uidropdown(app.UIFigure);
            app.ModelDropDown.Position = [165 550 200 22];
            app.ModelDropDown.ValueChangedFcn = createCallbackFcn(app, @ModelDropDownValueChanged, true);

            app.LoadImageButton = uibutton(app.UIFigure, 'push');
            app.LoadImageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadImageButtonPushed, true);
            app.LoadImageButton.Position = [400 545 100 30];
            app.LoadImageButton.Text = 'Görüntü Yükle';

            app.ClassifyButton = uibutton(app.UIFigure, 'push');
            app.ClassifyButton.ButtonPushedFcn = createCallbackFcn(app, @ClassifyButtonPushed, true);
            app.ClassifyButton.Position = [520 545 100 30];
            app.ClassifyButton.Text = 'Sınıflandır';

            app.UIAxes = uiaxes(app.UIFigure);
            app.UIAxes.Position = [50 150 700 380];

            app.ResultLabel = uilabel(app.UIFigure);
            app.ResultLabel.FontSize = 16;
            app.ResultLabel.Position = [50 100 300 30];
            app.ResultLabel.Text = 'Sonuç: -';

            app.ConfidenceLabel = uilabel(app.UIFigure);
            app.ConfidenceLabel.FontSize = 16;
            app.ConfidenceLabel.Position = [50 70 300 30];
            app.ConfidenceLabel.Text = 'Güven: -';
        end
    end

    methods (Access = public)
        function app = BrainTumorClassifierApp
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
            if nargout == 0
                clear app
            end
        end
    end
end