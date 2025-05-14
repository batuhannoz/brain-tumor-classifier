classdef TumorDetectorSegmenterApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure               matlab.ui.Figure
        LoadImageButton        matlab.ui.control.Button
        DetectTumorsButton     matlab.ui.control.Button
        SegmentTumorButton     matlab.ui.control.Button
        UIAxes                 matlab.ui.control.UIAxes
        StatusLabel            matlab.ui.control.Label
    end

    properties (Access = private)
        originalImage          % Stores the loaded image
        processedImage         % Stores the image preprocessed for the model
        yoloDetector           % Stores the loaded YOLOv2 detector
        modelPath = pwd        % Path to the model directory (current directory by default)
        modelFileName = 'trainedYOLOv2TumorDetector.mat' % Name of the model file
        inputSize = [224 224 3]; % Expected input size for the YOLOv2 model
        medSAM                 % MedSAM model for segmentation
        tumorMask              % Stores the segmentation mask
        detectedBbox           % Stores the detected bounding box
        imageEmbeddings        % Stores image embeddings for MedSAM
    end

    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            app.UIFigure.Name = 'Tümör Tespit ve Segmentasyon Uygulaması';
            app.StatusLabel.Text = 'Durum: Model yükleniyor...';
            drawnow; % Update UI
            loadYOLOv2Model(app);
            initializeMedSAM(app);
            if ~isempty(app.yoloDetector) && ~isempty(app.medSAM)
                app.StatusLabel.Text = 'Durum: Modeller yüklendi. Lütfen bir görüntü yükleyin.';
            elseif isempty(app.medSAM)
                app.StatusLabel.Text = 'Durum: MedSAM modeli yüklenemedi. YOLOv2 model yüklendi.';
                app.SegmentTumorButton.Enable = 'off';
            elseif isempty(app.yoloDetector)
                app.StatusLabel.Text = 'Durum: YOLOv2 modeli yüklenemedi. MedSAM model yüklendi.';
                app.DetectTumorsButton.Enable = 'off'; % Disable button if model fails to load
            else
                app.StatusLabel.Text = 'Durum: Modeller yüklenemedi. Lütfen dosyaları kontrol edin.';
                app.DetectTumorsButton.Enable = 'off';
                app.SegmentTumorButton.Enable = 'off';
            end
        end

        % Function to load the YOLOv2 model
        function loadYOLOv2Model(app)
            modelFileFullPath = fullfile(app.modelPath, app.modelFileName);
            if exist(modelFileFullPath, 'file')
                try
                    loadedData = load(modelFileFullPath);
                    if isfield(loadedData, 'detector')
                        app.yoloDetector = loadedData.detector;
                        % Check if the loaded detector has the expected input size property
                        if isprop(app.yoloDetector, 'Network') && ...
                           isprop(app.yoloDetector.Network, 'Layers') && ...
                           ~isempty(app.yoloDetector.Network.Layers) && ...
                           isprop(app.yoloDetector.Network.Layers(1), 'InputSize')
                           % Optionally update app.inputSize if model has a different one.
                           % For this example, we assume it matches the predefined app.inputSize.
                           % modelInputSize = app.yoloDetector.Network.Layers(1).InputSize;
                           % if ~isequal(modelInputSize(1:2), app.inputSize(1:2))
                           %    app.inputSize = modelInputSize;
                           %    disp(['Model input size updated to: ' num2str(app.inputSize)]);
                           % end
                        else
                             % If detector is a yolov2ObjectDetector, it doesn't directly store inputSize like a DAGNetwork.
                             % The inputSize used during training is implicitly handled. We rely on app.inputSize.
                             disp('YOLOv2 detector loaded. Using predefined input size for preprocessing.');
                        end
                        disp(['Model yüklendi: ' app.modelFileName]);
                    else
                        error('Model dosyasında "detector" değişkeni bulunamadı.');
                    end
                catch ME
                    uialert(app.UIFigure, ['Model yükleme hatası: ' ME.message], 'Hata');
                    app.yoloDetector = [];
                end
            else
                uialert(app.UIFigure, ['Model dosyası bulunamadı: ' modelFileFullPath], 'Hata');
                app.yoloDetector = [];
            end
        end
        
        % Function to initialize the MedSAM model
        function initializeMedSAM(app)
            try
                % Check if Medical Imaging Toolbox is available
                if ~exist('medicalSegmentAnythingModel', 'file')
                    warning('Medical Imaging Toolbox ile MedSAM modeli bulunamadı.');
                    app.medSAM = [];
                    return;
                end
                
                % Initialize Medical Segment Anything Model
                app.medSAM = medicalSegmentAnythingModel;
                disp('MedSAM modeli başarıyla yüklendi.');
            catch ME
                warning(['MedSAM modeli yüklenirken hata oluştu: ' ME.message]);
                app.medSAM = [];
            end
        end

        % Function to preprocess the image for the YOLOv2 detector
        function preprocessImageForDetection(app)
            if isempty(app.originalImage)
                uialert(app.UIFigure, 'Önce bir görüntü yükleyin.', 'Hata');
                app.processedImage = [];
                return;
            end

            try
                % Görüntü boyutlarını saklayın ki orijinal görüntüye geri dönüştürebilesiniz
                [height, width, channels] = size(app.originalImage);

                % YOLOv2 modelin beklediği giriş boyutuna göre yeniden boyutlandır
                resizedImage = imresize(app.originalImage, app.inputSize(1:2));

                % Resim RGB değilse RGB'ye dönüştür
                if size(resizedImage, 3) == 1
                    app.processedImage = cat(3, resizedImage, resizedImage, resizedImage);
                elseif size(resizedImage, 3) == 3
                    app.processedImage = resizedImage;
                else
                    error('Desteklenmeyen görüntü formatı: Görüntü gri tonlamalı veya RGB olmalıdır.');
                end

                % Hazırlanan görüntünün boyutlarının model beklentisine uyduğundan emin olun
                if ~isequal(size(app.processedImage), app.inputSize)
                    warning('İşlenen görüntü, modelin beklediği boyutla eşleşmiyor. Boyutlar: %s vs %s', ...
                            mat2str(size(app.processedImage)), mat2str(app.inputSize));
                end

            catch ME
                uialert(app.UIFigure, ['Görüntü ön işleme hatası: ' ME.message], 'Ön İşleme Hatası');
                app.processedImage = [];
            end
        end
        
        % Function to segment tumor using MedSAM
        function segmentTumorWithMedSAM(app)
            if isempty(app.originalImage)
                uialert(app.UIFigure, 'Lütfen önce bir görüntü yükleyin!', 'Uyarı');
                return;
            end
        
            if isempty(app.medSAM)
                uialert(app.UIFigure, 'MedSAM modeli yüklenemedi. Lütfen modeli kontrol edin.', 'Hata');
                return;
            end
            
            if isempty(app.detectedBbox)
                uialert(app.UIFigure, 'Önce tumor tespiti yapmalısınız.', 'Uyarı');
                return;
            end
        
            app.StatusLabel.Text = 'Durum: Tümör segmente ediliyor...';
            drawnow;
        
            try
                % Prepare the image for MedSAM if needed
                imageToSegment = app.originalImage;
                if size(imageToSegment, 3) == 1
                    imageToSegment = cat(3, imageToSegment, imageToSegment, imageToSegment);
                end
                
                % Extract image size for segmentation
                imageSize = size(imageToSegment);
                imageSize = imageSize(1:2);
                
                % Extract embeddings from the image
                app.imageEmbeddings = extractEmbeddings(app.medSAM, imageToSegment);
                
                % Use the detected bounding box as a prompt for segmentation
                boxPrompt = app.detectedBbox;
                
                % Perform segmentation using MedSAM
                app.tumorMask = segmentObjectsFromEmbeddings(app.medSAM, app.imageEmbeddings, imageSize, BoundingBox=boxPrompt);
                
                % Display the segmentation result
                displaySegmentationResult(app);
                
                app.StatusLabel.Text = 'Durum: Tümör segmentasyonu tamamlandı.';
            catch ME
                uialert(app.UIFigure, ['Segmentasyon sırasında hata: ' ME.message], 'Segmentasyon Hatası');
                app.StatusLabel.Text = 'Durum: Segmentasyon hatası.';
                % Show original image in case of error during segmentation
                imshow(app.originalImage, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Segmentasyon Hatası');
            end
        end
        
        % Function to display segmentation results
        function displaySegmentationResult(app)
            if isempty(app.tumorMask) || isempty(app.originalImage)
                return;
            end
            
            % Create a RGB mask for visualization
            redChannel = zeros(size(app.tumorMask), 'uint8');
            greenChannel = zeros(size(app.tumorMask), 'uint8');
            blueChannel = zeros(size(app.tumorMask), 'uint8');
            
            % Set red channel for tumor segmentation (adjust alpha for visibility)
            redChannel(app.tumorMask) = 255;
            
            % Create the segmentation overlay
            segmentationOverlay = cat(3, redChannel, greenChannel, blueChannel);
            
            % Overlay segmentation on original image with transparency
            alpha = 0.3; % Transparency level
            overlayedImage = app.originalImage;
            
            % Apply the mask with transparency
            for c = 1:3
                overlayImage = app.originalImage(:,:,c);
                segmentOverlayChannel = segmentationOverlay(:,:,c);
                overlayImage(app.tumorMask) = uint8(alpha * double(segmentOverlayChannel(app.tumorMask)) + ...
                                                  (1-alpha) * double(overlayImage(app.tumorMask)));
                overlayedImage(:,:,c) = overlayImage;
            end
            
            % Display the result
            imshow(overlayedImage, 'Parent', app.UIAxes);
            
            % Calculate and display segmentation metrics
            tumorArea = sum(app.tumorMask(:));
            totalArea = numel(app.tumorMask);
            percentCoverage = (tumorArea / totalArea) * 100;
            
            title(app.UIAxes, sprintf('Tümör Segmentasyonu (Alan: %.2f%%)', percentCoverage));
        end

        % Button pushed function: LoadImageButton
        function LoadImageButtonPushed(app, event)
            [file, path] = uigetfile({'*.jpg;*.png;*.jpeg;*.tif;*.tiff', 'Görüntü Dosyaları (*.jpg, *.png, *.jpeg, *.tif, *.tiff)'}, 'Bir Görüntü Seçin');
            if file
                fullpath = fullfile(path, file);
                try
                    app.originalImage = imread(fullpath);
                    imshow(app.originalImage, 'Parent', app.UIAxes);
                    title(app.UIAxes, ''); % Clear previous title/annotations
                    app.StatusLabel.Text = 'Durum: Görüntü yüklendi. Tespit için butona basın.';
                catch ME
                    uialert(app.UIFigure, ['Görüntü yüklenirken hata oluştu: ' ME.message], 'Görüntü Yükleme Hatası');
                    app.originalImage = [];
                end
            end
        end

        % Button pushed function: DetectTumorsButton
        function DetectTumorsButtonPushed(app, event)
            if isempty(app.originalImage)
                uialert(app.UIFigure, 'Lütfen önce bir görüntü yükleyin!', 'Uyarı');
                return;
            end

            if isempty(app.yoloDetector)
                uialert(app.UIFigure, 'YOLOv2 modeli yüklenemedi. Lütfen modeli kontrol edin.', 'Hata');
                return;
            end

            app.StatusLabel.Text = 'Durum: Tümör tespit ediliyor...';
            drawnow;

            try
                % Preprocess the image
                preprocessImageForDetection(app);
                if isempty(app.processedImage)
                    app.StatusLabel.Text = 'Durum: Görüntü ön işlenemedi.';
                    return;
                end

                % Detect tumors with initial threshold
                [bboxes, scores, labels] = detect(app.yoloDetector, app.processedImage, 'Threshold', 0.4);
                
                % Display the original image
                imshow(app.originalImage, 'Parent', app.UIAxes);
                hold(app.UIAxes, 'on');
        
                if ~isempty(bboxes)
                    % Sadece en yüksek skorlu tümörü seç
                    [maxScore, maxIdx] = max(scores);
                    bestBbox = bboxes(maxIdx, :);
                    bestLabel = labels(maxIdx);
                    
                    % Tespit edilen kutuyu orijinal görüntü boyutlarına ölçeklendir
                    % Önce ölçek faktörü hesaplanır
                    [origHeight, origWidth, ~] = size(app.originalImage);
                    [procHeight, procWidth, ~] = size(app.processedImage);
                    
                    scaleX = origWidth / procWidth;
                    scaleY = origHeight / procHeight;
                    
                    % Sınırlayıcı kutuyu ölçeklendir
                    scaledBbox = zeros(size(bestBbox));
                    scaledBbox(1) = bestBbox(1) * scaleX;                 % x
                    scaledBbox(2) = bestBbox(2) * scaleY;                 % y
                    scaledBbox(3) = bestBbox(3) * scaleX;                 % width
                    scaledBbox(4) = bestBbox(4) * scaleY;                 % height
                    
                    % Save detected bbox for segmentation
                    app.detectedBbox = scaledBbox;
                    
                    % Enable segment button if MedSAM is available
                    if ~isempty(app.medSAM)
                        app.SegmentTumorButton.Enable = 'on';
                    end
                    
                    % Sadece ölçeklendirilmiş en iyi tespiti görüntüle
                    displayImage = insertObjectAnnotation(app.originalImage, 'rectangle', scaledBbox, cellstr(bestLabel), ...
                                                          'TextBoxOpacity', 0.9, 'FontSize', 16, 'LineWidth', 3, ...
                                                          'Color', 'red');
                    imshow(displayImage, 'Parent', app.UIAxes);
                    title(app.UIAxes, sprintf('Tümör tespit edildi (Güven: %.2f%%)', maxScore*100));
                    app.StatusLabel.Text = 'Durum: Tümör tespit edildi. Segmentasyon için butona basın.';
                else
                    % Disable segment button if no tumor detected
                    app.SegmentTumorButton.Enable = 'off';
                    app.detectedBbox = [];
                    title(app.UIAxes, 'Tümör tespit edilemedi');
                    app.StatusLabel.Text = 'Durum: Tümör tespit edilemedi.';
                end
                hold(app.UIAxes, 'off');

            catch ME
                uialert(app.UIFigure, ['Tespit sırasında hata: ' ME.message], 'Tespit Hatası');
                app.StatusLabel.Text = 'Durum: Tespit hatası.';
                % Show original image in case of error during detection/display
                imshow(app.originalImage, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Tespit Hatası');
            end
        end
        
        % Button pushed function: SegmentTumorButton
        function SegmentTumorButtonPushed(app, event)
            % Call the segmentation method
            segmentTumorWithMedSAM(app);
        end
    end

    % App initialization and construction
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure
            app.UIFigure = uifigure;
            app.UIFigure.Position = [100 100 650 500]; % Adjusted size
            app.UIFigure.Name = 'Tümör Tespit Uygulaması';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            app.UIAxes.Position = [20 80 610 390]; % Adjusted position

            % Create LoadImageButton
            app.LoadImageButton = uibutton(app.UIFigure, 'push');
            app.LoadImageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadImageButtonPushed, true);
            app.LoadImageButton.Position = [20 40 150 30];
            app.LoadImageButton.Text = 'Görüntü Yükle';

            % Create DetectTumorsButton
            app.DetectTumorsButton = uibutton(app.UIFigure, 'push');
            app.DetectTumorsButton.ButtonPushedFcn = createCallbackFcn(app, @DetectTumorsButtonPushed, true);
            app.DetectTumorsButton.Position = [190 40 150 30];
            app.DetectTumorsButton.Text = 'Tümörleri Tespit Et';

            % Create SegmentTumorButton
            app.SegmentTumorButton = uibutton(app.UIFigure, 'push');
            app.SegmentTumorButton.ButtonPushedFcn = createCallbackFcn(app, @SegmentTumorButtonPushed, true);
            app.SegmentTumorButton.Position = [360 40 150 30];
            app.SegmentTumorButton.Text = 'Tümörü Segmente Et';
            app.SegmentTumorButton.Enable = 'off'; % Initially disabled until tumor is detected
        
            % Create StatusLabel
            app.StatusLabel = uilabel(app.UIFigure);
            app.StatusLabel.Position = [20 10 610 22];
            app.StatusLabel.Text = 'Durum: Başlatılıyor...';
        end
    end

    methods (Access = public)

        % Construct app
        function app = TumorDetectorSegmenterApp(varargin)

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute one-time startup logic
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end