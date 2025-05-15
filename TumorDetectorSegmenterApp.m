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
        originalImage
        processedImage
        yoloDetector
        modelPath = pwd
        modelFileName = 'trainedYOLOv2TumorDetector.mat'
        inputSize = [224 224 3];
        medSAM
        tumorMask
        detectedBbox
        imageEmbeddings
    end

    methods (Access = private)

        function startupFcn(app)
            app.UIFigure.Name = 'Tümör Tespit ve Segmentasyon Uygulaması';
            app.StatusLabel.Text = 'Durum: Model yükleniyor...';
            drawnow;
            loadYOLOv2Model(app);
            initializeMedSAM(app);
            if ~isempty(app.yoloDetector) && ~isempty(app.medSAM)
                app.StatusLabel.Text = 'Durum: Modeller yüklendi. Lütfen bir görüntü yükleyin.';
            elseif isempty(app.medSAM)
                app.StatusLabel.Text = 'Durum: MedSAM modeli yüklenemedi. YOLOv2 model yüklendi.';
                app.SegmentTumorButton.Enable = 'off';
            elseif isempty(app.yoloDetector)
                app.StatusLabel.Text = 'Durum: YOLOv2 modeli yüklenemedi. MedSAM model yüklendi.';
                app.DetectTumorsButton.Enable = 'off';
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
                        if isprop(app.yoloDetector, 'Network') && ...
                           isprop(app.yoloDetector.Network, 'Layers') && ...
                           ~isempty(app.yoloDetector.Network.Layers) && ...
                           isprop(app.yoloDetector.Network.Layers(1), 'InputSize')
                        else
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
        
        function initializeMedSAM(app)
            try
                if ~exist('medicalSegmentAnythingModel', 'file')
                    warning('Medical Imaging Toolbox ile MedSAM modeli bulunamadı.');
                    app.medSAM = [];
                    return;
                end
                app.medSAM = medicalSegmentAnythingModel;
                disp('MedSAM modeli başarıyla yüklendi.');
            catch ME
                warning(['MedSAM modeli yüklenirken hata oluştu: ' ME.message]);
                app.medSAM = [];
            end
        end

        function preprocessImageForDetection(app)
            if isempty(app.originalImage)
                uialert(app.UIFigure, 'Önce bir görüntü yükleyin.', 'Hata');
                app.processedImage = [];
                return;
            end

            try
                [height, width, channels] = size(app.originalImage);

                resizedImage = imresize(app.originalImage, app.inputSize(1:2));

                if size(resizedImage, 3) == 1
                    app.processedImage = cat(3, resizedImage, resizedImage, resizedImage);
                elseif size(resizedImage, 3) == 3
                    app.processedImage = resizedImage;
                else
                    error('Desteklenmeyen görüntü formatı: Görüntü gri tonlamalı veya RGB olmalıdır.');
                end

                if ~isequal(size(app.processedImage), app.inputSize)
                    warning('İşlenen görüntü, modelin beklediği boyutla eşleşmiyor. Boyutlar: %s vs %s', ...
                            mat2str(size(app.processedImage)), mat2str(app.inputSize));
                end

            catch ME
                uialert(app.UIFigure, ['Görüntü ön işleme hatası: ' ME.message], 'Ön İşleme Hatası');
                app.processedImage = [];
            end
        end
        
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
                
                imageSize = size(imageToSegment);
                imageSize = imageSize(1:2);
                
                app.imageEmbeddings = extractEmbeddings(app.medSAM, imageToSegment);
                
                boxPrompt = app.detectedBbox;

                app.tumorMask = segmentObjectsFromEmbeddings(app.medSAM, app.imageEmbeddings, imageSize, BoundingBox=boxPrompt);
                
                displaySegmentationResult(app);
                
                app.StatusLabel.Text = 'Durum: Tümör segmentasyonu tamamlandı.';
            catch ME
                uialert(app.UIFigure, ['Segmentasyon sırasında hata: ' ME.message], 'Segmentasyon Hatası');
                app.StatusLabel.Text = 'Durum: Segmentasyon hatası.';

                imshow(app.originalImage, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Segmentasyon Hatası');
            end
        end
        
        % Function to display segmentation results
        function displaySegmentationResult(app)
            if isempty(app.tumorMask) || isempty(app.originalImage)
                return;
            end

            redChannel = zeros(size(app.tumorMask), 'uint8');
            greenChannel = zeros(size(app.tumorMask), 'uint8');
            blueChannel = zeros(size(app.tumorMask), 'uint8');
            
            redChannel(app.tumorMask) = 255;
            
            segmentationOverlay = cat(3, redChannel, greenChannel, blueChannel);
            
            alpha = 0.3;
            overlayedImage = app.originalImage;
            
            for c = 1:3
                overlayImage = app.originalImage(:,:,c);
                segmentOverlayChannel = segmentationOverlay(:,:,c);
                overlayImage(app.tumorMask) = uint8(alpha * double(segmentOverlayChannel(app.tumorMask)) + ...
                                                  (1-alpha) * double(overlayImage(app.tumorMask)));
                overlayedImage(:,:,c) = overlayImage;
            end
            
            imshow(overlayedImage, 'Parent', app.UIAxes);
            
            tumorArea = sum(app.tumorMask(:));
            totalArea = numel(app.tumorMask);
            percentCoverage = (tumorArea / totalArea) * 100;
            
            title(app.UIAxes, sprintf('Tümör Segmentasyonu (Alan: %.2f%%)', percentCoverage));
        end

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

                [bboxes, scores, labels] = detect(app.yoloDetector, app.processedImage, 'Threshold', 0.4);
                
                imshow(app.originalImage, 'Parent', app.UIAxes);
                hold(app.UIAxes, 'on');
        
                if ~isempty(bboxes)
                    % Sadece en yüksek skorlu tümörü seç
                    [maxScore, maxIdx] = max(scores);
                    bestBbox = bboxes(maxIdx, :);
                    bestLabel = labels(maxIdx);

                    [origHeight, origWidth, ~] = size(app.originalImage);
                    [procHeight, procWidth, ~] = size(app.processedImage);
                    
                    scaleX = origWidth / procWidth;
                    scaleY = origHeight / procHeight;
                    
                    scaledBbox = zeros(size(bestBbox));
                    scaledBbox(1) = bestBbox(1) * scaleX;
                    scaledBbox(2) = bestBbox(2) * scaleY;
                    scaledBbox(3) = bestBbox(3) * scaleX;
                    scaledBbox(4) = bestBbox(4) * scaleY;
                    
                    app.detectedBbox = scaledBbox;
                    
                    if ~isempty(app.medSAM)
                        app.SegmentTumorButton.Enable = 'on';
                    end
                    
                    displayImage = insertObjectAnnotation(app.originalImage, 'rectangle', scaledBbox, cellstr(bestLabel), ...
                                                          'TextBoxOpacity', 0.9, 'FontSize', 16, 'LineWidth', 3, ...
                                                          'Color', 'red');
                    imshow(displayImage, 'Parent', app.UIAxes);
                    title(app.UIAxes, sprintf('Tümör tespit edildi (Güven: %.2f%%)', maxScore*100));
                    app.StatusLabel.Text = 'Durum: Tümör tespit edildi. Segmentasyon için butona basın.';
                else
                    app.SegmentTumorButton.Enable = 'off';
                    app.detectedBbox = [];
                    title(app.UIAxes, 'Tümör tespit edilemedi');
                    app.StatusLabel.Text = 'Durum: Tümör tespit edilemedi.';
                end
                hold(app.UIAxes, 'off');

            catch ME
                uialert(app.UIFigure, ['Tespit sırasında hata: ' ME.message], 'Tespit Hatası');
                app.StatusLabel.Text = 'Durum: Tespit hatası.';
                imshow(app.originalImage, 'Parent', app.UIAxes);
                title(app.UIAxes, 'Tespit Hatası');
            end
        end
        
        function SegmentTumorButtonPushed(app, event)
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
            app.SegmentTumorButton.Enable = 'off';
        
            % Create StatusLabel
            app.StatusLabel = uilabel(app.UIFigure);
            app.StatusLabel.Position = [20 10 610 22];
            app.StatusLabel.Text = 'Durum: Başlatılıyor...';
        end
    end

    methods (Access = public)
        function app = TumorDetectorSegmenterApp(varargin)
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
            if nargout == 0
                clear app
            end
        end

        function delete(app)
            delete(app.UIFigure)
        end
    end
end