

selectedFeatures = [1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16];
selectedFeatures = selectedFeatures([1, 2, 3, 8, 7, 15, 4, 5, 6, 14, 10, 16]);
featureExtractor = @(img) calculateColourFeatures(img, selectedFeatures);


trainingImagePath = fullfile('trainingimages','PCB_image.png');
vegetationMapPath = fullfile('trainingimages','PCB_image_annonated.png');
classificationPath = fullfile('trainingimages','segmentedImage.png');

[classifier, featureMatrix, featureNames] = trainClassifierFromImagePair(trainingImagePath, vegetationMapPath, featureExtractor);


img = readRGBorRGBNimage(trainingImagePath);
res = segmentImage(img, classifier, featureExtractor);

