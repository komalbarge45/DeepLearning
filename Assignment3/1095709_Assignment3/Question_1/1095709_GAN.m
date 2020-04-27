downloadFolder = "C:\Users\bargek\Documents\cifar-100-matlab\CIFAR-100\TRAIN";

disp("Load the cifar100 'apple' folder images to train in GAN network.");
%This dataset of apple images has been picked up from cifar100
datasetFolder = fullfile(downloadFolder,'apple');
%Image datastore for apple images
imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%Augment(Preprocess) the data images to 64*64*3
%Resample out of bound points, resizing, flipping of images
disp('Augment the data images by resizing, flipping images');
augmenter = imageDataAugmenter( ...
    'FillValue', 3, ...
    'RandXReflection',true, ...
    'RandScale',[1 2]);
augimds = augmentedImageDatastore([64 64],imds,'DataAugmentation',augmenter);

%% Generator network
filterSize = [4 4];
numFilters = 100;
numLatentInputs = 50;

disp("Configuration of Generator to generate images from random values");
layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','input')
    transposedConv2dLayer(filterSize,8*numFilters,'Name','transposedConv1')
    batchNormalizationLayer('Name','batchNorm1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,4*numFilters,'Stride',2,'Cropping',1,'Name','transposedConv2')
    batchNormalizationLayer('Name','batchNorm2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',2,'Cropping',1,'Name','transposedConv3')
    batchNormalizationLayer('Name','batchNorm3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping',1,'Name','transposedConv4')
    batchNormalizationLayer('Name','batchNorm4')
    reluLayer('Name','relu4')
    transposedConv2dLayer(filterSize,3,'Stride',2,'Cropping',1,'Name','transposedConv5')
    tanhLayer('Name','tanhLayer')];

disp(layersGenerator);
lgraphGenerator = layerGraph(layersGenerator);

%Train a network with a custom training loop and enable automatic differentiation
dlnetGenerator = dlnetwork(lgraphGenerator);

%% Discriminator network
disp("Configuration of Discriminator that classifies real and generated images");
scale = 0.3;
layersDiscriminator = [
    imageInputLayer([100 100 3],'Normalization','none','Name','input')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding',1,'Name','conv1')
    leakyReluLayer(scale,'Name','leakyrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding',1,'Name','conv2')
    batchNormalizationLayer('Name','batchnorm1')
    leakyReluLayer(scale,'Name','leakyrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding',1,'Name','conv3')
    batchNormalizationLayer('Name','batchnorm2')
    leakyReluLayer(scale,'Name','leakyrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding',1,'Name','conv4')
    batchNormalizationLayer('Name','batchnorm3')
    leakyReluLayer(scale,'Name','leakyrelu4')
    convolution2dLayer(filterSize,1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);
disp(layersDiscriminator);

dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

numEpochs = 1000;
miniBatchSize = 200;
augimds.MiniBatchSize = miniBatchSize;
learnRateGenerator = 0.0002;
learnRateDiscriminator = 0.0001;

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

gradientDecayFactor = 0.45;
squaredGradientDecayFactor = 0.999;

executionEnvironment = "gpu";

ZValidation = randn(1,1,numLatentInputs,64,'single');
dlZValidation = dlarray(ZValidation,'SSCB');
dlZValidation = gpuArray(dlZValidation);

figure
iteration = 0;
start = tic;
disp('Train images through GAN network');
% Loop over epochs.
for i = 1:numEpochs
    
    % Reset and shuffle augemented datastore.
    reset(augimds);
    augimds = shuffle(augimds);
    
    % Loop over batches.
    while hasdata(augimds)
        iteration = iteration + 1;
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
        
        % Concatenate mini-batch of data and generate latent inputs for the
        % generator network.
        X = cat(4,data{:,1}{:});
        Z = randn(1,1,numLatentInputs,size(X,4),'single');
        
        % Normalize the images
        X = (single(X)/255)*2 - 1;
        
        % Convert mini-batch of data to dlarray specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        
        % Convert data to gpuArray.
        dlX = gpuArray(dlX);
        dlZ = gpuArray(dlZ);
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator.Learnables, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRateDiscriminator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator.Learnables,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every 100 iterations, display batch of generated images using the
        % held-out generator input.
        if mod(iteration,100) == 0 || iteration == 1
            
            % Generate images using the held-out generator input.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
            
            % Rescale the images in the range [0 1] and display the images.
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            image(I)
            
            % Update the title with training progress information.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            drawnow
        end
    end
end

%% *********************
% Generate new images as discriminator has learnt strong features that 
% identifies real images and Generator learnt to generate real looking
% images.
%% *********************
disp('Generate new images');
ZNew = randn(1,1,numLatentInputs,16,'single');
dlZNew = dlarray(ZNew,'SSCB');
dlZNew = gpuArray(dlZNew);

dlXGeneratedNew = predict(dlnetGenerator,dlZNew);

I = imtile(extractdata(dlXGeneratedNew));
I = rescale(I);
image(I)
title("Generated Images")

%% Model gradient function

function [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ)

    % Calculate the predictions for real data with the discriminator network.
    dlYPred = forward(dlnetDiscriminator, dlX);

    % Calculate the predictions for generated data with the discriminator network.
    [dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
    dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

    % Calculate the GAN loss
    [lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated);

    % For each network, calculate the gradients with respect to the loss.
    gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
    gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end

function [lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated)

    % Calculate losses for the discriminator network.
    lossGenerated = -mean(log(1-sigmoid(dlYPredGenerated)));
    lossReal = -mean(log(sigmoid(dlYPred)));

    % Combine the losses for the discriminator network.
    lossDiscriminator = lossReal + lossGenerated;

    % Calculate the loss for the generator network.
    lossGenerator = -mean(log(sigmoid(dlYPredGenerated)));

end