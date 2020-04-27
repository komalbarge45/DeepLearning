%% Define layers for scratch network model
layers =[imageInputLayer([227 227 3])
    convolution2dLayer(11,96,'Stride',4,'BiasLearnRateFactor', 2)  %%filterSize = 11, NumFilters = 96
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(3,'Stride',2)  %% PoolSize = 3
    groupedConvolution2dLayer(5,128,2, 'Padding', 2,'BiasLearnRateFactor', 2); %%filterSize = 5, NumFilters = 128 NumberOfFilterGroups = 2
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer(3,384, 'NumChannels', 256, 'Padding', 1,'BiasLearnRateFactor', 2)   %%filterSize = 3, NumFliters=384
    reluLayer
    groupedConvolution2dLayer(3,192,2, 'Padding', 1,'BiasLearnRateFactor', 2);
    reluLayer
    groupedConvolution2dLayer(3,128,2, 'Padding', 1,'BiasLearnRateFactor', 2);
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    fullyConnectedLayer(4096,'BiasLearnRateFactor', 2) %%NumOfNeurons = 4096
    reluLayer
    dropoutLayer
    fullyConnectedLayer(4096,'BiasLearnRateFactor', 2)
    reluLayer
    dropoutLayer
    fullyConnectedLayer(100,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Load images for CIFAR100 dataset
Ximds = imageDatastore('CIFAR-100\TRAIN\','IncludeSubfolders',true,'LabelSource','foldernames');
Xtestimds = imageDatastore('CIFAR-100\TEST\','IncludeSubfolders',true,'LabelSource','foldernames');
%% Convert images into AlexNet input data format
X = augmentedImageDatastore([227 227],Ximds);
Xtest = augmentedImageDatastore([227 227],Xtestimds);

%% Options to be provided for network
miniBatchSize = 64;
valFrequency = floor(numel(X.Files)/miniBatchSize);
opts = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',2e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',Xtest, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'ExecutionEnvironment','auto', ...
    'Plots','training-progress');
start = cputime;

%% Train network
network = trainNetwork(X, layers, opts);
endtime=cputime;
predictedlabels = classify(network, Xtest);

%% Accuracy for pre-trained network model
accuracy = mean(predictedlabels == Xtestimds.Labels);
disp("Accuracy for a network from scratch");
disp(accuracy);
disp('Time required for training an alexnet network');
disp(endtime-start);