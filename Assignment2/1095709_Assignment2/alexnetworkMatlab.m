%% Load AlexNet package
alexnetwork = alexnet;
layers = alexnetwork.Layers;
%% Updation of 23rd layer in AlexNet pretrained model
layers(23)=fullyConnectedLayer(100,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20);
%% Replacement of 25th layer of classification 
layers(25)=classificationLayer;

%% Load images for CIFAR100 dataset
Ximds = imageDatastore('CIFAR-100\CIFAR-100\TRAIN\','IncludeSubfolders',true,'LabelSource','foldernames');
Xtestimds = imageDatastore('CIFAR-100\CIFAR-100\TEST\','IncludeSubfolders',true,'LabelSource','foldernames');
%% Convert images into AlexNet input data format
X = augmentedImageDatastore([227 227],Ximds);
Xtest = augmentedImageDatastore([227 227],Xtestimds);

%% Options to be provided for network
miniBatchSize = 64;
valFreq = floor(numel(X.Files)/miniBatchSize);
opts = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',Xtest, ...
    'ValidationFrequency',valFreq, ...
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
disp("Accuracy for pretrained network");
disp(accuracy);
disp('Time required for training a network');
disp(endtime-start);