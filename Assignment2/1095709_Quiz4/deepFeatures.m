%% Fetch the images from Caltech256 dataset
imageData = imageDatastore('256_ObjectCategories',...
'LabelSource', 'foldernames', 'IncludeSubfolders', true);

%% Split the labels 
% Use 30 images for training and remaining images for testing purpose
disp('1. Spliting labels for training and testing');
[trainingSet, testingSet] = splitEachLabel(imageData, 30);

%% Load ResNet101
disp('2. Loading Pretrained ResNet101');
net = resnet101;

%% Redefining read function to process images
disp('3. Preprocessing images for ResNet101 network');
trainingSet.ReadFcn = @(filename)PrepImageWithResnet101Dim(filename);
testingSet.ReadFcn = @(filename)PrepImageWithResnet101Dim(filename);

 %% Get features from ResNet101
disp('4. Extracting ResNet101 deeper layer features');
extractionLayer = 'fc1000';
resNet_trainFeatures = activations(net,trainingSet,extractionLayer,'MiniBatchSize',120);
resNet_trainFeatures = reshape(resNet_trainFeatures,[1*1*1000,size(resNet_trainFeatures,4)])' ;
resNet_testFeatures = activations(net,testingSet,extractionLayer,'MiniBatchSize',120);
resNet_testFeatures = reshape(resNet_testFeatures,[1*1*1000,size(resNet_testFeatures,4)])';
 
train_labels = grp2idx(trainingSet.Labels);
test_labels = grp2idx(testingSet.Labels);

%% Creating training and testing dataset
training = horzcat(train_labels,resNet_trainFeatures);
testing = horzcat(test_labels,resNet_testFeatures);

disp('5. Classification using ELM algorithm');
[TrainingTime, TestingAccuracy,Training,Testing] = ELM(training, testing, 1, 10000, 'sig');