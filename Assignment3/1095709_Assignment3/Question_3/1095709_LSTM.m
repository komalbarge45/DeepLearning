%% Read the COVID-19 dataset
disp('Read the covid-19 dataset');
data = readtable('covid_19_data.csv');
data = data(:,[5,7]);
% Convert the date format to numerical
data.LastUpdate = datenum([data.LastUpdate]);
disp('Manipulate the date format as it is not in the proper format');
time = data.LastUpdate;
time = time - time(1);
data.LastUpdate = time;

disp('Build the graph for time vs death cases');
plot(data.LastUpdate,data.Deaths);figure(gcf);xlabel('Time in hours');ylabel('Death Cases');
data = table2array(data);

% data format manipulation
[uniquearray,~,duplicatearray] = unique(data(:,1));
data = [uniquearray  accumarray(duplicatearray, data(:,2), [], @sum)];
% Train and test data
numTimeStepsTrain = floor(0.9*numel(data(:,2)));
dataTrain = data(1:numTimeStepsTrain+1,2);
dataTest = data(numTimeStepsTrain+1:end,2);
% standard normalization on data
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 500;
% layers in neural network
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    lstmLayer(250)
    lstmLayer(150)
    fullyConnectedLayer(100)
    fullyConnectedLayer(36)
    fullyConnectedLayer(numResponses)
    regressionLayer];
disp('Train the neural network');
options = trainingOptions('adam', ...
    'MaxEpochs',300, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',50, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'Plots','training-progress');
% Train the network
net = trainNetwork(XTrain',YTrain',layers,options);

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
%Predict time steps one at a time and update the network state at each prediction
net = predictAndUpdateState(net,XTrain');
[net,YPred] = predictAndUpdateState(net,YTrain(end)');
disp('Predict time steps one at a time and update the network state at each prediction');
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1)','ExecutionEnvironment','gpu');
end

YPred = sig*YPred + mu;
YTest = dataTest(2:end)';
rmse = sqrt(mean((YPred-YTest).^2));
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Time period in hours")
ylabel("Death Cases")
title("Forecast")
legend(["Observed" "Forecast"])


figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)