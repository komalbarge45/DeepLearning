function backPropagationAlgorithm()
%% Load the dataset
data = load("EEGEyeState.csv");
trainingSamp   = 10000;
testingSamp   = 4980;
totalSamp = trainingSamp+testingSamp;
%% Shuffle the data
[n_row, n_col] = size(data);
shuffle_seq = randperm(n_row);
for i = (1:n_row)
    data_shuffled(i,:) = data(shuffle_seq(i),:);
end
%% Define hidden neurons and input neurons
inputN = n_col-1;
hiddenN = 30;
outputN = 1;
%% Define random weights for all 4 layers
currWeight{1} = rand(hiddenN,inputN+1);
currWeight{2} = rand(hiddenN,hiddenN+1);
currWeight{3} = rand(hiddenN,hiddenN+1);
currWeight{4} = rand(outputN,hiddenN+1);
%% Initialize previous weight values as zeros
prevWeight{1}= zeros(hiddenN,inputN+1);
prevWeight{2}= zeros(hiddenN,hiddenN+1);
prevWeight{3}= zeros(hiddenN,hiddenN+1);
prevWeight{4}= zeros(outputN,hiddenN+1);
%% Initialize epochs, momentum constant and learning rate
numOfEpoch = 50;
epoch = 1;
alpha = 0.1;
err = 0;
trainingError = 0;
eta1 = annealing(0.1,1E-5,numOfEpoch);
eta2 = annealing(0.1,1E-5,numOfEpoch);
eta3 = annealing(0.1,1E-5,numOfEpoch);
eta4 = annealing(0.1,1E-5,numOfEpoch);

for i = 1:totalSamp
    inputData(:,i) = data_shuffled(i,:)';
end

for epoch = 1:numOfEpoch
    %% shuffle the input data
    shuffle_seq = randperm(trainingSamp);
    nor_data1 = inputData(:,shuffle_seq);
   
    %% Train the data
    for i = 1:trainingSamp
        %% Forward 4 layer computation
        inputLayer  = [nor_data1(1:14,i);1];     % fetching input data from database # added 1 as a bias to input
        targetVal  = nor_data1(15,i);% fetching desired response from database
        hidlayout1 = [hyperb(currWeight{1}*inputLayer);1];          % hidden neurons are nonlinear #added 1 as a bias for next hidden layer
        hidlayout2 = [hyperb(currWeight{2}*hidlayout1);1];
        hidlayout3 = [hyperb(currWeight{3}*hidlayout2);1];
        output  = hyperb(currWeight{4}*hidlayout3);         % output neuron is nonlinear
        e(:,i)  = targetVal - output;
        
        out = 1*(output>=0) + 0*(output<0);
        if abs(out) ~= targetVal
            trainingError = trainingError + 1;
        end
        
        %% Backward loss function computation
        % delta for each layer
        delta_output = e(:,i).*d_hyperb(currWeight{4}*hidlayout3);
        delta_hidlay3 = d_hyperb(currWeight{3}*hidlayout2).*(currWeight{4}(:,1:hiddenN)'*delta_output);
        delta_hidlay2 = d_hyperb(currWeight{2}*hidlayout1).*(currWeight{3}(:,1:hiddenN)'*delta_hidlay3);
        delta_hidlay1 = d_hyperb(currWeight{1}*inputLayer).*(currWeight{2}(:,1:hiddenN)'*delta_hidlay2);
        % delta for weights in each layer
        delta_weight{1} = eta1(epoch)*delta_hidlay1*inputLayer';
        delta_weight{2} = eta2(epoch)*delta_hidlay2*hidlayout1';
        delta_weight{3} = eta3(epoch)*delta_hidlay3*hidlayout2';
        delta_weight{4} = eta4(epoch)*delta_output*hidlayout3';
              
        %% update the weights with previous and delta weights
        updatedWeight{1} = currWeight{1} + alpha*prevWeight{1} + delta_weight{1};  % weights input -> hidden
        updatedWeight{2} = currWeight{2} + alpha*prevWeight{2} + delta_weight{2};  % weights hidden-> output
        updatedWeight{3} = currWeight{3} + alpha*prevWeight{3} + delta_weight{3};
        updatedWeight{4} = currWeight{4} + alpha*prevWeight{4} + delta_weight{4};
        
        %% move weights one-step
        prevWeight = delta_weight;
        currWeight  = updatedWeight;
    end
    err = trainingError;
    trainingError = 0;
    mse(epoch) =sum(mean(e'.^2));
end
fprintf('Training Accuracy : %5.2f\n',((trainingSamp-err)/trainingSamp)*100);
err = 0;
%% Draw a learning curve
figure;
plot(mse,'k');
xlabel('Number of epochs');ylabel('MSE');
title('Learning curve');
%% Test the testing sample with the trained weights
for i = trainingSamp+1:totalSamp
    inputLayer   = [inputData(1:14,i);1];
    hidlayout1  = [hyperb(currWeight{1}*inputLayer);1];
    hidlayout2 = [hyperb(currWeight{2}*hidlayout1);1];
    hidlayout3 = [hyperb(currWeight{3}*hidlayout2);1];
    output(:,i)= hyperb(currWeight{4}*hidlayout3);
end
% Calculate testing error rate
for i = trainingSamp+1:totalSamp
    out = 1*(output(i)>=0) + 0*(output(i)<0);
    if out ~= inputData(15,i)
        err = err + 1;
    end
end
fprintf('Testing Accuracy : %5.2f\n',((testingSamp-err)/testingSamp)*100);

