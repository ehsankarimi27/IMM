%% https://de.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html
%% this time train with 8 epochs to avoid overfitting 
%epochs = 25, lr = 3e-4 --- accuracy = 0.9357
%epochs = 30, lr = 3e-4 --- accuracy = 0.8874
%epochs = 10, lr = 3e-4 --- accuracy = 0.9142
%epochs = 40, lr = 3e-4 --- accuracy ~ 0.8
%epochs = 90, lr = 1e-3 --- accuracy = 0.8204

imds = imageDatastore('Data', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
%% Automatic split of the data
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
%cd 'C:\Users\medghalchi\Documents\MATLAB\Examples\R2019a\nnet\TransferLearningUsingGoogLeNetExample'
net = inceptionv3;

analyzeNetwork(net)

net.Layers(1)

inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% To check that the new layers are connected correctly, plot the new layer graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
%% Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%% We should include some 
% pixelRange = [-30 30];
%scaleRange = [1.1725 1.1725];
%imageAugmenter = imageDataAugmenter( ...
   % 'RandXScale',scaleRange, ...
   % 'RandYScale',scaleRange);
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
   % 'DataAugmentation',imageAugmenter);


%augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


miniBatchSize = 10; % was 10 by defualt - is 5 in Tom 
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);

   % 'ExecutionEnvironment','gpu', ... %define the running envirement to gpu
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',90, ...    %was 6 by default %90 was in Tom but it over fits
    'InitialLearnRate',3e-4, ... % was 3e-4 by default
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% Train the network
tic
net = trainNetwork(imdsTrain,lgraph,options);
toc

%%  Classify Validation Images
[YPred,probs] = classify(net,imdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%% Display some validation images:
cd  'I:\Daten\Damage_Matlab_IncpV3_Cat2_CorrectedData_New\Data';
mkdir ('Results_1\');
cd 'I:\Daten\Damage_Matlab_IncpV3_Cat2_CorrectedData_New\Data\Results_1'

%idx = randperm(numel(imdsValidation.Files),numel(imdsValidation.Files));
idx = randperm(numel(imdsValidation.Files),100);

% store the shown validation images
for i = 1: length(idx)
    clearvars I label title
    I = readimage(imdsValidation,idx(1,i));
    %imshow(I)
    label = YPred(idx(i));
    %baseFileName = sprintf('%s.png',char(label));  %%'evolved.png'  % char (label);
    title = title(char(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
    imwrite (I, sprintf('%s.png',get(get(gca,'title'),'string')))
    
end


%% new image testing 

imds_Test = imageDatastore('Resized_IncpV3', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[YPred_test,probs_test] = classify(net,imds_Test);
%accuracy = mean(YPred == imds_Test.Labels)




% Display some results:
cd  'Y:\SE_003\DBSCAN_Boxes_test';
mkdir ('Results_1\');
cd 'Y:\SE_003\DBSCAN_Boxes_test\Results_1'

idx = randperm(numel(imds_Test.Files),numel(imds_Test.Files));
for i = 1: length(idx)-1
    clearvars I label title
    I = readimage(imds_Test,idx(1,i));
    %imshow(I)
    label = YPred_test(idx(i));
    %baseFileName = sprintf('%s.png',char(label));  %%'evolved.png'  % char (label);
    title = title(char(label) + ", " + num2str(100*max(probs_test(idx(i),:)),3) + "%");
    imwrite (I, sprintf('%d_%s.png',i,get(get(gca,'title'),'string')))
   i 
end
     
%% At the end: save the workspace in the current directory 
save ('data.mat')
