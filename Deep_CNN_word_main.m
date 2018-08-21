%% Proposed CNN based word image classification

% MULTICLASS CLASSIFICATION: Fully Connected Layer contanins neurons
% indicating different classes
% FOR EXAMPLE, if there are 120 classes in all, there will have 120 output
% neurons

    %%%%    Authors:    DIBYASUNDAR DAS AND DEEPAK RANJAN NAYAK 
    %%%%    NATIONAL INSTITUTE OF TECHNOLOGY ROURKELA, INDIA
    %%%%    EMAIL:      DIBYASUNDARIT@GMAIL.COM; DEPAKRANJANNAYAK@GMAIL.COM
    %%%%    WEBSITE:    https://www.researchgate.net/profile/Deepak_Nayak11
    %%%%    DATE:       AUGUST 2018
clear;clc;close all;
%% Loading the traning images
load('train_data.mat');
%%%%%%%%%%%%%% Designing the network %%%%%%%%%%%%%%%
[m,n,p,q]=size(train_data);
%% Proposed CNN layers
layers=[imageInputLayer([m,n,p])
        convolution2dLayer(11,16)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2) 
        convolution2dLayer(7,32)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2) 
        convolution2dLayer(5,64)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2) 
        convolution2dLayer(3,128)
        batchNormalizationLayer
        reluLayer
        averagePooling2dLayer(2,'Stride',2)
        fullyConnectedLayer(120)
        softmaxLayer
        classificationLayer]';
%% Converting traning class labels from numerical value to categorical
    u=unique(train_cls);
    for i=1:size(u,1)
        cld{i}=strcat('c',num2str(u(i)));
    end
train_cls_cat=categorical(train_cls,u,cld);
%% Traning options   
options=trainingOptions('sgdm','MiniBatchSize',10,'InitialLearnRate',0.001,'Plots','training-progress',...
    'MaxEpochs',15,'ExecutionEnvironment','gpu',...
    'Shuffle','once','VerboseFrequency',1,'Verbose',true,'L2Regularization',0.001);

 %% Train the designed network  
  [net_pop,v]=trainNetwork(train_data,train_cls_cat,layers,options);

%% Testing phase
disp('start the testing phase ......')
load('test_data.mat'); % Load the test images

%% Converting testing class labels from numerical value to categorical
    u=unique(test_cls);
    for i=1:size(u,1)
        cld{i}=strcat('c',num2str(u(i)));
    end
test_cls_cat=categorical(test_cls,u,cld);

%% Classify testing images 
YPred = classify(net_pop,test_data);

%% Output the confusion matrix
[mriConf,names]=confusionmat(test_cls_cat,YPred);
%% Heat map representation of confusion matrix
heatmap(names,names,mriConf)
%% Calculating accuracy
Accuracy = sum(YPred == test_cls_cat)/numel(test_cls_cat);

