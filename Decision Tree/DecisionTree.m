% script loads data from 20new-bydate



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load the train data
f1=load('C:\Users\malmardi\Dropbox\machinelearning\ml\train.data');
f2=load('C:\Users\malmardi\Dropbox\machinelearning\ml\test.data');
maxWord = max([f1(:,2); f2(:,2)]);
clear f1
clear f2

f=fopen('C:\Users\malmardi\Dropbox\machinelearning\ml\train.data');
  trainData=textscan(f,'%d %d %d');
  fclose(f); 
  
  f=fopen('C:\Users\malmardi\Desktop\New folder (2)\ITCS6156_HW1 (3)\ITCS6156_HW1\20news-bydate\data\test.data');
  testData=textscan(f,'%d %d %d');
  fclose(f); 
  %l1=length(unique(trainData(:,2)));
  %l2=length(unique(testData(:,2)));
 
  
trainData= double(cell2mat(trainData));
trainData = sparse(trainData(:,1), trainData(:,2), trainData(:,3), max(trainData(:,1)), maxWord);
trainData = full(trainData);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


testData= double(cell2mat(testData));
testData = sparse(testData(:,1), testData(:,2), testData(:,3), max(testData(:,1)),maxWord);
testData = full(testData);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load the train labels

f=fopen('C:\Users\malmardi\Dropbox\machinelearning\ml\train.label');
  trainLabels=textscan(f,'%d');
  fclose(f); 

trainLabels= double(cell2mat(trainLabels));
trainLabels = sparse(trainLabels(:,1));
trainLabels = full(trainLabels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load test labels
f=fopen('C:\Users\malmardi\Dropbox\machinelearning\ml\test.label');
  testLabels=textscan(f,'%d');
  fclose(f); 

testLabels= double(cell2mat(testLabels));
testLabels = sparse(testLabels(:,1));
testLabels = full(testLabels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mdl=fitctree(trainData,trainLabels); %this is to train your model
%[~,~,~,bestlevel] = cvLoss(mdl,'SubTrees','All'); % this is to find the best tree depth which is turned to be 35

cvtree=crossval(mdl); %Apply cross validation of the model
cvloss = kfoldLoss(cvtree); % find the accuracy after running the cross validation
%the accuracy you will get here is before prunning the tree

mdl = prune(mdl,'Level',35);% this is to prune the tree based on the best depth calculated in the above command

cvtree=crossval(mdl); %Apply cross validation of the model
cvloss2 = kfoldLoss(cvtree); % find the accuracy after running the cross validation
%The accuracy you will get here is after prunning the tree

predictedLabels=predict(mdl,testData); % prdeict the labels of the test data (T). L contains the predicted labels
temp=numel(find(predictedLabels~=testLabels)); %this to compare the predicted labels with the actual labels
Predict_accuracy=1-temp/length(predictedLabels); % calculating the accuracy of the model after testing it against the test data




