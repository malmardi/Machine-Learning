% This script is to load the trainData, trainLabels, testData, and testLabesl

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load the train data
f1=load('C:\Users\malmardi\Dropbox\machinelearning\ml\train.data');% loading the trainData
f2=load('C:\Users\malmardi\Dropbox\machinelearning\ml\test.data');%loading the testData
maxWord = max([f1(:,2); f2(:,2)]);% there are 7213 word_ID exist in the testData and don't exist in the trainData
                                  %this will make the number of columns
                                  %different in both data set after
                                  %transforming them into
                                  %sparse reepresentations. Therefore, we
                                  %need to find the maximum number of words
                                  %and assign it to both of them.
                                  
clear f1
clear f2
%Note:max function does not work with "fopen", so I used "load" instead.

%
f=fopen('C:\Users\malmardi\Dropbox\machinelearning\ml\train.data');%read trainData again
  trainData=textscan(f,'%d %d %d');% Read formatted data from text file or string
  fclose(f); 
  
  f=fopen('C:\Users\malmardi\Dropbox\machinelearning\ml\test.data');%read testData again
  testData=textscan(f,'%d %d %d');% Read formatted data from text file or string
  fclose(f); 
  
  
trainData= double(cell2mat(trainData));%Convert cell array to ordinary array of the underlying data type
trainData = sparse(trainData(:,1), trainData(:,2), trainData(:,3), max(trainData(:,1)), maxWord);% changing the dataset into sparse representation, given into account the maxWord calculated abovee
trainData = full(trainData);

testData= double(cell2mat(testData));
testData = sparse(testData(:,1), testData(:,2), testData(:,3), max(testData(:,1)),maxWord);
testData = full(testData);

%%%%%%%%%%%%%%%%%%%%%load the train labels%%%%%%%%%%%%%%%%%%%%%%%%%%% 

f=fopen('C:\Users\malmardi\Dropbox\machinelearning\ml\train.label');
  trainLabels=textscan(f,'%d');
  fclose(f); 

trainLabels= double(cell2mat(trainLabels));
trainLabels = sparse(trainLabels(:,1));
trainLabels = full(trainLabels);

%%%%%%%%%%%%%%%%%%%%load test labels%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
f=fopen('C:\Users\malmardi\Dropbox\machinelearning\ml\test.label');
  testLabels=textscan(f,'%d');
  fclose(f); 

testLabels= double(cell2mat(testLabels));
testLabels = sparse(testLabels(:,1));
testLabels = full(testLabels);

%Finished loading all the needed data

mdl=fitcnb(trainData,trainLabels, 'Distribution', 'mn'); %this is to train your model

FreqDist=tabulate(testLabels);
l=sum(FreqDist(:,2));
alpha=0.4;
for n = 1:20
    FreqDist(n,3)=((FreqDist(n,2)+alpha)/(l+2*alpha));
end

mdl.Prior = FreqDist(:,3);

CVMdl = crossval(mdl,'KFold',3);

%cvmdl=mdl.crossval; % run cross validation on the model (10-fold)
Loss = kfoldLoss(CVMdl); % to find the accuracy after running the cross validation
accuracy=1-Loss% this is the accuracy of the model after running the cross validation

predicteLabels=predict(mdl,testData); % this will generate a vector of the predicted classes

temp=numel(find(predicteLabels~=testLabels)); %This will find the element-by-element differences between the predicted and actual labels

Predict_accuracy=1-temp/length(predicteLabels)






