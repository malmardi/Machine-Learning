disp('please enter the path of the data');
path=input('','s');

trainDataPath=strcat(path,'\train.data');
trainLabelsPath=strcat(path,'\train.label');
testDataPath=strcat(path,'\test.data');
testLabelsPath=strcat(path,'\test.label');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% load the train data
f1=load(trainDataPath);% loading the trainData
f2=load(testDataPath);%loading the testData
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
f=fopen(trainDataPath);%read trainData again
  trainData=textscan(f,'%d %d %d');% Read formatted data from text file or string
  fclose(f); 
  
  f=fopen(testDataPath);%read testData again
  testData=textscan(f,'%d %d %d');% Read formatted data from text file or string
  fclose(f); 
  
  
trainData= double(cell2mat(trainData));%Convert cell array to ordinary array of the underlying data type
trainData = sparse(trainData(:,1), trainData(:,2), trainData(:,3), max(trainData(:,1)), maxWord);% changing the dataset into sparse representation, given into account the maxWord calculated abovee
trainData = full(trainData);

testData= double(cell2mat(testData));
testData = sparse(testData(:,1), testData(:,2), testData(:,3), max(testData(:,1)),maxWord);
testData = full(testData);

%%%%%%%%%%%%%%%%%%%%%load the train labels%%%%%%%%%%%%%%%%%%%%%%%%%%% 

f=fopen(trainLabelsPath);
  trainLabels=textscan(f,'%d');
  fclose(f); 

trainLabels= double(cell2mat(trainLabels));
trainLabels = sparse(trainLabels(:,1));
trainLabels = full(trainLabels);

%%%%%%%%%%%%%%%%%%%%load test labels%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
f=fopen(testLabelsPath);
  testLabels=textscan(f,'%d');
  fclose(f); 

testLabels= double(cell2mat(testLabels));
testLabels = sparse(testLabels(:,1));
testLabels = full(testLabels);

%Finished loading all the needed data
