function[Prediction_accuracy]=NBPredict(mdl,testData, testLabels)

%This function: 
%(1) Finds the predicted classes for a given test data
%(2) Finds the accuracy of the model against the test data

%Inputs:
% testData: contains the test data with out labels
% testLabels: contains the actual labels(classes) of the test data

% Output:
% prediction_accuracy: the accuracy of the model against the test data

predictedLabels=predict(mdl,testData); % this will generate a vector of the predicted classes

temp=numel(find(predictedLabels~=testLabels)); %This will find the element-by-element differences between the predicted and actual labels

Prediction_accuracy=1-temp/length(predictedLabels);% this variabl contains the prediction accuracy


end