function[Predict_accuracy]=DTpredict(mdl, testData, testLabels)


predictedLabels=predict(mdl,testData); % prdeict the labels of the test data (T). predictedLabels contains the predicted labels
temp=numel(find(predictedLabels~=testLabels)); %this to compare the predicted labels with the actual labels
Predict_accuracy=1-temp/length(predictedLabels); % calculating the accuracy of the model after testing it against the test data

end