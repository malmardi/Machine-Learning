function []=NBmain()

loadData;

trainData=trainingData;
trainLabels=trainingLabels;
testData=testingData;
testLabels=testingLabels;

disp('Training your model...');
mdl=NBTrain(trainData, trainLabels);
disp('Done training');
disp('Please enter the number of folds for the cross validation process: ')
fold=input(' ');
disp('cross validating...');
[accuracy]=NBcrossValidation(mdl, fold);
disp('Done cross validation');
disp('the accuracy after performing cross validation =');
accuracy

disp('Predicting...');
Prediction_accuracy=NBPredict(mdl,testData, testLabels);
disp('Done prediction');
disp('the accuracy against the test data =');
Prediction_accuracy

end