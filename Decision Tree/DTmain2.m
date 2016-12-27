function[]= DTmain2()

loadData;

'Training your model...'
mdl=DTtrain(trainData, trainLabels);% this function trains your model based on the train data and train labels
'Done training'


%depth=input('Please enter the depth: ');% asking the user to enter the desired depth
'prunning...'
mdlp = DTprune(mdl, 35);% this is to prune the tree based on the depth entered by the user
'Done prunning'

'cross validating...'
[accuracy]=DCcrossValidation(mdlp);% this function applies cross validation on the trained model
'Done cross validation'

'the accuracy after performing cross validation ='
accuracy


'Predicting...'
Prediction_accuracy=DTpredict(mdlp,testData, testLabels);% this function finds the predicted classes given the test data
'Done prediction'
'the accuracy against the test data ='
Prediction_accuracy


end