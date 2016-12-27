function[]= DTmain()

loadData;

disp('Training your model...');
mdl=DTtrain(trainData, trainLabels);% this function trains your model based on the train data and train labels
disp('Done training');

disp('Please enter the depth: ');
depth=input(' ');% asking the user to enter the desired depth
disp('prunning...');
mdlp = DTprune(mdl, depth);% this is to prune the tree based on the depth entered by the user
disp('Done prunning');

disp('cross validating...');
[accuracy]=DCcrossValidation(mdlp);% this function applies cross validation on the trained model
disp('Done cross validation');

disp('the accuracy after performing cross validation =');
accuracy


disp('Predicting...');
Prediction_accuracy=DTpredict(mdlp,testData, testLabels);% this function finds the predicted classes given the test data
disp('Done prediction');
disp('the accuracy against the test data =');
Prediction_accuracy


end