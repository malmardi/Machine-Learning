function[mdl]=DTtrain(trainData, trainLabels)

mdl=fitctree(trainData,trainLabels); %this is to train your model
%[~,~,~,bestlevel] = cvLoss(mdl,'SubTrees','All'); % this is to find the best tree depth which is turned out to be 35

end