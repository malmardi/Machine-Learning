function [accuracy]=DCcrossValidation(mdl)

cvtree=crossval(mdl); %Apply cross validation of the model
cvloss = kfoldLoss(cvtree); % find the accuracy after running the cross validation

accuracy=1-cvloss;
end