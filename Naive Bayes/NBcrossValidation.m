%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is written by Mamoun Almardini. Feel Free to copy the code with %
% acknowledgement                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [accuracy]=NBcrossValidation(mdl, fold)
% This function takes the trained model and the number of folds needed for
% cross validation

CVMdl = crossval(mdl,'KFold',fold); % applies cross validation on the trained model to have a better
                                    %sense of the accuracy of the model. The number of folds is
                                    %specified by the user

%cvmdl=mdl.crossval; % run cross validation on the model (10-fold)
Loss = kfoldLoss(CVMdl); % to find the accuracy after running the cross validation
accuracy=1-Loss;% this is the accuracy of the model after running the cross validation


end