%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is written by Mamoun Almardini. Feel Free to copy the code with %
% acknowledgement                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[mdl]= NBTrain(trainData, trainLabels)

%mn : Multinomial bag-of-tokens model. Indicates that all predictors have this distribution
dist_type='mn';

%train the model using the fitcnb function which trains your model based on
%the data (trainData) and classes (trainLabels). Also keep in mind that we
%have to set the disribution to mn

mdl=fitcnb(trainData,trainLabels, 'Distribution', dist_type); %this is to train your model

end