
function[mdlp]=DTprune(mdl, depth)

%[~,~,~,bestlevel] = cvLoss(mdl,'SubTrees','All'); % this is to find the best tree depth which is turned out to be 35
mdlp = prune(mdl, 'Level', depth);% this is to prune the tree based on the depth entered by the user


end
