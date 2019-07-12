
depths_gt = load('../gt/depths.mat');
depths_gt = depths_gt.depths;

list = load('../gt/splits.mat');
trainlist = list.trainNdxs;
testlist = list.testNdxs;

% test list 
depth_gt_eval = depths_gt(:,:,list.testNdxs);
depth_gt_eval = depth_gt_eval(45:471, 41:601,:);

depths_pred = load('../trainmodel/depths_pred.mat');
depths_pred = depths_pred.depths;

%center crop following previous work
depth_pred_eval = depths_pred(45:471, 41:601,:);
% evaluation
errors = error_metrics_new(double(depth_pred_eval),double(depth_gt_eval),[]);