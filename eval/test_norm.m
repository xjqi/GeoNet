
norm_gt = load('../gt/norm_gt_l.mat');
norm_gt = norm_gt.norm_gt_l;
masks = load('../gt/masks.mat');
masks = masks.masks;

list = load('../gt/splits.mat');
trainlist = list.trainNdxs;
testlist = list.testNdxs;
norm_gt = norm_gt(:,:,:,testlist);
masks =  masks(:,:,testlist);


norms_pred = load('../trainmodel/norms_pred.mat');
norms_pred = norms_pred.norms;
norms_pred = norms_pred(1:480,1:640,:,:);

eval_pred_sn((norms_pred(:,:,:,:)), (norm_gt(:,:,:,:)), (masks(:,:,:)));
