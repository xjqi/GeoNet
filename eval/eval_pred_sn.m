% --------------------------------------------------------------------------- %
% MarrRevisited - Surface Normal Estimation
% Copyright (c) 2016 Adobe Systems Incorporated and Carnegie Mellon University. 
% All rights reserved.[see LICENSE for details]
% -------------------------------------------------------------------------- %

% Written by Aayush Bansal. Please contact ab.nsit@gmail.com
function[nums_e] = eval_pred_sn(pred_norm, gt_norm, mask)

	num_images = size(pred_norm,4);
	for i = 1:num_images
	
		% load the file from cache
		

		% CHANGE THE NAME OF THE DATA FILE -- if 
		pred = pred_norm(:,:,:,i);
        pred = double(pred);
	        
		%
		NG = gt_norm(:,:,:,i);
        NG = double(NG);
		NV = mask(:,:,i);
		%
		NP = pred;
		%normalize both to be sure
	        NG = bsxfun(@rdivide,NG,sum(NG.^2,3).^0.5);
                NP = bsxfun(@rdivide,NP,sum(NP.^2,3).^0.5);
		%compute the dot product, and keep on the valid
		DP = sum(NG.*NP,3);
		T = min(1,max(-1,DP));
		pixels{i} = T(find(NV));
	end

	E = acosd(cat(1,pixels{:}));
	nums_e = [mean(E(:)),median(E(:)),mean(E.^2).^0.5,mean(E < 11.25)*100,mean(E < 22.5)*100,mean(E < 30)*100]
	display('---------------------------------------');
	display(['Mean: ', num2str(mean(E(:)))]);
	display(['Median: ', num2str(median(E(:)))]);
	display(['RMSE: ', num2str(mean(E.^2).^0.5)]);
	display(['11.25: ', num2str(mean(E < 11.25)*100)]);
	display(['22.5: ', num2str(mean(E < 22.5)*100)]);
	display(['30: ', num2str(mean(E < 30)*100)]);
	display(['45: ', num2str(mean(E < 45)*100)]);
	display('---------------------------------------');
end
