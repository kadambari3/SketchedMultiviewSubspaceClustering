function [acc] = compute_clus_acc(gnd,idx)
%CALC_ACC Function that calculates the accuracy of a clustering algorithm.
% Inputs: gnd - Nx1 vector of ground truth labels
%         idx - Nx1 vector of estimated labels
% Output: acc - Accuracy in [0,1] of the estimated labels
%   Panagiotis Traganitis email: traga003@umn.edu

newidx = bestMap(gnd,idx);
acc = sum(gnd(:) == newidx(:)) / length(newidx);

end