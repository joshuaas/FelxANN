function [ inds ] = sel_batch( ind,n )
%SEL_BATCH Summary of this function goes here
%   Detailed explanation goes here
  inds = randsample(ind,n);

end

