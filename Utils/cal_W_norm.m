function [ norm ] = cal_W_norm( W,lambda )
%CAL_W_NORM Summary of this function goes here
%   Detailed explanation goes here
norm = 0;
for i =1 :size(W)
   norm = norm + sum(sum(W{i}.^2));
end

norm = lambda * norm ;

end

