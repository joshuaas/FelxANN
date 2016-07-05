function [ res ] = loglike_fun( Y,H )
%LOGLIKE_LOSS Summary of this function goes here
%   Detailed explanation goes here

res = -sum(Y(:) .* log(H(:)))/size(Y,2);

end

