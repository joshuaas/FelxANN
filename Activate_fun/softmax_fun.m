function [y ] = softmax_fun( X )
%SOFTMAX_FUN Summary of this function goes here
%   Detailed explanation goes here
y =  exp(X) ;
y = bsxfun(@rdivide,y,sum(y,1));

end

