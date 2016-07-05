function [ dy ] = dsigmoid_fun( X )
%DSIGMOID_FUN Summary of this function goes here
%   Detailed explanation goes here
y = sigmoid_fun(X);
dy = y.*(1-y);

end

