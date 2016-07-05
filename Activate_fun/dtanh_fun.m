function [ dy ] = dtanh_fun(X)
%TANH_FUN Summary of this function goes here
%   Detailed explanation goes here
dy  = 1-tanh(X).^2;
end