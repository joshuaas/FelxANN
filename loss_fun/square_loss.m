function [ l1 ] = square_loss( Y,h)
%SQUARE_LOSS Summary of this function goes here
%   Detailed explanation goes here
l1  = 0.5*sum(sum((Y-h).^2))/size(Y,2) ;
 
end

