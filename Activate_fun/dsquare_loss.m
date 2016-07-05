function [ dl ] = dsquare_loss(  Y,h,zh,gh )
%DSQUARE_LOSS Summary of this function goes here
%   Detailed explanation goes here
dl  = -(Y-h).*gh.grad(zh) ;
 

end

