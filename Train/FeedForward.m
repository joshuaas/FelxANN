function [ A,Z ] = FeedForward( X,W,b,config,clean)
%FEEDFORWARD Summary of this function goes here
%   Detailed explanation goes here
A = cell(config.nlayer,1);
Z=  cell(config.nlayer-1,1);
Z{1} = -1;
A{1} = X;

for i  =1:(config.nlayer-1)
   g = config.g{i} ;
   Z{i+1} = W{i} * A{i} + b{i} * ones( 1,size(X,2) );
   A{i+1} = g.val(Z{i+1});
end

if(clean)
   A = A{config.nlayer};   
   Z = [];
end

end

