function [ y ] = predict_ANN( X,model )
%PREDICT_ANN Summary of this function goes here
%   Detailed explanation goes here
 y = FeedForward( X',model.W,model.b,model.config,true);
 y = y';

end

