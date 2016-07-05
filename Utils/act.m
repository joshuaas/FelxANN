function [ g ] = act( type )
%ACT Summary of this function goes here
%   Detailed explanation goes here
%ACITIVATE_FUN Summary of this function goes here
%   Detailed explanation goes here

    switch type
        case 'tanh'
            g.val  = @(x)tanh_fun(x);
            g.grad = @(x)dtanh_fun(x);
        case 'sigmoid'
            g.val = @(x)sigmoid_fun(x);
            g.grad = @(x)dsigmoid_fun(x);
        case 'linear'
            g.val  = @(x)lin_fun(x);
            g.grad = @(x)dlin_fun(x);
        case 'softmax'
            g.val  = @(x)softmax_fun(x);
            g.grad = @(x)dsoftmax_fun(x);
        otherwise
            error('activate fun should either be tanh sigmoid or linear');
    end

end



