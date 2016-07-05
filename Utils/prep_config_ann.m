function [ config ] = prep_config_ann( numhid,acts,loss)
%PREP_CONFIG_ANN Summary of this function goes here
%   Detailed explanation goes here
if( length( numhid ) ~= ( length(acts) -1) )
  error('unequal size of hidden layers and activate functions!')
end

config.nlayer = length(numhid)+2;

config.numhid = numhid;
g = cell(config.nlayer-1,1);

for i = 1:(config.nlayer-1)
    
    
    g{i} =act( acts{i} );

end

if(strcmp(loss,'loglike'))
 config.task = 'clas';
else 
 config.task = 'reg' ;   
end

config.g = g;

config.loss = choose_loss_fun(loss);
end

