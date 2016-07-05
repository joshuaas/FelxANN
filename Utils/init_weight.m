function [ W,b ] = init_weight( config )
%INIITIALIZE_WEIGHT Summary of this function goes here
%   Detailed explanation goes heren
nl =  config.nlayer;
W  =  cell(nl-1,1);
b  =  cell(nl-1,1);
for i = 1:(nl-1)
    a = sqrt(6 / ( config.numhid(i)+config.numhid(i+1)) );
  W{i} = -a + 2*a*rand(config.numhid(i+1),config.numhid(i));
  b{i} = -1 +2*rand(config.numhid(i+1),1);
end


end

