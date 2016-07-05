function [ param ] = struct2par( W,b )
%STRUCT2PAR Summary of this function goes here
%   Detailed explanation goes here
param = [];
for i  = 1:numel(W)
    Wi = W(i);
    bi = b(i);
 param = [param;Wi(:);bi(:)];
end
    

end

