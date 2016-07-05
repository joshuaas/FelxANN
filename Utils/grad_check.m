function [ gradW,gradb ] = grad_check( X,Y,W,b,ind,config)
%GRAD_CHECK Summary of this function goes here
%   Detailed explanation goes here
gradW = zeros(size(W{ind}));
gradb = zeros(size(b{ind}));
epsilon = 0.00001;
for i = 1:numel(W{ind})
   delta  = epsilon ;

   W{ind}(i) = W{ind}(i) +delta ;
   
   A  = FeedForward(X,W,b,config,true);
   
   j1 = config.loss.val(Y,A);
   
  
   W{ind}(i) = W{ind}(i) -2*delta ;
   
   A  = FeedForward(X,W,b,config,true);
   
   j2 = config.loss.val(Y,A);
   
    W{ind}(i) = W{ind}(i) +delta ;
   
    gradW(i) = (j1-j2)/(2*delta);
end

for i = 1:numel(b{ind})
   delta  = epsilon  ;
   b{ind}(i) = b{ind}(i)+ delta ;
   
   A  = FeedForward(X,W,b,config,true);
   
   j1 = config.loss.val(Y,A);
   
  
   b{ind}(i) = b{ind}(i) -2*delta;
   
   A  = FeedForward(X,W,b,config,true);
   
   j2 = config.loss.val(Y,A);
   
    b{ind}(i) = b{ind}(i) +delta;
   
    gradb(i) = (j1-j2)/(2*delta);
end

end

