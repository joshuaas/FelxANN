function [ modelANN ] = Train_ANN( X,Y,X_val,Y_val,alpha,config,lambda,itermax,display,check,k)
%TRAIN_ANN Summary of this function goes here
%   Detailed explanation goes here
%%config nlayer fun
X = X';
Y = Y';
config.numhid = [size(X,1) config.numhid size(Y,1)] ;
[W,b]         = init_weight( config );
nl = config.nlayer;
cnt=0;
if display
    training_loss = zeros(itermax,1);
    training_loss_reg = zeros(itermax,1);
end

pred_val =  training_loss;
if check
 dgW = zeros(1,itermax);
 dgb = dgW;
end
flag = false;
for i = 1:itermax
       
  dW =cell(size(W));
  db =cell(size(b));
  
  for nin = 1:(nl-1)
   dW{nin} = zeros(size(W{nin}));
   db{nin} = zeros(size(b{nin}));
  end
  alpha = alpha * 0.9;
  
  W_org = W;
  b_org = b;
  
  for xind = 1:size(X,2)
    [ A,Z] = FeedForward( X(:,xind),W,b,config,false);
 
   %%% calculate loss derivatives
    if display
    training_loss(i) = training_loss(i)+ config.loss.val(Y(:,xind),A{nl})/size(Y,2);
   end
   Delta = Z; 
  
   Delta{nl} = config.loss.grad(Y(:,xind),A{nl},Z{nl},config.g{end});

   for ni = (nl-1):-1:2
       if( numel(Z{ni}) > 1 )
        Delta{ni} = W{ni}'*Delta{ni+1} .* config.g{ni-1}.grad(Z{ni});
       else
        Delta{ni} = W{ni}'*Delta{ni+1} * config.g{ni-1}.grad(Z{ni});   
       end
   end

   for nn = (nl-1):-1:1
    
        dw = Delta{nn+1}* A{nn}' / size(X,2);
        dbb = Delta{nn+1}/size(X,2);
       
        dW{nn} = dW{nn} + dw ;
        db{nn} = db{nn} + dbb ;
   end
  end
  
  for nn = (nl-1):-1:1
      
      
       if(check)
        [gradW,gradb] = grad_check(X,Y,W_org,b_org,nn,config);
        dgW(i) = dgW(i) + mse(gradW,dW{nn});
        dgb(i) = dgb(i) + mse(gradb,db{nn});
       end
       
       
       dW{nn} = dW{nn} +lambda * W{nn};
       W{nn} = W{nn} -alpha * W{nn};
   end
       
  
  modelANN.W = W;
  modelANN.b = b;
  modelANN.config = config ; 
 
        
  if(i>1) 
   flag = training_loss(i) > training_loss(i-1);
  end
     
    
    pred_Y = predict_ANN(X_val, modelANN);
    pred_val(i) = mse(pred_Y, Y_val);
    if(i>1)
       if(abs( pred_val(i) - pred_val(i-1) ) < 1E-6)
         cnt = cnt +1;
       end
        
       if pred_val(i) > pred_val(i-1) + 1E-4 || cnt > 5
          break;
       end
    end
 end

dgW((i+1):end) =  -1;
dgb((i+1):end) =  -1;
modelANN.W = W;
modelANN.b = b;
modelANN.config = config ;
modelANN.training_loss = training_loss;
modelANN.training_loss_reg = training_loss_reg;
if check
    modelANN.dgW = dgW / numel(W{:}) ;
    modelANN.dgb = dgb / numel(b{:}) ;
end
plot(training_loss)
%plot(training_loss_reg)
figure 
plot(training_loss_reg)
end

