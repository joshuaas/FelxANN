function [ modelANN ] = Train_ANN( X,Y,X_val,Y_val,alpha,config,lambda,itermax,display,check,gamma1,gamma2,batchsize)
%TRAIN_ANN Summary of this function goes here
%   Detailed explanation goes here
%%config nlayer fun
X_org = X';
Y_org = Y';
gamma = gamma1;
modelANN.config = config ; 
if strcmp(config.task,'clas')
    config.numhid = [size(X_org,1) config.numhid length(unique(Y_org))] ;
else
    config.numhid = [size(X_org,1) config.numhid size(Y_org,1)] ;
end
[W,b]         = init_weight( config );
nl = config.nlayer;
cnt  = 0;
cnt1 = 0;
if display
    training_loss = zeros(itermax,1);
    training_loss_reg = zeros(itermax,1);
end

pred_val =  training_loss;
if check
 dgW = zeros(1,itermax);
 dgb = dgW;
end
decrease_alpha = false;
increase_alpha = false;

dW = cell(size(W));
db = cell(size(b));
ind_his =  0;
for i = 1:itermax
  inds = sel_batch(1:size(X_org,2),batchsize);
  X = X_org(:,inds);
  Y = ind2vec(Y_org(:,inds)+1);
  if decrease_alpha
   alpha = alpha*0.9;
   decrease_alpha =false;
  elseif increase_alpha
      alpha = alpha*1.05;
   increase_alpha =false;     
  end
  
  
  
  
  W_org = W;
  b_org = b;
  dW_old  = dW ;
  db_old  = db ;
  
    [ A,Z] = FeedForward( X,W,b,config,false);
 
   %%% calculate loss derivatives
    if display
     if strcmp(config.task,'reg')
        
         training_loss(i) = training_loss(i)+ config.loss.val(Y,A{nl});
     
     else
        
        training_loss(i) =  -sum((vec2ind(A{nl})-1)==Y_org(:,inds))/length(inds);
     end
    end
  
   Delta_old = config.loss.grad(Y,A{nl},Z{nl},config.g{end});

   for ni = (nl-1):-1:1
        
        if(ni > 1)
            Delta_new = W{ni}'*Delta_old .* config.g{ni-1}.grad(Z{ni});
        end
        
        dW{ni} = Delta_old* A{ni}' / size(X,2);
        db{ni} = sum(Delta_old,2)/size(X,2);
        
  
        
        if(check)
            
            [gradW,gradb] = grad_check(X,Y,W_org,b_org,ni,config);
            dgW(i) = dgW(i) + mse(gradW,dW{ni});
            dgb(i) = dgb(i) + mse(gradb,db{ni});
            
        end
       
        dW{ni} = dW{ni} +lambda * dW{ni};
   
        Delta_old = Delta_new;
        if(i >1)
        
            W{ni} = W{ni} - alpha * dW{ni} - gamma *(dW{ni} -  dW_old{ni}) ;
            b{ni} = b{ni} - alpha * db{ni} - gamma *(db{ni} -  db_old{ni}) ;
        else
             W{ni} = W{ni} - alpha * dW{ni} ;
             b{ni} = b{ni} - alpha * db{ni} ;
        end
   end

       
  
  modelANN.W = W;
  modelANN.b = b;
 
        
  if(i>1) 
   flag = training_loss(i) > training_loss(i-1);
  end
     
    
    pred_Y = predict_ANN(X_val, modelANN);
    if(strcmp(config.task,'reg'))
        pred_val(i) = mse(pred_Y, Y_val);
    else
       pred_val(i) = -sum(vec2ind(pred_Y')-1 == Y_val')/length(Y_val); 
    end
    if(i>1)
       
        if(training_loss(i) > training_loss(i-1) )
          decrease_alpha = true ;
        end
        
       if(abs( training_loss(i)  -training_loss(i-1)  ) < 1E-6)
         cnt = cnt +1;
       elseif cnt >1
           cnt = 0;
           gamma = gamma1 ;
       end
        
       if(~decrease_alpha && cnt >15)
           gamma = gamma2 ;
           increase_alpha = true  ;
       end
       
       if  cnt > 50
    
           break;
       end
       
       if pred_val(i) > min(pred_val(1:i)) + 1E-4 
           if cnt1 == 0
            W_his = W_org;
            b_his = b_org;
            ind_his = i ; 
            cnt1 = cnt1 + 1;
           else
            cnt1 = cnt1 + 1;   
           end
          
       elseif  cnt1 > 0
            cnt1 = 0;
            ind_his = 0 ;
       end
       
       if(cnt1 > 6)
            W = W_his ;
            b = b_his ;
        break;
       end
       
       
    end
 end
if check
    dgW((i+1):end) =  -1;
    dgb((i+1):end) =  -1;
end
modelANN.W = W;
modelANN.b = b;
modelANN.config = config ;
modelANN.training_loss = training_loss;
modelANN.training_loss_reg = training_loss_reg;
if check
    modelANN.dgW = dgW / numel(W) ;
    modelANN.dgb = dgb / numel(b) ;
end

if(display)
    plot(training_loss(1:i))
    %plot(training_loss_reg)
    hold on 
    plot(pred_val(1:i))

    legend('training err','valid err')

    if ind_his > 0
        
        yr = get(gca,'ylim');
        if ind_his == 0
           
        end
        plot([i-1 i-1],yr,'--')

    end

    hold off
end
end

