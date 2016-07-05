function [ g ] = choose_loss_fun( type )
%CHOSS_LOSS_FUN Summary of this function goes here
%   Detailed explanation goes here
  switch type
        case 'square'
            g.val  = @(y,h)square_loss(y,h);
            g.grad = @(Y,h,zh,gh)dsquare_loss(Y,h,zh,gh);
      case   'loglike'
            g.val  = @(y,h)loglike_fun(y,h) ;
            g.grad = @(y,h,z,g)dloglike_fun(y,h,z,g) ;
        otherwise
            error('loss fun should either be square or softmax or logit');
   end

end

