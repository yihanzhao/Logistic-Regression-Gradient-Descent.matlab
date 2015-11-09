# machine-learning
function [theta_new res count] = LRTheta(X, Y, theta_old,tolerance,maxIter,alpha)
%X=inputs
%Y=binomial outputs
%theta_old=the arbitrary starting points of model parameter
%theta_new=the arbitrary starting points of model parameter
%tolerance=the arbitrary difference between model prameters from two
%iterations=if difference larger than tolerance, continue to iterate;
%otherwise, stop;
%maxIter=prevent the iteration does not stop;an arbitrary large number
%alpha=arbitrary value to indicate the length of every step
    [nSamples, nFeature] = size(X);%find the sample size and number of predicted feature 
    theta_new = theta_old;
    count=0;%iteration number
    for j = 1:maxIter
        theta_old=theta_new;
        temp = zeros(nFeature,1);%matrix with zeros
        for k = 1:nSamples
            temp = temp + (1 / (1 + exp(-(X(k,:) * theta_new)))- Y(k)) * [X(k,:)]'; %model pridiction based on gradient descent
        end
       theta_new =theta_new - alpha* temp;%updata new theta
       if abs(theta_new-theta_old)<tolerance %stop if the difference between two iterations is smaller than tolerance
            break
       end
       count=count+1;
    end 
   res = zeros(nSamples,1);
   for i = 1:nSamples
       sigm = 1 / (1 + exp(-(X(i,:) * theta_new)));%predict labels based on model
       if sigm >= 0.5
            res(i) = 1 %if the value is larger than 0.5, give a label of 1; otherwise,0.
        else
            res(i) = 0
        end
   end
end
