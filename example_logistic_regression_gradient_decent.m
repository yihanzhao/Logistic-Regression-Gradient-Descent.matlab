theta_old = rand(3,1)%random theta old
theta_new=rand(3,1)%random theta new
alpha=0.05;%step size
tolerance=0.01;%tolerance
maxIter=100000%max number of iteration
[theta_new res count] = LRTheta(X, Y, theta_old,tolerance,maxIter,alpha)
errors = abs(Y - res);%absolute difference between true labels and predicted ones
err = sum(errors)%sum errors up
percentage = 1 - err / size(X, 1)%correct prediction rate
for l=1:200%empirical risk function
    empR=sum((Y-1)*log(1-1 / (1 + exp(-(X(l,:) * theta_new))))- Y*(1/1 / (1 + exp(-(X(l,:) * theta_new)))));
end
empR
figure(1)%plot X's and Y's based on original data
scatter(X(:,1),X(:,2),40,Y)
title('Logistic Regression Predicted Boundary')
xlabel('X1')
ylabel('X2')
hold on;
%plot decision boundary
x1=0:0.1:1;
x2=(-theta_new(3)-theta_new(1)*x1)/theta_new(2);%plot decision boundary
plot(x1,x2)
hold off;
