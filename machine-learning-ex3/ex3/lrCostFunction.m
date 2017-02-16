function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

Y1 = sigmoid(X * theta);
J = sum((-y .* log(Y1)) - ((1 - y) .* log(1 - Y1))) / m;
thetaSquare = theta.^2;
regTerm = (lambda  / (2 * m)) * sum(thetaSquare(2:end,:));
J = J + regTerm;

grad = X' * (sigmoid(X * theta) - y) / m;
temp = theta;
temp(1) = 0;
temp = temp * (lambda / m);
grad = grad + temp;


grad = grad(:);

end
