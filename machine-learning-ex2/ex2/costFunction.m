function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
features = size(X)(1);
for i = 1:features
	xHypot = calcHypot(X(i,:), theta);
	J = J + (-y(i) * log(xHypot) - (1 - y(i)) * log(1 - xHypot));
end
J = J / features;

grad = X' * (sigmoid(X * theta) - y) / features;

function hypotVal = calcHypot(X, theta)
hypotVal = sigmoid(X * theta);
end
