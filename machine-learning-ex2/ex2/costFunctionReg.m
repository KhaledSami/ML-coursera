function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

J = 0;
grad = zeros(size(theta));
features = size(X)(1);
regTerm = 0;
for i = 2:length(theta)
	regTerm = regTerm + theta(i)^2;
end
for i = 1:features
	xHypot = calcHypot(X(i,:), theta);
	J = J + (-y(i) * log(xHypot) - (1 - y(i)) * log(1 - xHypot));
end
regTerm = (lambda  / (2 * features)) * regTerm;
J = J / features;
J = J + regTerm;


grad = X' * (sigmoid(X * theta) - y) / features;
for i = 2:length(grad)
	grad(i) = grad(i) + ((lambda / features) * theta(i));
end

end

function hypotVal = calcHypot(X, theta)
hypotVal = sigmoid(X * theta);
end
