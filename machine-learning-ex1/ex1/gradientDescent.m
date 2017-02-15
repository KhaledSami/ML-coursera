function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta);
    % printf('J = %f \n', J_history(iter));
    thetaOne = getThetaUpdatedValue(X, y, theta, X(:, 1), alpha);
    thetaTwo = getThetaUpdatedValue(X, y, theta, X(:, 2), alpha);
    theta(1) = theta(1) - thetaOne;
    theta(2) = theta(2) - thetaTwo;
end
end

function J = getThetaUpdatedValue(X, y, theta, X2, alpha)
m = length(y); 
h = theta' * X';
h = h';
J = alpha * (sum((h - y).* X2) / m);

end
