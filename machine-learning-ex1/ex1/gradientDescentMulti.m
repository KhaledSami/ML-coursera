function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta);
    % printf('J = %f \n', J_history(iter));
    thetaPrv = theta;
    for i = 1: length(theta)
        thetaPrv(i) = getThetaUpdatedValue(X, y, theta, X(:, i), alpha);
    end
    for i = 1: length(theta)
        theta(i) = theta(i) - thetaPrv(i);
    end
end
end

function J = getThetaUpdatedValue(X, y, theta, X2, alpha)
m = length(y); 
h = theta' * X';
h = h';
J = alpha * (sum((h - y).* X2) / m);

end
