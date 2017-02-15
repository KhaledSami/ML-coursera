function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));
for i = 1:size(g)(1)
	for j = 1:size(g)(2)
		g(i, j) = calcSigmoid(z(i, j));
	end
end
end

function sig = calcSigmoid(z)
sig =  1 /  (1 + exp(-1 * z));
end