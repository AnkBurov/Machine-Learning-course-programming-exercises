function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X * theta;

#calc regularization modifier
theta_only = theta(2:end, :);
reg_modifier = lambda * sum(theta_only .^ 2) / (2 * m);

#Calc cost
J = 1 / (2 * m) * sum((hypothesis - y) .^ 2) + reg_modifier;


# calc gradient
grad(1) = 1 / m * X(:, 1)' * (hypothesis - y);

grad(2:end) = 1 / m * X(:, 2:end)' * (hypothesis - y) + lambda * theta_only / m;

# normal equa
#eq_theta = pinv(X' * X) * X' * y;





% =========================================================================

grad = grad(:);

end
