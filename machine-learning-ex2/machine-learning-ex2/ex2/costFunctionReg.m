function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hipotesis = sigmoid(theta' * X')';

#calc first feature
XX = X(:, 1);

grad(1) = XX' * (hipotesis - y) / m;

#calc rest features

#number of theta parameters
z = rows(theta);

XX = X(:, 2:z);
theta_only = theta(2:z, :);

grad(2:z) = XX' * (hipotesis - y) / m + lambda * theta_only / m;

J = (-y' * log(hipotesis) - (1 - y)' * log(1 - hipotesis)) / m + lambda * sum(theta_only .^ 2) / (2 * m);


% =============================================================

end
