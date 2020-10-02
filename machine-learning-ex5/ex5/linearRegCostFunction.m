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

hypothesis = X*theta;

fprintf("theta\n");
size(theta); % 2행 1열
theta;

% We should not regularize the theta_0 term
theta_reg = [0;theta(2:end, :)];

fprintf("theta_reg\n");
size(theta_reg); % 2행 1열
theta_reg;

J = (1/(2*m)) * sum((hypothesis - y).^2) + (lambda/(2*m))*theta_reg'*theta_reg;

% calculate gradient
grad = (1/m)*(X'*(hypothesis-y)) + (lambda/m)*theta_reg;
fprintf("grad\n");
size(grad) % 2행 1열

% =========================================================================

% row vector 를 column vector 로 바꿔 준다.
% A = [1 2 3 4 ] -> A = [1;2;3;4]
grad = grad(:);

end
