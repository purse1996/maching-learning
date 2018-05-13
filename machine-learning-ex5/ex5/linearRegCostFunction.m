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
% 2*1
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X 12*2输入的时候已经加过1了

J = 1/2/m*sum((X*theta-y).^2)+lambda/m/2*(theta'*theta-theta(1)^2);

%  注意下面式子第一个theta中[1]不为0，只有第二个才为0，故不能在上面修改theta的值
mask = ones(size(theta));
mask(1) = 0;

% 你必须学会自己设计矩阵形式
a = theta.*mask;
grad = 1/m*((X*theta-y)'*X)'+ lambda/m*a;












% =========================================================================

grad = grad(:);

end
