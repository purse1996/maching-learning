function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% 在训练集中训练参数，在剩下集合验证
%
error = 1;
C_example = [0.01 0.1 1 10];
sigma_example = [0.01 0.1 0.3 1 2];
for i=1:length(C_example)
    for j=1:length(sigma_example)
        model = svmTrain(X,y,C_example(i),@(x1,x2) gaussianKernel(x1,x2,sigma_example(j)));
        predictions = svmPredict(model,Xval);
        error_now = mean(double(predictions~=yval));
        if error_now<error
            error = error_now;
            C = C_example(i);
            sigma = sigma_example(j);
        end
    end
end
    



% =========================================================================

end
