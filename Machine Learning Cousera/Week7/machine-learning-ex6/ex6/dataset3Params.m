function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vec = [0.01,0.03,0.1,0.3,1.3,10,30];
sigma_vec = [0.01,0.03,0.1,0.3,1.3,10,30];
n = numel(C_vec);
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


% Initialization, 1st step
model= svmTrain(X, y, C_vec(1), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(1)));
predictions = svmPredict(model, Xval);
min_error = mean(double(predictions ~= yval));
C = C_vec(1);
sigma = sigma_vec(1);

% Train the SVM
for i = 1:n
    for j=1:n
        % train the model using trining data set
        model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        % predict the model using cross-validation data set
        predictions = svmPredict(model, Xval);
        % get the model mismatch percentage, we want the model with the smallest cross-validation
        % mismatch rate
        mismatch = mean(double(predictions ~= yval));
        if mismatch < min_error
            % if we find a set C, sigma that can generate a smaller mismatch
            % we assign these numbers to C and sigma
            min_error = mismatch;
            C = C_vec(i);
            sigma = sigma_vec(j);
        end
    end
end

% =========================================================================
end
