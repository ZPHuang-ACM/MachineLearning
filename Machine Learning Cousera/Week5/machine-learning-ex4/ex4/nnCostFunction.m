function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
y_predict = predict_out(Theta1, Theta2, X); % feedforward NN prediction

% y_predict is a mx1 column vector, i.e. [1,2,10,7,..]';
% Here we need to reroll it to [1,0,0..],[0,1,0,..] format
% we can use "sub2ind" function to achieve this
% https://www.mathworks.com/matlabcentral/answers/164993-how-to-use-a-vector-as-indices-for-matrix
 
 % we apply the same procedures to the y vector
 num_columns = numel(y); % # of columns
 num_rows = max(y); % # of rows
 idx = sub2ind([num_rows,num_columns],y',1:num_columns);
 out = zeros(num_rows,num_columns);
 out(idx)=1;
 % unroll the y_predict data to a single long vector
 y_unrolled = out(:); % recored output

% calculate the cost function
J = sum(-y_unrolled.*log(y_predict)-(1-y_unrolled).*log(1-y_predict))/m;

% excluding the bias gains
Theta1_unbiased = Theta1(:,2:end);
Theta2_unbiased = Theta2(:,2:end);

% regularization cost function
J_regularization = lambda*(sum(sum(Theta1_unbiased.^2)) + sum(sum(Theta2_unbiased.^2)))/(2*m);

% final cost function
J = J+J_regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
for i = 1:m
    % step 1, forward prop
    a1 = X(i,:)'; a1 = [1;a1]; % add 1, bias
    size(a1);
    z2 = Theta1*a1; a2 = sigmoid(z2); a2 = [1;a2]; % add 1, bias
    z3 = Theta2*a2; a3 = sigmoid(z3); % the hypothesis output
    
    % step 2, delta 3
    yi = zeros(num_labels,1); % convert to [0,0,1,0,..]' format
    yi(y(i)) = 1;
    delta3 = a3 - yi;
    
    % step 3, delta 2
    delta2 = Theta2'*delta3.*[0;sigmoidGradient(z2)];
    delta2 = delta2(2:end); % remove delta2_0 term, important!
    
    % step 4, Accumulate the gradient
    Theta1_grad = Theta1_grad + delta2*a1';
    Theta2_grad = Theta2_grad + delta3*a2';
end

% step 5
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% =========================================================================

% gradient regularization 
Theta1_regularization = Theta1*lambda/m; 
Theta1_regularization = [zeros(size(Theta1_regularization,1),1),Theta1_regularization(:,2:end)];
Theta2_regularization = Theta2*lambda/m; 
Theta2_regularization = [zeros(size(Theta2_regularization,1),1),Theta2_regularization(:,2:end)];

Theta1_grad = Theta1_grad + Theta1_regularization;
Theta2_grad = Theta2_grad + Theta2_regularization;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
