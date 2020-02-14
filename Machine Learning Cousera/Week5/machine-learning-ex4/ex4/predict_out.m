function p = predict_out(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%  Notice "predict_out" is different from "predict", the output in this function
% the original output on each output unit, not the converted k-element vector shown in
% "predict" function
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% # of trainning examples
m = size(X, 1);

% You need to return the following variables correctly, consider a three layer network
h1 = sigmoid([ones(m, 1) X] * Theta1'); % activation value on the hidden layer
h2 = sigmoid([ones(m, 1) h1] * Theta2'); % activation value on the output layer 

intermediate = h2';
p = intermediate(:);

% =========================================================================

end
