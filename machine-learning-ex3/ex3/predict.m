function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % number of training examples. There are 5000 examples
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

fprintf("size(X, 1)\n");
size(X, 1)

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

fprintf("Theta1\n");
size(Theta1)  % 25행 401열

fprintf("Theta2\n");
size(Theta2)  % 10행 26열

fprintf("p\n");
size(p)  % 1행 1열

fprintf("X:\n");
size(X)  % 1행 400열

a1 = [ones(m, 1) X];
size(a1) % 1행 401열

z2 = a1*Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
size(a2) % 1행 26열

z3 = a2*Theta2';
a3 = sigmoid(z3);
size(a3) % 1행 10열

fprintf('max(a3, [], 2):\n');
a3 % [ 0.0016849121, 0.0519500154, 0.9731121954, 0.0000080549, 0.0019211425, 0.0000424914, 0.0009749237, 0.0234547574, 0.0003344442, 0.0017040597 ]

[p_max, p] = max(a3, [], 2);

p_max % 0.97311
p     % 3

% =========================================================================


end
