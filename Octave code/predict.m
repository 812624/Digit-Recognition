%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name -> PREDICT
% Working   -> Predict the label of an input returns the index in the network 
%              output that has the largest probability

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function p = predict(Theta1, Theta2, X)
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);% dimensions of p
x1 = [ones(m, 1) X] * Theta1';
h1 = 1.0 ./ (1.0 + exp(-x1));
x2 = [ones(m, 1) h1] * Theta2';
h2 = 1.0 ./ (1.0 + exp(-x2));
[dummy, p] = max(h2, [], 2);


end
