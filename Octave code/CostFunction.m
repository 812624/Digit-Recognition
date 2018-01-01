%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name -> NNCOSTFUNCTION
% Working   -> Computes the cost and gradient of the neural network

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [J, grad] = CostFunction(nn_params, input_layer_size,...
                                    hidden_layer_size, num_labels, X, y, lambda)


%============== Getting Theta1 and Theta2 back from nn_params=====================================
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
h = size(Theta1, 1);
r = num_labels;
                      
y_matrix = bsxfun(@eq, y, 1:num_labels);

%========================= Feedforward the neural network==================================

% Activation of the units in the layer 1
a1 = [ones(m,1) X];

% Activation of units in the layer 2
z2 = a1*Theta1';
g2 = 1.0 ./ (1.0 + exp(-z2));
a2 = g2;                         % Sigmoid function as activation function
a2 = [ones(m,1) a2];

%  Activation of units in layer 3
z3 = a2*Theta2';
g3 = 1.0 ./ (1.0 + exp(-z3));
a3 = g3;                         % Sigmoid function as activation function

Cost = sum(sum(  -y_matrix.*log(a3) - (1 - y_matrix).*log(1 - a3)  ));
Reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = (1/m)*Cost + Reg;

%====================== Implementing the backpropagation algorithm==========================

d3 = a3 - y_matrix;
x = g2.*(1-g2);
d2 = d3*Theta2(:,2:end) .* x;

Delta1 = d2'* a1;
Delta2 = d3' * a2;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

% Add UNREGULARIZED gradients to the REGULARIZED gradients

Theta1_grad = (1/m)*Delta1 + (lambda/m)*Theta1;
Theta2_grad = (1/m)*Delta2 + (lambda/m)*Theta2;



%========================== Unrolling gradients==============================================
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
