%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name -> TRAINNEURALNETWORK 
% Working   -> Trains the neural network and displays accuracy of training set

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function trainNeuralNetwork(X, y)

input_layer_size  = 784;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 100 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

%displayData(X(sel, :));

fprintf('\nInitializing Neural Network Parameters ...\n')

%===================== Randomly initializing Neural Network Parameters===========================

initial_Theta1 = zeros(hidden_layer_size, 1 + input_layer_size);
initial_Theta1 = rand(hidden_layer_size, 1 + input_layer_size) * 2 * 0.12 - 0.12;
initial_Theta2 = zeros(num_labels, 1 + hidden_layer_size);
initial_Theta2 = rand(num_labels, 1 + hidden_layer_size) * 2 * 0.12 - 0.12;


%===================== Unroll parameters=========================================================

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Training Neural Network ==================================================

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);  % Maxitr is the number of iteration we want to run our model for


% Create function handle is a pointer to another function
costFunction = @(p) CostFunction(p, input_layer_size, ...
                                   hidden_layer_size, num_labels, X, y, 0.01);

%====================== Training the Neural Network===============================================

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%====================== Saving the trained network================================================
save trained.mat nn_params;

%============== Getting Theta1 and Theta2 back from nn_params=====================================
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%==================== Predicting the labels for training and test set=============================
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n\n', mean(double(pred == y)) * 100);

end
