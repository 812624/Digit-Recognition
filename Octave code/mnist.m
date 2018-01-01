% This is the main file for mnist data set, has different function calls for respective choices

%================= Declaring initial parameters========================

input_layer_size  = 784;    % 28*28 pixel values
hidden_layer_size = 100;    % 100 hidden units
num_labels = 10;            % 10 labels (0 to 9) 


%================= Load Training Data =================================

fprintf('Loading Data ...\n')
load('mnist_all.mat');

%================= Dividing data into training and testing data========

% Getting the RGB values from dataset
X_data = [
    train0; test0 
    train1; test1 
    train2; test2 
    train3; test3 
    train4; test4 
    train5; test5 
    train6; test6 
    train7; test7 
    train8; test8 
    train9; test9 
];

% Getting the labels from the dataset
y_data = [
    0 * ones(size(train0, 1) + size(test0, 1), 1)
    1 * ones(size(train1, 1) + size(test1, 1), 1)
    2 * ones(size(train2, 1) + size(test2, 1), 1)
    3 * ones(size(train3, 1) + size(test3, 1), 1)
    4 * ones(size(train4, 1) + size(test4, 1), 1)
    5 * ones(size(train5, 1) + size(test5, 1), 1)
    6 * ones(size(train6, 1) + size(test6, 1), 1)
    7 * ones(size(train7, 1) + size(test7, 1), 1)
    8 * ones(size(train8, 1) + size(test8, 1), 1)
    9 * ones(size(train9, 1) + size(test9, 1), 1)
];

% Shuffling data
i = randperm(size(X_data, 1));
X_data = X_data(i, :);
y_data = y_data(i, :);

%========================== Dividing dataset into training and testing dataset=============

m = size(X_data, 1);                                % Getting the row number from X_data
m_test = 10000;         
m_train = m - m_test;                               % m_train becomes 70000-10000 = 60000

X = X_data(1:m_train, :);                           % First 60000 values gets assigned to X 
X_test = X_data(m_train + 1:m_train + m_test, :);   % Left over 10000 values gets assigned to X_test

y = y_data(1:m_train, :);
y_test = y_data(m_train + 1:m_train + m_test, :);


%===================== Randomly select 100 data points to display===============================
%random_data = randperm(size(X, 1));
%random_data = random_data(1:100);
X= double(X);
y=double(y);
X_test= double(X_test);
y_test=double(y_test);

%===================== Visualising random data==================================================
%displayData(X(sel, :));


%=====================Displaying the choices to select from=====================================

fprintf('\t\t\tWelcome to Digit recognition System\n\n');
while true
    fprintf('Pick one option:\n\n');
    fprintf('1. Train Neural Network using the MNIST training set\n');
    fprintf('2. Test the accuracy of the Neural Network on the MNIST test set\n');
    fprintf('3. Test images taken from video camera\n');
    fprintf('3. Test images taken from file\n');
    fprintf('4. STOP\n\n');
    input('Choice: ')

    if ans == 1
        trainNeuralNetwork(X, y)
    elseif(ans == 4)
        break;
    else
        testset = (ans == 2);   % Sets testset 1 if choice is 2
        camera = (ans == 3);    % Sets camera 1 if choice is 3
        file = (ans == 3);      % Sets file 1 if choice is 4
        filename = '';          

        testNeuralNetwork(testset, camera, file, X_test, y_test);
        testNeuralNetwork(testset, file, X_test, y_test);
        fprintf('\n\n--------------- Inside main program ---------------------\n\n');
     end   
end

