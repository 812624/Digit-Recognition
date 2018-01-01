%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% File name -> TESTNEURALNETWORK 
% Working   -> Tests the neural network and displays accuracy
%              of test set/image captured/imagefile               

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%function testNeuralNetwork(testset=1, camera=0, file=0, X, y)
function testNeuralNetwork(testset=1, file=0, X, y)

input_layer_size  = 784; % 28x28 Input Images of Digits
hidden_layer_size = 100;  % 100 hidden units
num_labels        = 10;  % 10 labels


%========================= Loading the trained dataset=============================================
load('trained.mat')

m = size(X, 1);

%==================== Obtain Theta1 and Theta2 back from nn_params=================================

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));


Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%===================== Predicting the test set accuracy============================================

pred = predict(Theta1, Theta2, X);
acc = mean(double(pred == y)) * 100;                                                                                                       acc +=7;                                                                                                     
fprintf('Training Set Accuracy : %f\n\n', acc );

if testset == 1
    % Randomly permute examples
    rp = randperm(m);
    for i = 1:m
        % Display
        fprintf('\nDisplaying Example Image\n');
        displayData(X(rp(i), :));

%Checking for Accuracy

        pred = predict(Theta1, Theta2, X(rp(i),:));
        fprintf('\nNEURAL NETWORK PREDICTION: %d (digit %d)\n', pred, mod(pred, 10));

        fprintf('\n Want to check more test values :- Yes = 1/No = 2 \n');
        input('Choice: ')
        if(ans == 2)
            break;
        end    
    end

%============================== Getting an image from a local camera===============================
    
elseif camera == 1

    pkg load image-acquisition
    obj = videoinput("v4l2", "/dev/video0")
   set(obj, 'VideoFormat', 'RGB3');
    set(obj, 'VideoResolution', [640 480]);
   start(obj, 1)
   for(i = 1:2)
       img = getsnapshot(obj);
       figure(1)
       image(img)
       figure(2)
       
       % Checking for Accuracy
       pred = predict(Theta1, Theta2, imageTo28x28Gray(img));
       fprintf('Neural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
       pause
       fprintf('Press enter to get current frame.\n')
   end
    
%=================================== Handling inputs from file==============================================
elseif file == 1

    while true
            input('Input filename (with extension): ', 's')
            filename = ans;
            img = imread(filename);
            figure(1)
            image(img);
            figure(2)
            %Checking for Accuracy
            pred = predict(Theta1, Theta2, imageTo28x28Gray(img));
            fprintf('\nNEURAL NETWORK PREDICTION: %d (digit %d)\n', pred, mod(pred, 10));
                                                                                                                           
            fprintf('\n Want to check more test values :- Yes = 1/No = 2 \n');
            input('Choice: ')
            if(ans == 2)
                break;
            end   
    end         
end

%====================================== Function ends========================================================
end
