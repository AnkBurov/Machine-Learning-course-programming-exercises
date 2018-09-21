function [error_train, error_val] = learningCurveRandom(X, y, Xval, yval, lambda)
  
% Number of training examples
m = size(X, 1);

number_of_iterations = 5;

% You need to return these values correctly
error_train = zeros(m, number_of_iterations);
error_val   = zeros(m, number_of_iterations);   
  
  
for iteration = 1:number_of_iterations
  #randomize train set
  Train_set_random = [X y];
  Train_set_random = Train_set_random(randperm(size(Train_set_random, 1)), :);
  X_random = Train_set_random(:, 1:end - 1);
  y_random = Train_set_random(:, end);

  #randomize validation set
  Val_set_random = [Xval yval];
  Val_set_random = Val_set_random(randperm(size(Val_set_random, 1)), :);
  Xval_random = Val_set_random(:, 1:end - 1);
  yval_random = Val_set_random(:, end);  
  
  [error_train_rand, error_val_rand] = learningCurve(X_random, y_random, Xval_random, yval_random, lambda);
  error_train(:, iteration) = error_train_rand;
  error_val(:, iteration) = error_val_rand;
  
  #function [error_train, error_val] = ...
  #  learningCurve(X, y, Xval, yval, lambda)
  
endfor
  
3 + 3;
   

endfunction
