clear ; close all; clc

fprintf('Loading data ...\n');

data = load('train_headless.csv');

# 1-15 dataset
X_train = data(:, 2:14);
% Add intercept term to X
X_train = [ones(length(X_train), 1) X_train];

y_train = data(:, 15);

fprintf('Solving with normal equations...\n');

theta = normalEqn(X_train, y_train);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

# test dataset
fprintf('Predicting test dataset \n');

test_data = load('test_headless.csv');

ids = test_data(:,1);

X_test = test_data(:, 2:14);
X_test = [ones(length(X_test), 1) X_test];

predictions = X_test * theta;

Predicted = [ids predictions];

#plot X2 and y

#plot(X_train(:, 2), y_train, 'rx', 'MarkerSize', 10);
#ylabel('Profit in $10,000s');
#xlabel('crime rate by town');

## gradient descent

% Choose some alpha value
alpha = 0.000006;
num_iters = 200;

theta = zeros(14, 1);
[theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

predictions_gradient = X_test * theta;

Predicted_gradient = [ids predictions_gradient];

## figure;
figure;
#plot(1:numel(y_train), y_train, '-b', 'LineWidth', 2);
y_train = sort(y_train);
plot(1:numel(y_train), y_train, '-b', 'LineWidth', 2);
xlabel('index in matrice');
ylabel('Cost');
# try polinomian regression