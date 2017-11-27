function [X_train, X_test, y_train, y_test] = get_training_data(X, training_ratio, P)
    %GET_TRAINING_DATA Summary of this function goes here
    %   [P] :frequency of every label
    
    M = size(X, 1);  % number of features
    N = size(X, 2);  % length of dataset
    L = N/P; %number of labels

    y_train = repmat(0:1:L, P * training_ratio, 1); % L*train_size matrix
    y_train = y_train(:)'; % Collapse matrix
    y_test = repmat(0:1:L, P - (P * training_ratio), 1);
    y_test = y_test(:)';
    
    X_train = zeros(M, size(y_train, 2));
    X_test = zeros(M, size(y_test, 2));

    train_size = P * training_ratio;
    test_size = P - train_size;
    
    for k = 1:L
        idx = (k - 1) * P;
        [i_train, i_test] = dividerand(P, training_ratio, 1-training_ratio, 0.0);
        
        i_train = i_train + idx;
        i_test = i_test + idx;
        
        X_train(:, 1 + (k - 1) * train_size:k * train_size) = X(:, i_train);
        X_test(:, 1 + (k - 1) * test_size:k * test_size) = X(:, i_test);
    end
end

