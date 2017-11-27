function [] = pca_recognition(X_train, X_test, y_train, y_test)
        
    % [evectors1, evalues1, meanface1] = eigenfaces(X_train);
    [evectors2, evalues2, meanface2] = eigenfaces_2(X_train);
    A = X_train - meanface2;

    % [normalized_evectors2] ought to be mathematically equal to evectors1
    normalized_evectors2 = A * evectors2 ./ vecnorm(A * evectors2, 2, 1);
    bases = 40;
    
    U = normalized_evectors2(:, 1:bases);
    
    % NN classification
    predicted_test = predict_nn(X_train, y_train, U, X_test);
    test_accuracy = mean(predicted_test == y_test)
    
    predicted_train = predict_nn(X_train, y_train, U, X_train);
    train_accuracy = mean(predicted_train == y_train)
end

function [classes, distances] = predict_nn (X_train, y_train, U, sample)
    training_projections = X_train' * U;
    training_projections = training_projections';
    
    test_projections = (sample' * U)';

    distances = zeros(size(X_train, 2), size(sample, 2));

    for t = 1:size(sample, 2)
        d = (training_projections - test_projections(:, t))';
        distances(:, t) = sum(d .^ 2, 2);
    end
    
    [unused, minimum_indices] = min(distances);
    classes = y_train(minimum_indices);
end