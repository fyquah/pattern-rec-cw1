function [] = pca_nn_draw_incorrect(X_train, X_test, y_train, y_test)
    [evectors2, evalues2, meanface2] = eigenfaces_2(X_train);
    A = X_train - meanface2;
    normalized_evectors2 = A * evectors2 ./ vecnorm(A * evectors2, 2, 1);

    U = normalized_evectors2(:, 1:101);
    interesting_indices= [ 2 , 21 , 51 ];
    samples = X_test(:, interesting_indices);
    
    [predicted_test, predicted_distances] = predict_nn(X_train, y_train, U, samples);
    size(predicted_distances)
    
    for p = 1:size(interesting_indices, 2)
        compressed_distance = zeros(1, 52);

        for k = 1:52
            l = 1 + (k - 1) * 7;
            r = 7 + (k - 1) * 7;
            compressed_distance(k) = min(predicted_distances(l:r, p)); 
        end
        
        [ unused, indices ] = sort(predicted_distances(:, p));
        d = predicted_distances(:, p);
        indices = indices(1:10);

        figure;
        c = cellstr(num2str(indices));
        b = bar(1:10, d(indices));
        b.FaceColor = 'flat';
        aaa = (y_train(indices) == predicted_test(p));
        b.CData(aaa, :) = repmat([.5 0 0], sum(aaa), 1);
        
        aaa = (y_train(indices) == y_test(interesting_indices(p)));
        b.CData(aaa, :) = repmat([0 .5 0], sum(aaa), 1);

        
    end
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