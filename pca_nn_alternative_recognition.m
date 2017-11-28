function [] = pca_nn_alternative_recognition(X_train, X_test, y_train, y_test)
        
    % [evectors1, evalues1, meanface1] = eigenfaces(X_train);
    [evectors2, evalues2, meanface2] = eigenfaces_2(X_train);
    A = X_train - meanface2;

    % [normalized_evectors2] ought to be mathematically equal to evectors1
    normalized_evectors2 = A * evectors2 ./ vecnorm(A * evectors2, 2, 1);
    
    parfor bases = 1:50
        U = normalized_evectors2(:, 1:bases);

        % NN classification
        predicted_test = predict_nn(X_train, y_train, meanface2, U, X_test);
        test_accuracy = mean(predicted_test == y_test);

        predicted_train = predict_nn(X_train, y_train, meanface2, U, X_train);
        train_accuracy = mean(predicted_train == y_train);
        
        fprintf( ...
          "Number of eigenvectors = %d train_accuracy = %.3f test_accuracy = %.3f\n", ...
          bases, train_accuracy, test_accuracy ...
        );
    
        % Write results to [eigenfaces_plots] to process in bash
        filename = sprintf('eigenfaces_plots/predictions/pca_nn_alt_%d.csv', bases);
        fid = fopen(filename,'w'); 
        fprintf(fid, 'actual,predicted\n');
        fclose(fid);
        M = [y_test; predicted_test];
        M = M';
        dlmwrite(filename, M,'-append');
    end
    
end

function [classes, distances] = predict_nn (X_train, y_train, meanface, U, sample)

    test_projections = (X_train' * U)';
    test_reconstructions = meanface + U * test_projections;

    distances = zeros(size(X_train, 2), size(sample, 2));

    parfor t = 1:size(sample, 2)
        d = (X_train - test_reconstructions(:, t))';
        distances(:, t) = sum(d .^ 2, 2);
    end
    
    [unused, minimum_indices] = min(distances);
    classes = y_train(minimum_indices);
end