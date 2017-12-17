function [all_accuracies, C_test, predicted_test] = pca_nn_recognition(X_train, X_test, y_train, y_test, bases_list)
        
    % [evectors1, evalues1, meanface1] = eigenfaces(X_train);
    [evectors2, evalues2, meanface2] = eigenfaces_2(X_train);
    A = X_train - meanface2;

    % [normalized_evectors2] ought to be mathematically equal to evectors1
    normalized_evectors2 = A * evectors2 ./ vecnorm(A * evectors2, 2, 1);
    all_accuracies = zeros(size(X_test, 2), 1);
    
    if nargin == 4
       bases_list = 1:364; 
    end
    
    for bases = bases_list
        U = normalized_evectors2(:, 1:bases);

        % NN classification
        predicted_test = predict_nn(X_train, y_train, U, X_test);
        test_accuracy = mean(predicted_test == y_test);
        all_accuracies(bases) = test_accuracy;

        predicted_train = predict_nn(X_train, y_train, U, X_train);
        train_accuracy = mean(predicted_train == y_train);
        
        fprintf( ...
          "Number of eigenvectors = %d train_accuracy = %.3f test_accuracy = %.3f\n", ...
          bases, train_accuracy, test_accuracy ...
        );
    
        % Write results to [eigenfaces_plots] to process in bash
        filename = sprintf('eigenfaces_plots/predictions/pca_nn_%d.csv', bases);
        
        [C_test, order] = confusionmat(y_test', predicted_test');
%         fid = fopen(filename,'w'); 
%         fprintf(fid, 'actual,predicted\n');
%         fclose(fid);
%         M = [y_test; predicted_test];
%         M = M';
%         dlmwrite(filename, M,'-append');
    end
    
    if nargin == 4
        plot(all_accuracies);
        grid;
        xlabel('Number of components');
        title('Number of components vs Test accuracy');
        ylabel('Test accuracy');
        print('eigenfaces_plots/pca_nn_test_accuracy','-deps');
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