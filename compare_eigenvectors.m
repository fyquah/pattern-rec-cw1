function [] = compare_eigenvectors(X_train, X_test, flag_save)

    % This is for {Application of Eigenfaces, part (A)}
    if nargin == 2
        flag_save = false;
    end
    
    % [evectors1, evalues1, meanface1] = eigenfaces(X_train);
    [evectors2, evalues2, meanface2] = eigenfaces_2(X_train);
    A = X_train - meanface2;

    % [normalized_evectors2] ought to be mathematically equal to evectors1
    normalized_evectors2 = A * evectors2 ./ vecnorm(A * evectors2, 2, 1);

    % test up to 50 bases?
    reconstruction_bases = 100;
    training_reconstruction_errors = zeros(reconstruction_bases, 1);
    test_reconstruction_errors = zeros(reconstruction_bases, 1);
    N = size(X_train, 2);

    for bases = 1:reconstruction_bases
        U = normalized_evectors2(:, 1:bases);

        % on training data
        projection = X_train' * U;
        projection = projection';
        reconstructed = meanface2 + U * projection;
        training_reconstruction_errors(bases) = 1 / N * norm(X_train - reconstructed);

        % on test data
        projection = (X_test' * U)';
        reconstructed = meanface2 + U * projection;
        test_reconstruction_errors(bases) = 1 / N * norm(X_test - reconstructed);
    end

    figure
    plot(1:reconstruction_bases, training_reconstruction_errors);
    hold on
    plot(1:reconstruction_bases, test_reconstruction_errors);
    legend("training", "test");
    title("Number of eigenvectors vs Reconstruction error");
    xlabel("Number of eigenvectors");
    ylabel("Reconstruction error (Norm2)");
    
    if flag_save
       print('eigenfaces_plots/reconstruction_error_plot','-depsc');
    end
end