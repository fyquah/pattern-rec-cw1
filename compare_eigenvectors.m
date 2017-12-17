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
    reconstruction_bases = 364;
    training_reconstruction_errors = zeros(reconstruction_bases, 1);
    test_reconstruction_errors = zeros(reconstruction_bases, 1);
    N = size(X_train, 2);
    
    should_draw = zeros(364, 1);
    should_draw(1) = 1;
    should_draw(2) = 1;
    should_draw(4) = 1;
    should_draw(8) = 1;
    should_draw(16) = 1;
    should_draw(32) = 1;
    should_draw(64) = 1;
    should_draw(128) = 1;
    should_draw(256) = 1;
    should_draw(364) = 1;
    
    image_indices = [ 20, 50, 100 ];
    images_to_draw = length(image_indices);
    num_bases_to_draw = sum(should_draw);
    
    ctr = 0;
    train_outputs = zeros(images_to_draw, num_bases_to_draw, 56 * 46);
    test_outputs = zeros(images_to_draw, num_bases_to_draw, 56 * 46);
    

    for bases = 1:reconstruction_bases
        U = normalized_evectors2(:, 1:bases);

        % on training data
        projection = ((X_train - meanface2)' * U)';
        train_reconstructed = meanface2 + U * projection;
        training_reconstruction_errors(bases) = mean(vecnorm(X_train - train_reconstructed, 2, 2));

        % on test data
        projection = ((X_test - meanface2)' * U)';
        test_reconstructed = meanface2 + U * projection;
        test_reconstruction_errors(bases) = mean(vecnorm(X_test - test_reconstructed, 2, 2));
        
        if should_draw(bases)
            ctr = ctr + 1;
            p = 1;
            for idx = image_indices
                train_outputs(p, ctr, :) = train_reconstructed(:, idx);
                test_outputs(p, ctr, :) = test_reconstructed(:, idx);
                p = p + 1;
            end
        end
    end
    
    train_thing = zeros(images_to_draw, 56, (46 + 1) * (num_bases_to_draw + 1));
    test_thing = zeros(images_to_draw, 56, (46 + 1) * (num_bases_to_draw + 1));
    
    for p = 1:images_to_draw
        train_thing(p, :, 1:46) = reshape(X_train(:, image_indices(p)), 56, 46);
        for b = 1:num_bases_to_draw
            l = 1 + 47 * b;
            r = 46 + 47 * b;
            train_thing(p, :, l:r) = reshape(squeeze(train_outputs(p, b, :)), 56, 46);
        end
        prefix = sprintf('eigenfaces_plots/reconstruction_strips/train_strip_%u.png', p);
        imwrite(mat2gray(squeeze(train_thing(p, :, :))), prefix);
        
        test_thing(p, :, 1:46) = reshape(X_test(:, image_indices(p)), 56, 46);
        for b = 1:num_bases_to_draw
            l = 1 + 47 * b;
            r = 46 + 47 * b;
            test_thing(p, :, l:r) = reshape(squeeze(test_outputs(p, b, :)), 56, 46);
        end
        prefix = sprintf('eigenfaces_plots/reconstruction_strips/test_strip_%u.png', p);
        imwrite(mat2gray(squeeze(test_thing(p, :, :))), prefix);
    end


    figure
    plot(1:reconstruction_bases, training_reconstruction_errors);
    hold on
    plot(1:reconstruction_bases, test_reconstruction_errors, '--');
    legend("training", "test");
    title("Number of components vs Reconstruction error");
    xlabel("Number of components ");
    ylabel("Reconstruction error (Norm2)");
    grid;
    
    if flag_save
       print('eigenfaces_plots/reconstruction_error_plot','-deps');
    end
end