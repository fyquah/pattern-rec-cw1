load data_partition.mat

validation_size = 52;
bag_size = 7;
arr_num_eigen_bases = 40:51;
arr_y_guess = zeros(length(arr_num_eigen_bases), size(y_test, 2));
arr_all_validation_accuracies = zeros(length(arr_num_eigen_bases), 40);
arr_all_test_accuracies = zeros(length(arr_num_eigen_bases), 40);

for hyper_param_index = 1:length(arr_num_eigen_bases)
    num_eigen_bases = arr_num_eigen_bases(hyper_param_index);
    N = 52;
    train_samp = 7;
    test_samp = 3;
    feature_len = size(X_train,1);
    means = zeros(N,feature_len);
    X_t = X_train';
    stripped_X_train = zeros(size(X_train, 1), size(X_train, 2));

    S_i = cell(N,1);
    S_w = zeros(feature_len,feature_len);
    
    y_validation = zeros(validation_size, 1);
    X_validation = zeros(validation_size, size(X_t, 2));

    for i = 1:N
       idx = 1+(i-1)*train_samp;
       means(i,:) = mean(X_t(idx:idx+6,:));
       S_i(i) = {zeros(feature_len,feature_len)};
       training_indices = randperm(7, 7);
       validation_index = training_indices(1);
       training_indices = training_indices(2:7);
       X_validation(i, :) = X_t(idx + validation_index - 1, :);
       y_validation(i, 1) = i;
       
       for j = datasample(training_indices, bag_size)
           assert(j ~= validation_index);
           stripped_X_train(:, idx + j - 1) = X_train(:, idx + j - 1);
            x_del = X_t(idx+j-1,:)-means(i,:);
            S_x =  x_del'*x_del;
            S_i(i) = {S_i{i} + S_x};
       end
       S_w = S_w + S_i{i};
    end

    meanface = mean(means);

    S_b = zeros(feature_len,feature_len);

    for i = 1:N
        meanx_del = means(i, :) - meanface;
        S_b = S_b + meanx_del'*meanx_del;
    end

    % PCA stuff
    [evectors2, evalues2, meanface2] = eigenfaces_2(stripped_X_train);
    A = stripped_X_train - meanface2;
    normalized_evectors2 = A * evectors2 ./ vecnorm(A * evectors2, 2, 1);
    U = normalized_evectors2(:, 1:num_eigen_bases);

    S_w_pca = U'*S_w*U;
    S_b_pca = U'*S_b*U;

    fprintf(">>= num_eigen_bases = %d :\n", num_eigen_bases);
    
    cur_y_guess = zeros(num_eigen_bases, size(y_test, 2));
    cur_accuracies = zeros(num_eigen_bases, 1);
    cur_test_accuracies = zeros(num_eigen_bases, 1);
    y_validation = y_validation';

    for lda_dim = 1:num_eigen_bases
      the_real_lda_essemble;
      cur_y_guess(lda_dim, :) = y_test_guess;
      
      test_accuracy = sum(y_test_guess == y_test) / length(y_test);
      validation_accuracy = ...
          sum(y_validation_guess == y_validation) / length(y_validation);
      cur_accuracies(lda_dim, :) = validation_accuracy;
      cur_test_accuracies(lda_dim, :) = test_accuracy;
                
      if lda_dim <= 40
          arr_all_validation_accuracies(num_eigen_bases, lda_dim) = ...
              validation_accuracy;
          arr_all_test_accuracies(num_eigen_bases, lda_dim) = test_accuracy;
      end
    end
    
    [cur_best_accuracy, cur_best_index] = max(cur_accuracies);
    arr_y_guess(hyper_param_index, :) = cur_y_guess(cur_best_index, :);

    fprintf(" CHOSEN: %d bases (validation = %.3f, test = %.3f)\n", ...
        cur_best_index, cur_best_accuracy, cur_test_accuracies(cur_best_index));
end