% Should be only run once

load("face.mat");
[X_train, X_test, y_train, y_test] = get_training_data(X, 0.7, 10);
save("data_partition.mat", 'X_train', 'X_test', 'y_train', 'y_test');