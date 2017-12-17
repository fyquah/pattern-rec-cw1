function [C_test] = pca_recognition(X_train, X_test, y_train, y_test)
  [dont_care, C_test] = pca_nn_recognition(X_train, X_test, y_train, y_test, 104:104);
  A = mat2gray(C_test );
  imshow(A);

  dlmwrite('eigenfaces_plots/best_confusion_matrix.csv', C_test);
  imwrite(mat2gray(C_test), 'eigenfaces_plots/best_confusion_matrix.png');
end