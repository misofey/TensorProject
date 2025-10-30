function Z = mode_n_product(X,Y,n)
    % MODE_N_PRODUCT takes tensor X and compatible matrix Y and performs mode-n product between X and Y.
    % INPUT tensor X, matrix Y.
    % OUTPUT tensor Z.
    
    og_shape = size(X);
    unfolded = mode_n_matricization(X, n);
    mutliplied = Y * unfolded;
    Z = mode_n_folding(mutliplied, n, og_shape);
end

function [unfolding_permutation, folding_permutation] = mode_n_permutations(n, og_size)
    order = length(og_size);
    unfolding_permutation = [n 1:n-1 n+1:order];
    folding_permutation = [2:n 1 n+1:order];
end

function [X] = mode_n_folding(X, n, og_size)
    [unfolding_permutation, folding_permutation] = mode_n_permutations(n, og_size);
    permuted_shape = og_size(unfolding_permutation);
    a = size(X);
    permuted_shape(1) = a(1);
    X = reshape(X, permuted_shape);
    X = permute(X, folding_permutation);
end
