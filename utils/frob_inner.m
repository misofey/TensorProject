function z = frob_inner(X,Y)
    % INNER takes as input tensor X and tensor Y of equal sizes and of any order and returns 
    % their Frobenius inner product.
    % INPUT tensor X, Y.
    % OUTPUT scalar z.

    z = X(:)' * Y(:);
end