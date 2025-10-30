function vec_X = vec(X)
    % VEC takes as input a tensor of any order and returns its
    % vectorization, i.e. a column vector.
    % INPUT tensor X.
    % OUTPUT vector vec_X.

    output_size = [numel(X), 1];
    vec_X = reshape(X, output_size);
end

% function red_X = remove_one_order(X)
%     sz = size(X);
%     last_dimension = sz(end);
%     output_size = sz(1:end-1);
%     output_size(end) = output_size(end) * last_dimension;
%     red_X = reshape(X, output_size);
% end