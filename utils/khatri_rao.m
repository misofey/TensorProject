function Z = khatri_rao(varargin)
    % KHATRI_RAO takes a list of matrices and returns the (right)
    % Khatri-Rao product.
    % INPUT list of variable number of matrices varargin.
    % OUTPUT outer product outer.

    Z = varargin{1};
    for i=2:length(varargin)
        Z = khatri_rao_single(Z, varargin{i});
    end
end

function prod = khatri_rao_single(A, B)
    sza = size(A);
    szb = size(B);

    B = repmat(B, [sza(1) 1]);
    A = A(repelem(1:sza(1), szb(1)), :);
    
    % A = repmat(A, [szb(1) 1]);
    % B = B(repelem(1:szb(1), sza(1)), :);
    prod = A.*B;
end