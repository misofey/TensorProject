function [] = tensor_reconstruction()
%TENSOR_RECONSTRUCTION Summary of this function goes here
%   Detailed explanation goes here
    [image, M, N] = load_image();
    n = ndims(image);
    addpath("utils/");
    [omega, omega_overline, removed_idx, kept_idx, lin_rem_idx] = create_masks(0.1);
    X_0 = double(image);
    X_0(lin_rem_idx) = 0;
    X_1 = mode_n_matricization(X_0, 1);
    X_2 = mode_n_matricization(X_0, 2);
    X_3 = mode_n_matricization(X_0, 3);
    alphas = ones(1, n) / n;
    betas = ones(1, n) / 100;
    rho = 0.01;
    T = {mode_n_matricization(image, 1), mode_n_matricization(image, 2), mode_n_matricization(image, 3)};
    M = {X_1, X_2, X_3};
    X = X_0;

    % [reconstructed, errors] = silrtc(M, X, alphas, betas, n, lin_rem_idx, 100, T, image);

    Y = {X*0, X*0, X*0};
    M_halrtc = {X_0, X_0, X_0};  % these are not matricized
    [reconstructed, errors] = halrtc(M_halrtc, X, Y, alphas, rho, n, lin_rem_idx, 100, T, image);
end

function [image, M, N] = load_image()
    image = imread("image1.tiff", "tiff");
    % image = image(1:30, 1:30, :);
    sz = size(image);
    M = sz(1);
    N = sz(2);
end

function [omega, omega_overline, removed_idx, kept_idx, lin_rem_idx] = create_masks(removed_percentage)
    omega = [];
    omega_overline = [];
    removed_idx = [];
    kept_idx = [];
    image_size = 512;
    pixel_count = image_size*image_size;
    no_removed = round(pixel_count*removed_percentage);
    removed_flattened = randperm(pixel_count, no_removed);  % idx to remove from the flattened array
    removed_idx = zeros(no_removed, 2,  "int32");
    [removed_idx(:, 1), removed_idx(:, 2)] = ind2sub([image_size, image_size], removed_flattened);
    omega = ones(image_size);
    omega(removed_idx(:, 1), removed_idx(:, 2)) = 0;
    omega_overline = 1-omega;
    [rk, ck] = find(omega(:,:) == 1);
    kept_idx = int64([rk ck]);
    lin_rem_idx = [removed_flattened, removed_flattened, removed_flattened];
end
    
function [X, errors] = silrtc(M, X, alphas, betas, n, lin_rem_idx, K, T, image)
    % idx removed are the pixels removed from the image
    errors = ones(K, 4);
    for j = 1:K
        disp("beginning iteration");
        selected = zeros(length(lin_rem_idx),1)';  % for collecting the average of all versions of the optimizaiton variables
        for i = 1:n
            disp(["beginning update of block matrix", num2str(n)]);
            drawnow('update');
            M{i} = shrinkage(mode_n_matricization(X, i), alphas(i)/betas(i));
            folded = mode_n_folding(M{i}, i, size(X));
            selected = selected + folded(lin_rem_idx) * betas(i);

        end

        X(lin_rem_idx) = selected / sum(betas);
        for asdf = 1:3
            errors(j, asdf) = frob_norm(double(T{asdf})-M{asdf});
        end
        errors(j, 4) = frob_norm(double(image) - X);
    end
end

function [X, errors] = halrtc(M, X, Y, alphas, rho, n, lin_rem_idx, K, T, image)
    X_things = zeros(size(X));  % for collecting the average of all versions of the optimizaiton variables
    errors = ones(K, 4);

    for j = 1:K
        X_things = zeros(size(X));
        for i = 1:n
            M{i} = mode_n_folding(shrinkage(mode_n_matricization(X, i) + mode_n_matricization(Y{i}, i) / rho, alphas(i) / rho), i, size(X));
            X_things = X_things + M{i} - Y{i} / rho;
        end
        X(lin_rem_idx) = X_things(lin_rem_idx) / n;
        for i = 1:n
            Y{i} = Y{i} - (M{i} - X) * rho;
        end

        for asdf = 1:3
            errors(j, asdf) = frob_norm(double(image)-M{asdf});
        end
        errors(j, 4) = frob_norm(double(image) - X);
    end
end

function [D] = shrinkage(X, tau)
    %% rounds tau down to the lowest nu
    [U, S, V] = svd(X, "econ");
    S_tau = diag(max(diag(S) - tau, 0));
    D = U * S_tau * V';
end

function [T] = truncation(X, tau)
    [U, S, V] = svd(X);
    S_tau = max(S, tau);
    T = U * S_tau * V';
end
