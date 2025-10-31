% function [] = tensor_reconstruction()
%TENSOR_RECONSTRUCTION Summary of this function goes here
%   Detailed explanation goes here
rng(1);
    [I, Ms, N] = load_image();
    I = double(I) / 255;
    n = ndims(I);
    addpath("utils/");
    keep_ratio = 0.9;
    [omega, omega_overline, removed_idx, kept_idx, lin_rem_idx, log_rem] = create_masks(keep_ratio);
    X_0 = double(I);
    Iobs = X_0;
    X_0(lin_rem_idx) = 0;
    Iobs(log_rem) = 0;
    X_1 = mode_n_matricization(X_0, 1);
    X_2 = mode_n_matricization(X_0, 2);
    X_3 = mode_n_matricization(X_0, 3);
    alphas = ones(1, n) / n;
    betas = ones(1, n) / 100;
    rho = 0.01;
    T = {mode_n_matricization(I, 1), mode_n_matricization(I, 2), mode_n_matricization(I, 3)};
    M = {X_1, X_2, X_3};
    X = X_0;

    % [reconstructed, errors] = silrtc(M, X, alphas, betas, n, log_rem, 100, T, image);

    Y = {X*0, X*0, X*0};
    M_halrtc = {X_0, X_0, X_0};  % these are not matricized
    [Xhat, errors] = halrtc(M_halrtc, X, Y, alphas, rho, n, log_rem, 10, T, I);

    relerr = norm(I(:)-Xhat(:)) / norm(I(:));
    psnr_val = psnr_local(Xhat, I);
    fprintf('RELERR = %.4f, PSNR = %.2f dB\n', relerr, psnr_val);

    % visual part
    f = figure('Name', sprintf('Matrix completion – keep %.0f%%', keep_ratio*100), ...
               'Color','w','Position',[100 100 1400 420]);
    tiledlayout(1,4,'Padding','compact','TileSpacing','compact');
    nexttile; imshow(I,[]); title('Original');
    nexttile; imshow(Iobs); title(sprintf('Observed (%.0f%%)', keep_ratio*100));
    nexttile; imshow(Xhat,[]);
    title(sprintf('Reconstruction\nRELERR=%.4f  PSNR=%.2f dB', relerr, psnr_val));
    errmap = mean(abs(I - Xhat),3);
    nexttile; imagesc(errmap); axis image off; colorbar; title('Error map (|I - \hat{I}|)');
    exportgraphics(f, sprintf('compare_matrix_IST_%d.png', round(keep_ratio*100)));
    imwrite(Xhat, sprintf('recon_matrix_IST_%d.png', round(keep_ratio*100)));

% end

function [image, M, N] = load_image()
    image = imread("image1.tiff", "tiff");
    % image = image(1:30, 1:30, :);
    sz = size(image);
    M = sz(1);
    N = sz(2);
end

function [omega, omega_overline, removed_idx, kept_idx, lin_rem_idx, log_rem] = create_masks(keep_ratio)
    omega = [];
    omega_overline = [];
    removed_idx = [];
    kept_idx = [];
    image_size = 512;
    pixel_count = image_size*image_size;
    no_removed = round(pixel_count*(1-keep_ratio));
    removed_flattened = randperm(pixel_count, no_removed);  % idx to remove from the flattened array
    removed_idx = zeros(no_removed, 2,  "int32");
    [removed_idx(:, 1), removed_idx(:, 2)] = ind2sub([image_size, image_size], removed_flattened);
    omega = ones(image_size);
    omega(removed_idx(:, 1), removed_idx(:, 2)) = 0;
    omega_overline = 1-omega;
    [rk, ck] = find(omega(:,:) == 1);
    kept_idx = int64([rk ck]);
    lin_rem_idx = [removed_flattened, removed_flattened, removed_flattened];
    log_rem = ones(image_size);
    log_rem(removed_flattened) = 0;
    log_rem = (log_rem < 0.5);
    log_rem =cat(3, log_rem, log_rem, log_rem);
end
    
function [X, errors] = silrtc(M, X, alphas, betas, n, log_rem, K, T, image)
    % idx removed are the pixels removed from the image
    errors = ones(K, 4);
    for j = 1:K
        disp("beginning iteration");
        selected = zeros(length(log_rem),1)';  % for collecting the average of all versions of the optimizaiton variables
        for i = 1:n
            disp(["beginning update of block matrix", num2str(n)]);
            drawnow('update');
            M{i} = shrinkage(mode_n_matricization(X, i), alphas(i)/betas(i));
            folded = mode_n_folding(M{i}, i, size(X));
            selected = selected + folded(log_rem) * betas(i);

        end

        X(log_rem) = selected / sum(betas);
        for asdf = 1:3
            errors(j, asdf) = frob_norm(double(T{asdf})-M{asdf});
        end
        errors(j, 4) = frob_norm(double(image) - X);
    end
end

function [X, errors] = halrtc(M, X, Y, alphas, rho, n, log_rem, K, T, image)
    X_things = zeros(size(X));  % for collecting the average of all versions of the optimizaiton variables
    errors = ones(K, 4);

    for j = 1:K
        X_things = zeros(size(X));
        for i = 1:n
            M{i} = mode_n_folding(shrinkage(mode_n_matricization(X, i) + mode_n_matricization(Y{i}, i) / rho, alphas(i) / rho), i, size(X));
            X_things = X_things + M{i} - Y{i} / rho;
        end
        X(log_rem) = X_things(log_rem) / n;
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

%% PSNR for images in [0,1]
function val = psnr_local(Ahat, A)
    mse = mean((Ahat(:) - A(:)).^2);
    if mse == 0
        val = Inf;
    else
        val = 10*log10(1.0/mse);
    end
end
