function ADC_map = compute_ADC(img, bvals, mask)
    % compute_ADC - calculates the ADC diffusion map for each image voxel

    % img - 4D DWI image [X, Y, Slice, b]
    % bvals = vector of b-values in s/mm^2
    % mask - 3D brain mask [X, Y, Slice]

    % (c) 2025 Kamil Stachurski

    [X, Y, Z, nB] = size(img);
    ADC_map = zeros(X, Y, Z);

    for z = 1:Z
        for x = 1:X
            for y = 1:Y
                if mask(x, y, z)
                    S = squeeze(img(x, y, z, :));
                    if all(S > 0)
                        lnS = log(S(:));
                        p = polyfit(bvals(:), lnS, 1);
                        ADC_map(x, y, z) = -p(1);
                    else
                        ADC_map(x, y, z) = 0;
                    end
                end
            end
        end
    end
end