% Stroke Extraction using GMM algorithm
% (c) 2025 kamil.stachurski@uj.edu.pl

clearvars;
close all;
fclose all;

jetgray = jet(256);
jetgray(1, :) = [0.3, 0.3, 0.3];



% Import data
data_path = '/Users/kamil/Documents/my_softwares/gmm_stroke/data/';
scan_name = 'rat_model_1';
data = {'6'};
data_INV = {'7'};
diff_clim = [0, 0.001];
brain_mask = load('/Users/kamil/Documents/my_softwares/gmm_stroke/data/brain_mask.mat').brain_mask;




% Read data 
[acqp, headers] = readBrukerParamFile([data_path, scan_name, '/', data{1}, '/acqp']);
method = readBrukerParamFile([data_path, scan_name, '/', data{1}, '/method']);
visu = readBrukerParamFile([data_path, scan_name, '/', data{1}, '/pdata/1/visu_pars']);
img = squeeze(readBruker2dseq([data_path, scan_name, '/', data{1}, '/pdata/1/2dseq'], visu));
img = rot90(permute(img, [2, 1, 3, 4]), 2); % [x, y, slice, diff]



% Extract brain area
diffImg = zeros(size(img));
for icnt = 1:size(img, 3)
    for jcnt = 1:size(img, 4)
        stackImg = img(:, :, icnt, jcnt) .* brain_mask(:, :, icnt);
        diffImg(:, :, icnt, jcnt) = stackImg;
    end
end


% Display img and img INV
nSlices = size(diffImg, 3);
validSlices = [];

for i = 1:nSlices
    if any(diffImg(:, :, i, 1), 'all')
        validSlices(end + 1) = i;
    end
end

nValid = numel(validSlices);
nCols = ceil(sqrt(nValid));
nRows = ceil(nValid / nCols);

fig0 = figure();
for idx = 1:nValid
    s = validSlices(idx);
    subplot(nRows, nCols, idx);
    imagesc(diffImg(:, :, s, 1));
    axis image off;
    colormap gray;
    title(['Slice ', num2str(s)]);
end
sgtitle('Brain Images');



% Create map
bvals = squeeze(sort(method.PVM_DwEffBval));
ADC_map = compute_ADC(diffImg, bvals, brain_mask);
%{
fig1 = figure();
imagesc(ADC_map(:, :, 21));
axis off image;
clim(diff_clim);
colormap(jetgray);
cb = colorbar;
ylabel(cb, '[mm^2/s]');
title(sprintf('Diffusion Map'));
%}
fig1 = figure();
for idx = 1:nValid
    s = validSlices(idx);
    subplot(nRows, nCols, idx);
    imagesc(ADC_map(:, :, s));
    axis image off;
    clim(diff_clim);
    colormap(jetgray);
    cb = colorbar;
    ylabel(cb, '[mm^2/s]');
    title(['Slice ', num2str(s)]);
end
sgtitle('Diffusion Map');



% Diffusuin values extraction
diff_vals_non = ADC_map(:);
diff_vals = diff_vals_non(diff_vals_non >= 0.00001 & diff_vals_non <= 0.0014);




% Plot histogram
fig2 = figure();
hold on;
h = histogram(diff_vals, 'BinWidth', 0.00001, 'DisplayName', 'Brain Tissue', 'FaceColor', 'y', 'FaceAlpha', 0.5);
xlabel('Diffusion Coefficient [mm^2/s]');
ylabel('Amount of pixels');
legend('show');
grid('on');
title((sprintf('Histogram of ADC')));
hold off;




% GMM algorithm
init_mu = [0.35e-3, 0.6e-3, 0.9e-3];
init_sigma = reshape([0.05e-3, 0.1e-3, 0.1e-3].^2, [1, 1, 3]);
init_w = [0.2, 0.7, 0.1];
start = struct();
start.mu = init_mu';
start.Sigma = init_sigma;
start.ComponentProportion = init_w;

options = statset('MaxIter', 500);
gmm = fitgmdist(diff_vals, 3, 'Options', options, 'Start', start);
x_vals = linspace(min(diff_vals), max(diff_vals), 100);

pdf_1 = gmm.ComponentProportion(1) * normpdf(x_vals, gmm.mu(1), sqrt(gmm.Sigma(1)));
pdf_2 = gmm.ComponentProportion(2) * normpdf(x_vals, gmm.mu(2), sqrt(gmm.Sigma(2)));
pdf_3 = gmm.ComponentProportion(3) * normpdf(x_vals, gmm.mu(3), sqrt(gmm.Sigma(3)));
pdf = pdf_1 + pdf_2 + pdf_3;

scale_factor = numel(diff_vals) * h.BinWidth;

hold on;
plot(x_vals, pdf_1 * scale_factor, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Ischemia Fit');
hold on;
plot(x_vals, pdf_2 * scale_factor, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Healthy Fit');
hold on;
plot(x_vals, pdf_3 * scale_factor, 'b--', 'LineWidth', 1.5, 'DisplayName', 'CBF Fit');
hold off;





% GMM extraction
probs = posterior(gmm, ADC_map(:));
first_probs = reshape(probs(:, 1), size(ADC_map));
second_probs = reshape(probs(:, 2), size(ADC_map));
third_probs = reshape(probs(:, 3), size(ADC_map));

[~, ischemic_class] = min(gmm.mu); % klasa z najniższym ADC
[~, maxClass] = max(probs, [], 2); % klasa o najwyższym prawdopodobieństwie

ischemia_mask = (maxClass == ischemic_class);
ischemia_mask = reshape(ischemia_mask, size(ADC_map));

confidence = 0.7; % 70% pewności
ischemia_mask = (maxClass == ischemic_class) & (probs(:, ischemic_class) > confidence);
ischemia_mask = reshape(ischemia_mask, size(ADC_map));

% Wczytanie danych
first_b_img = squeeze(diffImg(:,:,:,1));
first_b_img = double(first_b_img);
first_b_img = first_b_img ./ max(first_b_img(:));  % normalizacja [0,1]

mask = double(ischemia_mask);

% Przygotowanie RGB volume
rgbVol = zeros([size(first_b_img), 3]);

% Skala szarości = taki sam obraz w R, G, B
rgbVol(:,:,:,1) = first_b_img;   % red
rgbVol(:,:,:,2) = first_b_img;   % green
rgbVol(:,:,:,3) = first_b_img;   % blue

% Dodanie maski tylko do kanału czerwonego
rgbVol(:,:,:,1) = min(rgbVol(:,:,:,1) + 1.5*mask, 1); 

% Wizualizacja 3D
figure;
volshow(rgbVol);
title('3D GMM Visualization (Gray image + Red mask)');



