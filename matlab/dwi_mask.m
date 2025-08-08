function dwi_mask(img)

    brain_mask = zeros(size(img, 1), size(img, 2), size(img, 3));
    
    figure();
    for icnt = 1:size(img, 3)
        imagesc(img(:, :, icnt, 1));
        axis off image;
        colormap('gray');
        title(['Slice ' num2str(icnt) '/' num2str(size(img, 3))]);

        choice = questdlg('Draw Brain ROI...', 'ROI Selection', 'Draw ROI', 'Continue', 'Exit', 'Continue');

        switch choice
            case 'Draw ROI'
                roi = drawfreehand('Color', 'g');
                wait(roi);
                mask = createMask(roi);
                brain_mask(:, :, icnt) = mask;
            case 'Continue'
                continue
            case 'Exit'
                return
        end

    end
    save('brain_mask.mat', 'brain_mask');
end