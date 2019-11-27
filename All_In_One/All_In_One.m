
%% 1 - Contrast unification w.r.t a reference image 
% _________________________________
disp ('Please enter the directory of Your Panaroma Parts');
%uigetdir 
path = 'J:\Users\Medghalchi, Setareh\von Carl\Images\20\*.tif';
listing = dir (path);
pathsave  = [path(1:end-5) 'All_in_One\'];
mkdir(pathsave)

% Define the image with reference color contrast 
disp ('Please select the reference image contract');
ref = uigetfile('.tif') ;

% Create folder to save
disp ('Contrast Unification ...')
for j = 1 :  size (listing ,1)
    clear J
    im = imread([path(1:end-5) listing(j).name]);
    J = imhistmatch(im,ref);
    baseFileName = sprintf('SE_00%d.png',j); % save imgs as .png since at the momet we work with .pngs
    fullFileName = fullfile(pathsave, baseFileName);
    imwrite(J, [pathsave, baseFileName]);
end 


%% 2- Apply DBSCAN 
%_____________________________________________________
path_unified = 'J:\Users\Medghalchi, Setareh\von Carl\Images\20\All_in_One\*.png'
listing_unified = dir (path_unified);
pathsave_Detected =  [path_unified(1:end-5) 'Detected\'];
mkdir(pathsave_Detected)
for k = 1 : size (listing_unified , 1)
  clear img  
  img =  imread([path_unified(1:end-5) listing_unified(k).name]);
   DBSCAN_Double(img,pathsave_Detected,k) 
   disp ('Detecting the sites ...')
end 

%% 3 - resize and adjust the data proper for the IncpV3 (?)
%________________________________________________________________
path = 'I:\Daten\Data\Neuer Ordner\*.png';
listing = dir (path);
pathsave  = [path(1:end-5) 'Resized_IncpV3\'];
mkdir(pathsave)

for i = 1 :  size (listing ,1)
    clearvars J im
    im = imread([path(1:end-5) listing(i).name]);
    
    % to rescale - to convert the size of image to the input size of the network
    scale =  299 / (size(im,1));  % Input size of InceptionV3
    im = imresize(im,scale);
    
     % convert 2 dim images -> 3 dim. (299 x 299 uint8 -> 299 x 299x3 uint8)
     if size (im,3) == 1 
     im_new = repmat(im, 1, 1, 3); 
     else 
     im_new = im;
     end
    
    imwrite( im_new, [pathsave listing(i).name]);
    i
end 

%% 4 - Import the trained network
% ____________________________________________________
% load the pretrained network 
% Define the folder with the  cropped boxes to be classified.  

% Change th edirectory to where you have stored your images to be
% classified. 
disp ('Please Select the Trained Network')
uiopen;
disp ('Enter the directory with images to be classified')
cd 'I:\Daten\Data\Neuer Ordner'
imds_Test = imageDatastore('Resized_IncpV3', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[YPred_test,probs_test] = classify(net,imds_Test);

% Display some results:
cd 'I:\Daten\Data\Neuer Ordner\Resized_IncpV3'; mkdir ('Results_1\');
cd 'I:\Daten\Data\Neuer Ordner\Resized_IncpV3\Results_1'

idx = randperm(numel(imds_Test.Files),numel(imds_Test.Files));

for i = 1: length(idx)
    clearvars I label title
    I = readimage(imds_Test,idx(1,i));
    %imshow(I)
    label = YPred_test(idx(i));
    title = title(char(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
    imwrite (I, sprintf('%d_%s.png',i,get(get(gca,'title'),'string')))
   i 
end

