%% Obtains the relative angle between the camera in the RF source frame and the grid from a series from a given picture

clear all
close all
clc

folder = 'testvid';

folderpath='C:/Users/matro/Documents/AIUC/Scripts/GridAngle/frames/';
folderpath = strcat(folderpath, folder, '/');

% RXO-II cam parameters
width = 4800;
height = 3200;
cx = width/2;    
cy = height/2;
camera_matrix = [3.426e3, 0, 0; 0, 3.41930761e3, 0 ; cx, cy, 1];
radial_distortion= [-0.02160264,0.13696009,0.0019371];
tangential_distortion= [0.00262076,-0.23854474];

camparams= cameraParameters('IntrinsicMatrix',camera_matrix);
camparams= cameraParameters('RadialDistortion',radial_distortion);


imagefiles = dir([folderpath,'/*.jpg']);      
addpath(folderpath);

nfiles = length(imagefiles);    % Number of files found
images_list = [""];

for ii=1:nfiles
    currentfilename=imagefiles(ii).name;
    I=rgb2gray(imread(currentfilename));

[m,n]=size(I);

if n > m
    I=imrotate(I,90);
end

BW=imbinarize(I,9/255);
figure(1),subplot(1,2,1),imagesc(BW),title('Original Image (Binarized)')

se = strel('disk',2);
BW = imerode(BW,se);

figure(1),subplot(1,2,1),imagesc(BW),title('Original Image (Binarized)')

windowSize = 8;
kernel = ones(windowSize) / windowSize ^ 2;
blurryImage = conv2(single(BW), kernel, 'same');
BW=blurryImage > 0.15;

figure(1),subplot(1,2,1),imagesc(BW),title('Original Image (Binarized)')

BW_cntrd=imbinarize(I,0.995);
% BW_cntrd(1650:1750, 550:700) = 0; % C0017
BW_cntrd(1560:1860, 350:750) = 0;

figure(1),subplot(1,2,1),imagesc(BW),title('Original Image')
subplot(1,2,2),imagesc(BW_cntrd),title('Zero-order mode location')

stats0 = regionprops(BW_cntrd);
[sortedareas, areaidx] = sort([stats0.Area],'descend');
stats0_sorted= stats0(areaidx);
cntrd=stats0_sorted.Centroid;
cntrd=circshift(cntrd,1); % -> row column
figure(1),subplot(1,2,2),imagesc(BW_cntrd),title('Zero-order mode location')
yline(cntrd(1),':k');xline(cntrd(2),':k');

BW(:,1:ceil(cntrd(2))-60)=0;
BW(:,ceil(cntrd(2))+60:end)=0;

figure(1),subplot(1,2,1),imagesc(BW),title('Original Image (Binarized)')



%figure,imagesc(erodedI)

% B0 DSC06620, bin=0.25, window=12, blurbin=0.15, offset=300,
% cntrdsearch 50, 50, radius=900, [2449 1175]
% 
% stats = regionprops(BW);
% props=cell2mat(struct2cell(stats)');
% 
% [~,idx]=max(props(:,1));
% cntrd(1) = props(idx,3);
% cntrd(2) = props(idx,2);
% % 
% cntrd=[2449 1175]; % batch 0
% 
% cntrd=[2450 1260]; % batch 1
% 
% cntrd=[2460 1452]; % batch 2


cntrd_offset=300;
search_radius=1000;

% % Linear fit
% % [x,y,Iout]=obtainpoints(cntrd(1),BW,cntrd_offset,search_radius);
% % y=polyval(polyfit(x,y,1),x);
% % figure,plot(x,y);
% % linangle=atand((y(1)-y(end))/(x(end)-x(1)));
% % display(['Linear fit angle is: ',num2str(round(linangle,3)),'° '])

cntrd_search_vert_m=0;
cntrd_search_vert_p=0;
cntrd_search_horz_m=0;
cntrd_search_horz_p=0;
verbose=0;


skip=0;
skipstart=550;
skiplength=100;
fun=@(cntrd)calculate_gridangle(cntrd,BW,cntrd_offset,search_radius,verbose,skip,skipstart,skiplength);

%options = optimset('PlotFcns',@optimplotfval,'MaxIter',10);
options = optimset('MaxIter',10);

[c,fval]=fminsearchbnd(fun,[cntrd(1) cntrd(2)],[cntrd(1)-cntrd_search_vert_m cntrd(2)-cntrd_search_horz_m],[cntrd(1)+cntrd_search_vert_p cntrd(2)+cntrd_search_horz_p],options);

figure(1);yline(c(1),':k');xline(c(2),':k');

showSTD=0;
[error, angle]=calculate_gridangle([c(1) c(2)],BW,cntrd_offset,search_radius,showSTD,skip,skipstart,skiplength);

display([imagefiles(ii).name, ' relative angle is: ',num2str(round(angle,3)),'° ', char(177) ,num2str(round(error,3)),'°'])
images_list = [images_list, strcat(imagefiles(ii).name, ' relative angle is: ', num2str(round(angle,3)),' deg +/- ', num2str(round(error,3)),' deg')];

x(1) = c(1);
y(1) = c(2);
x(2) = x(1) + -(cntrd_offset + search_radius) * cosd(-angle);
y(2) = y(1) + -(cntrd_offset + search_radius) * sind(-angle);

figure(1),subplot(1,2,1),hold on, plot(y,x,':r','LineWidth',2)

%lineLength = 1000;
x(1) = c(1);
y(1) = c(2);
x(2) = x(1) + (cntrd_offset + search_radius) * cosd(-angle);
y(2) = y(1) + (cntrd_offset + search_radius) * sind(-angle);

figure(1),subplot(1,2,1),hold on, plot(y,x,':r','LineWidth',2)

ylim([c(1)-cntrd_offset - search_radius c(1)+cntrd_offset + search_radius])
xlim([c(2)-cntrd_offset - search_radius c(2)+cntrd_offset + search_radius])
hold off
end

fname = strcat('../results/', folder, '_output_matlab.txt');
fileID = fopen(fname, 'w');  % Open the file for writing
for i = 1:length(images_list)-1
    fprintf(fileID, '%s\r\n', images_list(i+1));  % Write each string to the file 
end
fclose(fileID);  % Close the file