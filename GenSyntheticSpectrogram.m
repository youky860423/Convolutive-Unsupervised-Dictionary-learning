clear 
close all;
% % %%%%parameters%%%%%%%
% % imageSize = [257 2497];
% % xBox = [1 49 49 1];
% % yBox = [50 50 149 149];
% % xTri = [1 99 50];
% % yTri = [20 20 100];
% % xRect = [1 150 150];
% % yRect = [200 200 100];
%%%%%%%smaller size%%%%%%
stripLength=3;
%%%%parameters%%%%%%%
imageSize = [50 500];
xBox = [1 20 20 1];
yBox = [12 12 40 40];
xBoxStrip = [1+stripLength 20-stripLength 20-stripLength 1+stripLength];
yBoxStrip = [12+stripLength 12+stripLength 40-stripLength 40-stripLength];
xTri = [1 25 12];
yTri = [5 5 20];
xTriStrip = [1+stripLength 25-stripLength 12];
yTriStrip = [5+stripLength 5+stripLength 20-stripLength];
xRect = [1 40 40];
yRect = [40 40 20];
xRectStrip = [1+2*stripLength 40-stripLength 40-stripLength];
yRectStrip = [40-stripLength 40-stripLength 20+stripLength];
image1 = zeros(imageSize);
% % actBox1 = [100 500 1000];
% % actTri1 = [300 1500 2000];
actBox1 = [0 100 200];
actTri1 = [60 300 400];
for i=1:length(actBox1)
    maskBox = poly2mask(xBox+actBox1(i),yBox,imageSize(1),imageSize(2));
    maskBoxStrip = poly2mask(xBoxStrip+actBox1(i),yBoxStrip,imageSize(1),imageSize(2));
    image1(maskBox)=120;
%     image1(maskBoxStrip)=0;
end

for i=1:length(actTri1)
    maskTri = poly2mask(xTri+actTri1(i),yTri,imageSize(1),imageSize(2));
    maskTriStrip = poly2mask(xTriStrip+actTri1(i),yTriStrip,imageSize(1),imageSize(2));
    image1(maskTri)=120;
%     image1(maskTriStrip)=0;
end
image{1}=image1(:,11:end);

image2 = zeros(imageSize);
% % actBox2 = [300 800];
% % actRect2 = [500 1000 1500 1800];
actBox2 = [60 160];
actRect2 = [100 200 300 460];
for i=1:length(actBox2)
    maskBox = poly2mask(xBox+actBox2(i),yBox,imageSize(1),imageSize(2));
    maskBoxStrip = poly2mask(xBoxStrip+actBox2(i),yBoxStrip,imageSize(1),imageSize(2));
    image2(maskBox)=255;
%     image2(maskBoxStrip)=0;
end
for i=1:length(actRect2)
    maskRect = poly2mask(xRect+actRect2(i),yRect,imageSize(1),imageSize(2));
    maskRectStrip = poly2mask(xRectStrip+actRect2(i),yRectStrip,imageSize(1),imageSize(2));
    image2(maskRect)=255;
%     image2(maskRectStrip)=0;
end
image{2}=image2(:,1:end-10);

image3 = zeros(imageSize);
% % actBox3 = [200 400 600 2000];
% % actTri3 = [300 900 1200];
% % actRect3 = [700  1500];
actBox3 = [40 80 120 400];
actTri3 = [60 180 240];
actRect3 = [160  300];
for i=1:length(actBox3)
    maskBox = poly2mask(xBox+actBox3(i),yBox,imageSize(1),imageSize(2));
    maskBoxStrip = poly2mask(xBoxStrip+actBox3(i),yBoxStrip,imageSize(1),imageSize(2));
    image3(maskBox)=200;
%     image3(maskBoxStrip)=0;
end
for i=1:length(actTri3)
    maskTri = poly2mask(xTri+actTri3(i),yTri,imageSize(1),imageSize(2));
    maskTriStrip = poly2mask(xTriStrip+actTri3(i),yTriStrip,imageSize(1),imageSize(2));
    image3(maskTri)=200;
%     image3(maskTriStrip)=0;
end
for i=1:length(actRect3)
    maskRect = poly2mask(xRect+actRect3(i),yRect,imageSize(1),imageSize(2));
    maskRectStrip = poly2mask(xRectStrip+actRect3(i),yRectStrip,imageSize(1),imageSize(2));
    image3(maskRect)=200;
%     image3(maskRectStrip)=0;
end
image{3}=image3(:,1:end-10);
figure(1)
for i=1:3
    subplot(3,1,i); imagesc(image{i}); colormap gray;
end

% % %%%%%%%display dictionary words%%%%%%%
% % F=imageSize(1);K=3;
% % Dplot={};
% % Dplot{1}=poly2mask(xBox,yBox,F,15);
% % Dplot{2}=poly2mask(xTri,yTri,F,25);
% % Dplot{3}=poly2mask(xRect,yRect,F,40);
% % for k=1:K
% %     figure;
% %     imshow(Dplot{k});
% % end
% % %%%%%%%%display activation signals%%%%%%%
% % Aplot(actBox1,1,1)=120;%120
% % Aplot(actTri1,2,1)=120;
% % Aplot(actBox2,1,2)=255;%255
% % Aplot(actRect2,3,2)=255;
% % Aplot(actBox3,1,3)=200;%200
% % Aplot(actTri3,2,3)=200;
% % Aplot(actRect3,3,3)=200;
% % N=3;
% % for n=1:N
% %     figure;
% %     plot(Aplot(:,1,n),'b','linewidth',3);
% %     figure;
% %     plot(Aplot(:,2,n),'r','linewidth',3);
% %     figure;
% %     plot(Aplot(:,3,n),'g','linewidth',3);
% % end

% save('toydata_forjose.mat','image');
save('imagesSmall','image');
