clear;
close all;

load('imagesSmall.mat');
% % %%%%%adding noise to the image%%%%
% % nvar=1;
% % nmean=0;
% % image1 = image1 + sqrt(nvar)*randn(size(image1)) + nmean;
% % image2 = image2 + sqrt(nvar)*randn(size(image2)) + nmean;
% % image3 = image3 + sqrt(nvar)*randn(size(image3)) + nmean;
%%%%%%%%%%
N=3;
for i=1:N
    tmp{i}=image{i}';
    Y(:,i)=tmp{i}(:);
end
F=size(image{1},1);
T=size(image{1},2);
Yfreq=zeros(F,N*T);
for f=1:F
    for i=1:N
    Yfreq(f,(i-1)*T+1:i*T)=image{i}(f,:);
    end
end
W=50;
K=3;
Ainirdm=randn(T+W-1,K,N);
Ainirdm=Ainirdm-min(Ainirdm(:));
Dinirdm=randn(W,F,K);
% % %initial with true activations;
% % Aini=zeros(T+W-1,K,N);
% % actBox1 = [100 500 1000]+W-1;
% % actTri1 = [300 1500 2000]+W-1;
% % actBox2 = [300 800]+W-1;
% % actRect2 = [500 1000 1500 1800]+W-1;
% % actBox3 = [200 400 600 2000]+W-1;
% % actTri3 = [300 900 1200]+W-1;
% % actRect3 = [700  1500]+W-1;
% % Aini(actBox1,1,1)=120;%120
% % Aini(actTri1,2,1)=120;
% % Aini(actBox2,1,2)=255;%255
% % Aini(actRect2,3,2)=255;
% % Aini(actBox3,1,3)=200;%200
% % Aini(actTri3,2,3)=200;
% % Aini(actRect3,3,3)=200;
% % %initial with true dictionary words;
% % Dini=zeros(W,F,K); 
% % xBox = [1 49 49 1];
% % yBox = [50 50 149 149];
% % xTri = [1 99 50];
% % yTri = [20 20 100];
% % xRect = [1 150 150];
% % yRect = [200 200 100];
% % Dini(:,:,1)=poly2mask(xBox,yBox,F,W)';
% % Dini(:,:,2)=poly2mask(xTri,yTri,F,W)';
% % Dini(:,:,3)=poly2mask(xRect,yRect,F,W)';
%%%%%%%main function : dictionary learning and setting parameters%%%%%
parameter.display=1;
parameter.displayNumPerIter=100;
parameter.outMaxIter=5000;
parameter.innerMaxIter=2;
parameter.eps=1e-8;
parameter.lambda=1;
Data.Yfreq=Yfreq;
Data.Yspect=Y;
parameter.dlfirst=1;
parameter.aefirst=0;
tic
[ D,A,objtot ] = DictionaryLearn( Dinirdm,Ainirdm,Data,parameter );
tnorm=toc
%%%%plotting the original dictionary and activation%%%%%%%
N=size(A,3);K=size(D,3);
W=size(D,1);F=size(D,2);
T=size(A,1)+1-W;
figure(1)
for k=1:size(D,3)
    subplot(1,size(D,3),k);imagesc(D(:,:,k)',[0 max(max(D(:,:,k)))]);colormap gray
end
title('learned dictionary words')
colortype=['b','r','g'];
for n=1:N
    figure(2)
    for k=1:K
       subplot(N,1,n); plot(A(:,k,n),colortype(k));
       hold on
    end
    hold off
end
title('learned activation signals')
for n=1:N
        for f=1:F
            for k=1:K
                Yrecon(:,f,k)=ifft(fft(A(:,k,n),T+W-1).*fft(D(:,f,k),T+W-1));
            end
            YreconSpectro(:,f)=sum(Yrecon(W:end,f,:),3);
        end
        figure(3)
        subplot(N,1,n);imagesc(YreconSpectro',[0 max(max(YreconSpectro))]);colormap gray
        title(['reconstructed spectrogram ',num2str(n)])
        figure(4)
        subplot(N,1,n);imagesc(image{n},[0 max(max(image{n}))]);colormap gray
        title(['true spectrogram ',num2str(n)])
end
figure(5)
semilogy(objtot)
grid on
title('objective value vs. iteration with uncompressed data')
pause()

%%%%%%%%%%%%%random projection%%%%%%%%%
nump=0;
for percent=0.2:0.2:0.8
    reducedDim=ceil(percent*F);
    q=randn(reducedDim,F);
    compressedImg1=(q*image{1})';
    compressedImg2=(q*image{2})';
    compressedImg3=(q*image{3})';
    Y_compr=[];
    Y_compr(:,1)=compressedImg1(:);Y_compr(:,2)=compressedImg2(:);Y_compr(:,3)=compressedImg3(:);
    Yfreq_compr=zeros(reducedDim,N*T);
    for f=1:reducedDim
        Yfreq_compr(f,1:T)=compressedImg1(:,f)';
        Yfreq_compr(f,T+1:2*T)=compressedImg2(:,f)';
        Yfreq_compr(f,2*T+1:3*T)=compressedImg3(:,f)';
    end
    Ainirdm_compr=randn(T+W-1,K,N);
    Ainirdm_compr=Ainirdm_compr-min(Ainirdm_compr(:));
    Dinirdm_compr=randn(W,reducedDim,K);
    Data_compr.Yfreq=Yfreq_compr;
    Data_compr.Yspect=Y_compr;
    parameter.outMaxIter=5000;
    tic
    [ D_compr,A_compr,objtot_compr ] = DictionaryLearn( Dinirdm_compr,Ainirdm_compr,Data_compr,parameter );
    tcompr=toc
    % % %%%%%%checking the recovered dictionary%%%%%
    DhatK=zeros(W,F,K);
    for k=1:K
        Dhat=zeros(W,F);
        for iter=1:1e4
           Dhat=Dhat+1e-5*((D_compr(:,:,k)-Dhat*q')*q-100*sign(Dhat));
        end
        DhatK(:,:,k)=Dhat;
    end
% %     %%%%%%Recover the dictionary words using cvx%%%%%
% %     Drecovered=zeros(W,F,K);
% %     lambda=10;
% %     for k=1:K
% %         for w=1:W
% %             signal_compr=D_compr(w,:,k)';
% %             cvx_begin
% %                 variable x(F)
% %                 minimize(sum_square_abs(q*x-signal_compr)+lambda*norm(x,1))
% %             cvx_end
% %             Drecovered(w,:,k)=x';
% %         end
% %             figure(3)
% %             subplot(1,K,k);imagesc(Drecovered(:,:,k)'); colormap gray;
% %             title('recovered dictionary words using cvx');
% %     end
    %%%%%checking dictionary words by applying again the dictionary learning
    %%%%%with the activation learned from the last step%%%%%
    parameter.outMaxIter=150;
    [ Drecovered,~,~ ] = DictionaryLearn( Dinirdm,A_compr,Data,parameter );

    %%%%%%%%%%plotting result%%%%%%%%%%%%
    figure(1)
    for k=1:K
         subplot(1,K,k);imagesc(D_compr(:,:,k)',[0 max(max(D_compr(:,:,k)))]);colormap gray
    end
    figure(2)
    for k=1:K
         subplot(1,K,k);imagesc(DhatK(:,:,k)',[0 max(max(DhatK(:,:,k)))]);colormap gray
    end
    figure(3)
    for k=1:K
         subplot(1,K,k);imagesc(Drecovered(:,:,k)',[0 max(max(Drecovered(:,:,k)))]);colormap gray
    end
    for n=1:N
    figure(4)
    for k=1:K
       subplot(N,1,n); plot(A_compr(:,k,n),colortype(k));
       hold on
    end
    hold off
    end
    r=size(D_compr,2);
    for n=1:N
        for f=1:r
            for k=1:K
                Yrecon_compr(:,f,k)=ifft(fft(A_compr(:,k,n),T+W-1).*fft(D_compr(:,f,k),T+W-1));
            end
            YreconSpectro_compr(:,f)=sum(Yrecon_compr(W:end,f,:),3);
        end
        figure(5)
        subplot(N,1,n);imagesc(YreconSpectro_compr',[0 max(max(YreconSpectro))]);colormap gray
        title(['reconstructed spectrogram ',num2str(n)])
        figure(6)
        subplot(N,1,n);imagesc(reshape(Data_compr.Yspect(:,n),T,[])',[0 max(Data_compr.Yspect(:,n))]);colormap gray
        title(['true spectrogram ',num2str(n)])
    end
    figure(7)
    semilogy(objtot_compr)
    grid on
    title('objective value vs. iteration with compressed data')
    pause()    

    %%%%%varying different compression coefficient%%%%%
    nump=nump+1;
    runtime(nump)=tcompr;
    Derr(nump)=norm(DhatK(:)-Drecovered(:));
end
