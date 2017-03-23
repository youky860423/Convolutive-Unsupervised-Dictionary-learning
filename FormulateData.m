clear;
close all;
%%%%%%%%%%read data files%%%%%
% %pathname='C:\Birds data\spectrogram_white';
% % pathname='C:\Birds data\';
% % addpath(pathname);
% % fmt='bmp';
% % foldername=[pathname,'\*.',fmt];
% % files=dir(foldername);
load('images.mat');
N=3;
F=257;
T=2497;
tmp1=image1';tmp2=image2';tmp3=image3';
Y(:,1)=tmp1(:);Y(:,2)=tmp2(:);Y(:,3)=tmp3(:);
Yfreq=zeros(F,N*T);
for f=1:F
    Yfreq(f,1:T)=image1(f,:);
    Yfreq(f,T+1:2*T)=image2(f,:);
    Yfreq(f,2*T+1:3*T)=image3(f,:);
end
% threshold=40;
% spectro_true=1;
% loc='pc1';
% % for i=1:N
% %     filename=files(i).name;
% %     img=imread(filename,fmt); 
% %     img=rgb2gray(img);
% % %     img=double(img)/threshold-1;
% % % %     figure(1)
% % % %     imshow(img);
% % % %     title(filename);
% % % %     pause()
% %     tmp = double(img)';
% %     Y(:,i)= tmp(:);
% %     for f = 1:F    
% %          Yfreq(f,(i-1)*T+1:i*T)=double(img(f,:));
% %     end   
% % end

%%%%%%Dictionary Learning%%%%
K=3;
W=300;
Ainirdm=randn(T+W-1,K,N);
Dinirdm=randn(W,F,K);
%true activations;
Aini=zeros(T+W-1,K,N);
actBox1 = [100 500 1000]+W-1;
actTri1 = [300 1500 2000]+W-1;
actBox2 = [300 800]+W-1;
actRect2 = [500 1000 1500 1800]+W-1;
actBox3 = [200 400 600 2000]+W-1;
actTri3 = [300 900 1200]+W-1;
actRect3 = [700  1500]+W-1;
Aini(actBox1,1,1)=120;%120
Aini(actTri1,2,1)=120;
Aini(actBox2,1,2)=255;%255
Aini(actRect2,3,2)=255;
Aini(actBox3,1,3)=200;%200
Aini(actTri3,2,3)=200;
Aini(actRect3,3,3)=200;
%true dictionary words;
Dini=zeros(W,F,K); 
xBox = [1 49 49 1];
yBox = [50 50 149 149];
xTri = [1 99 50];
yTri = [20 20 100];
xRect = [1 150 150];
yRect = [200 200 100];
Dini(:,:,1)=poly2mask(xBox,yBox,F,W)';
Dini(:,:,2)=poly2mask(xTri,yTri,F,W)';
Dini(:,:,3)=poly2mask(xRect,yRect,F,W)';
%%%%%%%main function : dictionary learning%%%%%
outiter=10;
D=Dini;
A=Ainirdm;
eps = 1e-8;
maxiter=100;
lambda=100;
for i=1:outiter
% %     %%%%%%dictionary learning%%%%%
% % Dlast=zeros(size(D));
% % iter=0;
% % rfreq=zeros(N*T,F);
% % while( norm(D(:)-Dlast(:))>eps && iter < maxiter )
% %     Dlast=D;
% % % %     figure(10)
% % % %     for k=1:K
% % % %         imshow(D(:,:,k)');
% % % %         pause()
% % % %     end
% %     for f=1:F
% %         for n=1:N
% %             for k=1:K
% %                 yhat(:,n,k)=ifft(fft(A(:,k,n),T+W-1).*fft(D(:,f,k),T+W-1));
% % %                   yhat(:,n,k)=conv(A(:,k,n),D(:,f,k),'valid');
% % % %                   Ta((n-1)*T+1:n*T,(k-1)*W+1:k*W)=toeplitz(A(W:end,k,n),flipud(A(1:W,k,n)));
% %             end
% % %             sumkyhat(:,n)=sum(yhat(:,n,:),3);
% %             sumkyhat(:,n)=sum(yhat(W:end,n,:),3);
% %             idx=(n-1)*T+1:n*T;
% %             rfreq(idx,f)=Yfreq(f,idx)'-sumkyhat(:,n);
% % 
% % % %             if sum(ismember([50,80,150],f))
% % % %                 figure(7);
% % % %                 plot(Yfreq(f,idx)');
% % % %                 figure(8);
% % % %                 plot(sumkyhat(:,n),'r')
% % % %                 figure(9);
% % % %                 plot(rfreq(idx,f),'g')
% % % %                 title(['f=',num2str(f),'and n=',num2str(n)])
% % % %                 pause()
% % % %             end
% %             for k=1:K
% %                 Vdn(:,n,k)=ifft(fft(flipud(A(:,k,n)),T+W-1).*fft(rfreq(idx,f),T+W-1));
% % %                 Vdn(:,n,k)=conv(flipud(A(:,k,n)),rfreq(idx,f),'valid');
% %             end
% %         end
% % %         Vd=sum(Vdn,2);
% %         Vd=sum(Vdn(T:end,:,:),2);
% %         for n=1:N
% %             for k=1:K
% %                 temp(:,n,k)=ifft(fft(A(:,k,n),T+W-1).*fft(Vd(:,1,k),T+W-1));
% % %                 temp(:,n,k)=conv(A(:,k,n),Vd(:,1,k),'valid');
% %             end
% %         end
% % % %         %%%%test value of Vd by using toeplitz matrix%%%
% % % %         Dfreq=D(:,f,:);
% % % %         Dfreq=Dfreq(:);
% % % %         V=Ta'*(Yfreq(f,:)'-Ta*Dfreq);
% % % %         norm(V-Vd(:))
% % % %         %%%%test gradient%%%%
% % % %         x1=squeeze(D(:,f,:));
% % % %         x2=x1;
% % % %         x2(100,1)=x2(100,1)+1e-10;
% % % %         y1 = testfunc( x1,A,Yfreq(f,:));
% % % %         y2 = testfunc( x2,A,Yfreq(f,:));
% % % %         grad=(y2-y1)/1e-10;
% % % %         yy1=norm(Yfreq(f,:)'-Ta*x1(:))^2;
% % % %         yy2=norm(Yfreq(f,:)'-Ta*x2(:))^2;
% % % %         grad2=(yy2-yy1)/1e-10;
% % % %         -2*V(100)-grad2
% % % %         -2*Vd(100,1,1)-grad
% % % %         pause()
% % 
% % %         temp2=sum(temp,3);
% %         temp2=sum(temp(W:end,:,:),3);
% %         if norm(temp2(:)) ~= 0
% %             c=(norm(Vd(:))^2)/(norm(temp2(:))^2);
% %         else
% %             c=0;
% %         end
% % % %         figure(1)
% % % %         plot(Ta*V)
% % % %         hold on
% % % %         plot(temp2(:),'r-.')
% % % %         hold off
% % % %         if norm(Ta*V)~=0
% % % %             c2=norm(V)^2/norm(Ta*V)^2;
% % % %         else
% % % %             c2=0;
% % % %         end
% % % %         c-c2
% % % %         pause()
% % % %         a=zeros(size(Vd));
% % % %         a(:,1,:)=reshape(V,W,[]);
% % % %         D(:,f,:) = D(:,f,:) + c2*a;
% %         D(:,f,:) = D(:,f,:) + c*Vd;
% %     end
% %     for k=1:K
% %         Words=D(:,:,k);
% %         if norm(Words,'fro') <= 1
% %             Words=Words/norm(Words,'fro');
% %         end
% %         D(:,:,k)=Words;
% %     end
% %     iter=iter+1;
% %     obj1(iter)=sum(sum(rfreq.^2));
% % end
% %  figure(1)
% %  subplot(1,K,1);imshow(D(:,:,1)');
% %  subplot(1,K,2);imshow(D(:,:,2)');
% %  subplot(1,K,K);imshow(D(:,:,K)');
% %  figure(2)
% %  imshow(rfreq);
% % figure(3)
% % plot(obj1);
% % % % pause()
%%%%%%%%Activation Extraction%%%%%
Alast=zeros(size(A));
iter2=0;
rspect=zeros(F*T,N);
while(norm(A(:)-Alast(:))>eps && iter2<maxiter )
    Alast = A;
    for n=1:N
        for f=1:F
            for k=1:K
                yhat2(:,f,k)=ifft(fft(A(:,k,n),T+W-1).*fft(D(:,f,k),T+W-1));
%                   yhat2(:,f,k)=conv(A(:,k,n),D(:,f,k),'valid');
% %                   Td((f-1)*T+1:f*T,(k-1)*(T+W-1)+1:k*(T+W-1))=toeplitz([D(end,f,k);zeros(T-1,1)],[flipud(D(:,f,k));zeros(T-1,1)]);
% %                   yyhat=Td((f-1)*T+1:f*T,(k-1)*(T+W-1)+1:k*(T+W-1))*A(:,k,n);
% %                   norm(yhat2(:,f,k)-yyhat)
% %                   figure(1)
% %                   plot(yyhat)
% %                   hold on
% %                   plot(yhat2(:,f,k),'r-.')
% %                   hold off
% %                   title(['f=',num2str(f),' and n=',num2str(n),' and k=', num2str(k)])
% %                   pause()
            end
%                   if sum(ismember([50,80,150],f))
%                       for l=1:K
%                         figure(10)
%                         plot(A(:,l,n))
%                         figure(11)
%                         plot(D(:,f,l),'b-.')
%                         figure(12)
%                         plot(yhat2(:,f,l),'r')
%                         title(['f=',num2str(f),' and n=',num2str(n), ' and k=',num2str(l)])
%                         pause()
%                       end
%                       
%                   end
            sumkyhat2(:,f)=sum(yhat2(W:end,f,:),3);
%             sumkyhat2(:,f)=sum(yhat2(:,f,:),3);
            idx2=(f-1)*T+1:f*T;
            rspect(idx2,n)=Y(idx2,n)-sumkyhat2(:,f);
%             if sum(ismember([50,80,150],f))
%                 figure(7);
%                 plot(Y(idx2,n));
%                 figure(8);
%                 plot(sumkyhat2(:,f),'r')
%                 figure(9);
%                 plot(rspect(idx2,n),'g')
%                 title(['f=',num2str(f),'and n=',num2str(n), ' and k=',num2str(k)])
%                 pause()
%             end
            for k=1:K
                Vaf(:,f,k)=ifft(fft(rspect(idx2,n),T+W-1).*fft(flipud(D(:,f,k)),T+W-1));
%                 Vaf(:,f,k)=conv(rspect(idx2,n),flipud(D(:,f,k)));
            end
        end
        Va=squeeze(sum(Vaf,2));
% %         %%%%test value of Va by using toeplitz matrix%%%
% %         Aspect=A(:,:,n);
% %         V2=Td'*(find());
% %         norm(V2-Va(:))
% %         pause()
        for f=1:F
            for k=1:K
                temp3(:,f,k)=ifft(fft(Va(:,k),T+W-1).*fft(D(:,f,k),T+W-1));
%                 temp3(:,f,k)=conv(Va(:,k),D(:,f,k),'valid');
            end
        end
%         temp4=sum(temp3,3);
        temp4=sum(temp3(W:end,:,:),3);
        if norm(temp4(:)) ~= 0
            c=norm(Va(:))^2/norm(temp4(:))^2;
        else
            c=0;
        end
% %         A(:,:,n) = A(:,:,n) + c*Va;
        idxGTZero = (A(:,:,n) + c*(Va-lambda)) > zeros(T+W-1,K);
        idxLTZero = (A(:,:,n) + c*(Va+lambda)) < zeros(T+W-1,K);
        Aprim=A(:,:,n);
        Aprim(idxGTZero) = Aprim(idxGTZero) + c*(Va(idxGTZero)-lambda);
        Aprim(idxLTZero) = Aprim(idxLTZero) + c*(Va(idxLTZero)+lambda);
        Aprim(~(idxGTZero | idxLTZero))=0;
        A(:,:,n)=Aprim;
    end
    iter2=iter2+1;
    obj2(iter2)=sum(sum(rspect.^2));
end
 figure(4)
 for k=1:K
     for n=1:N
        subplot(N,K,(n-1)*K+k); plot(A(:,k,n));
     end
 end
 figure(5)
 imshow(rspect);
figure(6)
plot(obj2);
pause()
end