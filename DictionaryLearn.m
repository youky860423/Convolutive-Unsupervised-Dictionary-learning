function [ D,A,objtot ] = DictionaryLearn( Dini,Aini,Data,parameter )
%DICTIONARYLEARN learns the dictionary words and activation together
%   input: Initial Dictionary words,inital activation, data and parameter.
%   output: learned dictionary words and activation signals.

display=parameter.display;
dispfactor=parameter.displayNumPerIter;
outiter=parameter.outMaxIter;
lambda=parameter.lambda;
maxiter=parameter.innerMaxIter;
eps = parameter.eps;
W=size(Dini,1);
F=size(Dini,2);
K=size(Dini,3);
T=size(Aini,1)+1-W;
N=size(Aini,3);
Yfreq=Data.Yfreq;
Y=Data.Yspect;
DL_flag=parameter.dlfirst;
AE_flag=parameter.aefirst;

D=Dini;
A=Aini;
color_vec=rand(K,3);
objtot=zeros(outiter,1);
for i=1:outiter
    if DL_flag
        %%%%%%dictionary learning%%%%%
        Dlast=1e5*ones(size(D));
        iter=0; 
        rfreq=zeros(N*T,F);
        while( norm(D(:)-Dlast(:)) > eps && iter < maxiter )
            Dlast=D;
            VD=zeros(size(D));
            gammaf=zeros(F,1);
            for f=1:F
                %%%%%initialize the variable to speedup%%%%%%
                yhat=zeros(T+W-1,N,K);
                sumkyhat=zeros(T,N);
                Vdn=zeros(T+W-1,N,K);
                for n=1:N
                    for k=1:K
                        yhat(:,n,k)=ifft(fft(A(:,k,n),T+W-1).*fft(D(:,f,k),T+W-1));
        %                   yhat(:,n,k)=conv(A(:,k,n),D(:,f,k),'valid');
                    end
        %             sumkyhat(:,n)=sum(yhat(:,n,:),3);
                    sumkyhat(:,n)=sum(yhat(W:end,n,:),3);
                    idx=(n-1)*T+1:n*T;
                    rfreq(idx,f)=Yfreq(f,idx)'-sumkyhat(:,n);
                    for k=1:K
                        Vdn(:,n,k)=ifft(fft(flipud(A(:,k,n)),T+W-1).*fft(rfreq(idx,f),T+W-1));
        %                 Vdn(:,n,k)=conv(flipud(A(:,k,n)),rfreq(idx,f),'valid');
                    end
                end
        %         Vd=sum(Vdn,2);
                Vd=sum(Vdn(T:end,:,:),2);
                dfreq=D(:,f,:);
                dfreq=dfreq(:);
                Vdvec=Vd(:);
                Vdnorm=Vdvec/norm(Vdvec);
                dtilde=dfreq-(dfreq'*Vdnorm)*Vdnorm;
                dnorm=dtilde/norm(dtilde);
% %                 temp=zeros(T+W-1,N,K);
                for n=1:N
                    for k=1:K
% %                         temp(:,n,k)=ifft(fft(A(:,k,n),T+W-1).*fft(Vd(:,1,k),T+W-1));
        %                 temp(:,n,k)=conv(A(:,k,n),Vd(:,1,k),'valid');
                        TaVd(:,n,k)=ifft(fft(A(:,k,n),T+W-1).*fft(Vdnorm((k-1)*W+1:k*W),T+W-1));
                        Tadt(:,n,k)=ifft(fft(A(:,k,n),T+W-1).*fft(dnorm((k-1)*W+1:k*W),T+W-1));
                    end
                end
                TaVd_mat=sum(TaVd(W:end,:,:),3);
                TaVd_vec=TaVd_mat(:);
                Tadt_mat=sum(Tadt(W:end,:,:),3);
                Tadt_vec=Tadt_mat(:);
                Xf=[TaVd_vec,Tadt_vec];
                gammaf(f,1)=max(eig(Xf'*Xf));
                VD(:,f,:)=Vd;
        %         temp2=sum(temp,3);
% %                 temp2=sum(temp(W:end,:,:),3);
% %                 if norm(temp2(:)) ~= 0
% %                     c=(norm(Vd(:))^2)/(norm(temp2(:))^2);
% %                 else
% %                     c=0;
% %                 end
% %                 D(:,f,:) = D(:,f,:) + c*Vd;
            end
            gamma=max(gammaf);
            D = D + (1/gamma)*VD;
            for k=1:K
                Words=D(:,:,k);
                if norm(Words,'fro') > 1
                    Words=Words/norm(Words,'fro');
                end
                D(:,:,k)=Words;
            end
            iter=iter+1;
            obj1_err(i,iter)=sum(sum(rfreq.^2));
            obj1_Al1(i,iter)=norm(A(:),1);
        end
        if display && rem(i,dispfactor)==0
         figure(1)
         for k=1:K
             subplot(1,K,k);imagesc(D(:,:,k)',[0 max(max(D(:,:,k)))]);colormap gray
         end
        % %  figure(2)
        % %  imshow(rfreq);
    % %     figure(3)
    % %     plot(obj1);
    % %     pause(0.1)
        end
        AE_flag=true;
    end
    if AE_flag
        %%%%%%%%Activation Extraction%%%%%
        Alast=1e5*ones(size(A));
        iter2=0; 
        rspect=zeros(F*T,N);
        while(norm(A(:)-Alast(:)) > eps && iter2 < maxiter )
    % %         norm(A(:)-Alast(:))
            Alast = A;
            for n=1:N
                %%%initializing the variables to speedup%%%
                Vaf=zeros(T+W-1,F,K);
                yhat2=zeros(T+W-1,F,K);
                sumkyhat2=zeros(T,F);
                temp4=0;%%%calculating the optimal stepsize
                for f=1:F
                    for k=1:K
                        yhat2(:,f,k)=ifft(fft(A(:,k,n),T+W-1).*fft(D(:,f,k),T+W-1));
        %                   yhat2(:,f,k)=conv(A(:,k,n),D(:,f,k),'valid');
                    end
                    sumkyhat2(:,f)=sum(yhat2(W:end,f,:),3);
        %             sumkyhat2(:,f)=sum(yhat2(:,f,:),3);
                    idx2=(f-1)*T+1:f*T;
                    rspect(idx2,n)=Y(idx2,n)-sumkyhat2(:,f);
                    temp3=zeros(T+W-1,K);
                    for k=1:K
                        temp3(:,k)=fft(flipud(D(:,f,k)),T+W-1);
                        Vaf(:,f,k)=ifft(fft(rspect(idx2,n),T+W-1).*temp3(:,k));                       
        %                 Vaf(:,f,k)=conv(rspect(idx2,n),flipud(D(:,f,k)));
                    end
                    temp4=temp4+max(sum(abs(temp3).^2,2));
                end
                Va=squeeze(sum(Vaf,2));
% %                 for f=1:F
% %                     for k=1:K
% % %                         temp3(:,f,k)=ifft(fft(Va(:,k),T+W-1).*fft(D(:,f,k),T+W-1));
% %         %                 temp3(:,f,k)=conv(Va(:,k),D(:,f,k),'valid');
% %                     end
% %                 end
% %         %         temp4=sum(temp3,3);
% %                 temp4=sum(temp3(W:end,:,:),3);
% %                 if norm(temp4(:)) ~= 0
% % %                     c=norm(Va(:))^2/norm(temp4(:))^2;
% %                     c=norm(Va(:)-lambda)^2/norm(temp4(:))^2;
% %                 else
% %                     c=0;
% %                 end
                  c=1/temp4;
%                 A(:,:,n) = A(:,:,n) + c*Va;
              %%%%%%%%%non-positivity constraints%%%%%%%%%%%%%%%%
               idxGTZero = (A(:,:,n) + c*(Va-lambda)) > zeros(T+W-1,K);
% %                if sum(sum(idxGTZero)) > 0
        %            idxLTZero = (A(:,:,n) + c*(Va+lambda)) < zeros(T+W-1,K);
                   Aprim=A(:,:,n);
                   Aprim(idxGTZero) = Aprim(idxGTZero) + c*(Va(idxGTZero)-lambda);
        %            Aprim(idxLTZero) = Aprim(idxLTZero) + c*(Va(idxLTZero)+lambda);
        %            Aprim(~(idxGTZero | idxLTZero))=0;
                   Aprim(~idxGTZero)=0;
                   A(:,:,n)=Aprim;
% %                end
            end
            iter2=iter2+1;
            obj2_err(i,iter2)=sum(sum(rspect.^2));
            obj2_Al1(i,iter2)=norm(A(:),1);
        end
        if display && rem(i,dispfactor)==0
            figure(4)
             for n=1:N
                 for k=1:K
                    subplot(N,1,n); 
                    plot(A(:,k,n),'color',color_vec(k,:));
                    hold on
                 end
                 hold off
             end
            % %  figure(5)
            % %  imshow(rspect);
        % %     figure(6)
        % %     plot(obj2);
    % %         pause(0.1)
%             %%%%%Reconstructed spectrogram vs. true spectrogram%%%%%
%             for n=1:N
%                 Yrecon=zeros(T,F,K);
%                 YreconSpectro=zeros(T,F);
%                     for f=1:F
%                         for k=1:K
% %                             Yrecon(:,f,k)=ifft(fft(A(:,k,n),T+W-1).*fft(D(:,f,k),T+W-1));
%                              Yrecon(:,f,k)=conv(A(:,k,n),D(:,f,k),'valid');
%                         end
% %                         YreconSpectro(:,f)=sum(Yrecon(W:end,f,:),3);
%                         YreconSpectro(:,f)=sum(Yrecon(:,f,:),3);
%                     end
%                     figure(7)
%                     subplot(N,1,n);imagesc(YreconSpectro',[0 max(max(YreconSpectro))]);colormap gray
%                     title(['reconstructed spectrogram ',num2str(n)])
%                     figure(8)
%                     subplot(N,1,n);imagesc(reshape(Y(:,n),T,[])',[0 max(Y(:,n))]);colormap gray
%                     title(['true spectrogram ',num2str(n)])
%             end
% %         save(['savedobj_sofar_compr_',num2str(parameter.pc),'.mat'],'D','A', 'obj1_err','obj1_Al1','obj2_err','obj2_Al1','i', 'objtot');
        end
        DL_flag=true;
    end
    %%%%%Objective value%%%%%%%
% %     i
    objtot(i,1)=obj2_err(i,end);
    objtot(i,2)=obj2_Al1(i,end);
%     if i>1 && (objtot(i-1)-objtot(i)) < eps
%         break
%     else
%         continue
%     end
end


end

