function [ind_min_error,accuracy_percent]=RBF_func(Train,Test,epoch,hid_layer,ro_w,ro_c,L_t,H_t,sigma_rbf);
% N : Number of feature vectors for training in each class
% M : Feature vector dimention in each class
% N2 : Number of test feature vectors for each class.
% C : Number of classes
% L_t : Low limit of target
% H_t : High limit of target
% Train must be a MxNxC matrix  
% Test must be a {C}(MxN2) cell arrey 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% for check!
% clear
% clc
% tic
% load('D:\MATLAB6p5\work\Train_sub6_mix.mat');
% load('D:\MATLAB6p5\work\Test_sub6_mix.mat');
% 
train6 = mix_train;
test6 = mix_test;

Train(:,:,1) = train6(:,:,1);
Train(:,:,2) = train6(:,:,2);
Test{1}(:,:) = test6{1}(:,:);
Test{2}(:,:) = test6{2}(:,:);

new_dim = 40;
[Train,Test]=PCA_Unsupervised(Train,Test,new_dim);
%
sigma_rbf = 12;
epoch = 50;
hid_layer = 60;
ro_w = .003;
ro_c = .01;
L_t = -.8;
H_t = .8;

Train = train;
Test = test;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% We convert 3-dim train matrix to 2-dim.Thus it will be Mx(C*N)

clf % Clear current figure window

%%%%%%%%%%%%%%%% Normalization
[Train,Test] = normalized(Train,Test);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[feature_dim,N,class_num] = size(Train);
A=Train(:,:,1);
for i=2:class_num
    A=[A Train(:,:,i)]; 
end
%%%
clear i
for i = 1 : class_num
    [zzz,end_data(i)] = size(Test{i}(:,:)); % end_data determines lenght of test matrix in each class
end
[f,TrainNo]=size(A); % TrainNo is the number of all feature vectors for training

%%%%%%%%%%%% kmeans clustering
% B3 = A';
% [I,c1] = kmeans(B3,class_num);
% c1 = c1';
% clear i
% for i = 1 : hid_layer
%     B4 = randperm(class_num);
%     c(:,i) = c1(:,B4(1));
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% finding mean vector

% % in this part,mean vector of each class (C terms) is calculated and all of
% % them are arreanged in a MxC matrix.
% A1=(Train(:,:,1))';
% B1=(Test{1}(:,:))';
% 
% mean_train = mean(A1); % Mx1
% mean_test = mean(B1); % Mx1
% 
% clear i
% for i=2:class_num
%     A1=[A1;(Train(:,:,i))'];
%     mean_train = [mean_train;mean((Train(:,:,i))')];
%     mean_test = [mean_test;mean((Test{i}(:,:))')];
% end
% 
% mean_train = mean_train';
% mean_test = mean_test';
% 
% % after this loop,mean_train and mean_test will be (MxC) - M :feature vector dimention
% 
% for jj = 1 : hid_layer
%     gg = randperm(class_num);
%     c(:,jj) = mean_train(:,gg(1));
% end

%%%%%%%%%%

%%%%%%%%%%%%%%% RBF Neural Network

w=randn(hid_layer,class_num); % Initial coeffitients for g functions

% for initial centers,we distribute random feature vectors of each class to hiddn layer 
B1 = randperm(TrainNo);
for hh = 1 : hid_layer % Initial centers for subspaces in RBF algorithm(we have 15(hid_layer) centers with dimention 36,for example)
    ind_c = B1(hh);
    c(:,hh) = Train(:,ind_c);
end
min_error = inf;

for ep=1:epoch
    
    E(ep) = 0; % Error of all training feature vectors in each epoch
    Et(ep) = 0; % Error of all test feature vectors in each epoch
        
    %%% Train
    s = 0;
    B=randperm(TrainNo); % for example randperm(6) might be the vector [3  2  6  4  1  5]
    clear i
	for i=1:class_num
        for j = 1 : N % N : Number of feature vectors for training in each class
            clear t g target z
            target = L_t*ones(1,class_num);
            s = s + 1;
            p=B(s); % start from a non repeated random feature vector
            t = Train(:,p); % Each feature vector for training
            target((floor((p-1)/N)+1)) = H_t;
            Eq = 0;
            %%%%%%
            coeff = 1/realpow(2*pi,1/feature_dim);
            norm_xn = (sum((repmat(t,1,hid_layer) - c ).*(repmat(t,1,hid_layer) - c ))');
            g = coeff * exp(-.5*norm_xn/(sigma_rbf^2));
            %%%%%%
            z = g' * w;
            %%%%%%
            dEqdwmj = 2 * g * ( z - target );
            w = w - ro_w * dEqdwmj;
            %%%%%%
            clear kk jj
%             for kk = 1 : hid_layer
%                 for jj = 1 : class_num
%                     dEqdcm = 2 * ( z(jj) - target(jj) ) * w(kk,jj) * (t' - c(:,kk)) * exp(-.5*norm(t' - c(:,kk)));
%                 end
%                 c(:,kk) = c(:,kk) - ro_c * dEqdcm;
%             end
%             dEqdcm = 2 * ( repmat(t,1,hid_layer) - c ) * w * ( z - target )' * g';
%             c = c - ro_c * dEqdcm;
            %%%%%%
            Eq = sum((z-target).*(z-target));
            %%%%
            E(ep)=E(ep)+Eq;
%             pause
		end
    end
%     w_all(:,:,ep) = w;
%     c_all(:,:,ep) = c;
    plot(1:ep,(E/TrainNo),'-*b');
    drawnow
%     
    
    %%% Test
    st = 0;
    clear i
    clear j
	for i = 1 : class_num
        clear target2
        target2 = L_t*ones(1,class_num);
        target2(i) = H_t;
        for j = 1 : end_data(i)
            clear t2 
            st = st + 1;
            t2 = Test{i}(:,j); % Each of feature vectors for training
            Eq2 = 0;
            %%%%%%
            coeff = 1/realpow(2*pi,1/feature_dim);
            norm_xn = (sum((repmat(t2,1,hid_layer) - c ).*(repmat(t2,1,hid_layer) - c ))');
            g2 = coeff * exp(-.5*norm_xn/(sigma_rbf^2));
            %%%%%%
            zt = g2' * w;
            %%%%%%
            Eq2 = sum((zt-target2).*(zt-target2));
            %%%%
            Et(ep)=Et(ep)+Eq2;
            
                       
            [m2,order2] = max(zt);
            RBF_5_class(st) = order2;
	%         pause
        end
	end
    if ( Et(ep)/st < min_error )
        min_error = Et(ep)/st;
        ind_min_error = ep;
        RBF_5_class_min = RBF_5_class;
    end
    hold on
    plot(1:ep,(Et/st),'-or');
    title(['Epoch ',int2str(ep)])
    xlabel('Epoch');
    ylabel('RBF Error Function');
    h = legend('Train','Test',1);
    drawnow
    
end

confusion = zeros(class_num,class_num);
m = 0;
s = 0;
for mn = 1 : class_num
    for hh = 1 : end_data(mn)
        m = hh + s;
        if(RBF_5_class_min(m) == mn)
            confusion(mn,mn) = confusion(mn,mn) + 1;
        else
            confusion(mn,RBF_5_class_min(m)) = confusion(mn,RBF_5_class_min(m)) + 1;
        end
    end
    s = s + end_data(mn);
end

clear i
for i = 1 : class_num
    main_diag(i) = confusion(i,i);
end

accuracy_percent = ( sum(main_diag./end_data)/class_num ) * 100

% confusion
% ind_min_error
% 
% save aaa w_all c_all
% toc