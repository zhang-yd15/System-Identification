function [ind_min_error,accuracy_percent]=MLP_BP_func(Train,Test,epoch,hid_layer,ro,L_t,H_t);
% N : Number of feature vectors for training in each class
% M : Feature vector dimention in each class
% N2 : Number of test feature vectors for each class.
% C : Number of classes
% L_t : Low limit of target
% H_t : High limit of target
% Train must be a MxNxC matrix  
% Test must be a {C}(MxN2) cell arrey 

%%%%%%%%%%%%%%%%%%%%%%%%%%%% for check!
% for modid,ro = 0.1 and hid_layer = 15  and Limit = +-.8 was proper.
% clear
% clc
% load Train
% load Test
% epoch = 50;
% hid_layer = 15;
% ro = .1;
% L_t = -.8;
% H_t = .8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% We convert 3-dim train matrix to 2-dim.Thus it will be Mx(C*N)

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

%%%%%%%%%% MLP Backpropagation Classifier

w=rand(feature_dim,hid_layer); % Initial weights for first layer
u=rand(hid_layer,class_num); % Initial weights for hidden layer
% B = randperm(TrainNo);
min_error = 100000000000000000000000;
for ep=1:epoch
    ro = ro*.99;
    E(ep)=0; % Error of all training feature vectors in each epoch
    s1 = 0;
    B = randperm(TrainNo);
	for i = 1 : class_num
        for j = 1 : N % N : Number of feature vectors for training in each class 
            clear t target s r z y dEqdWnm dEqdUmj
            target = L_t*ones(1,class_num);
            s1 = s1 + 1;
            p=B(s1); % start from a non repeated random feature vector
            t = Train(:,p); % Each feature vector for training,Nx1
            target((floor((p-1)/N)+1)) = H_t;
            
            %%%%
            r=zeros(1,hid_layer); 
            s=zeros(1,class_num);
            z=zeros(1,class_num);
            y=zeros(1,hid_layer);
            Eq=0;
            
            %%%%
            r = w' * t; % (hid_layer x 1)
            y=tanh(r); % (hid_layer x 1)
            
            %%%%
            s = u' * y; % Cx1
            z=tanh(s); % Cx1
            
            %%%%
            dg = (ones(class_num,1)-z).*(ones(class_num,1)+z); % g'(Sj) = .5*(1-Zj)*(1+Zj) - Cx1
            e_g = (z-target').*dg; % ej=(Zj-target'(j)) is a Cx1 matrix , e_g(j)=ej*g'(Sj) 
            sig_u = u * e_g; % sig_u(m) = sigma(ej*g'(Sj)*Umj) is a (hid_layer x 1) matrix
            dh = (ones(hid_layer,1)-y).*(ones(hid_layer,1)+y); % h'(Rm) = .5*(1-Ym)*(1+Ym) - (hid_layer x 1)
            dEqdWnm = .5*(sig_u.*dh)*t'; % (hid_layer x N)
            w = w - ro * dEqdWnm'; % w : (hid_layer x M)
            
            %%%%
            dEqdUmj = .5 * e_g * y'; % e_g = [e1*g'(S1) e2*g'(S2) ....]',dEqdUmj: (C x hid_layer) 
            u = u - ro * dEqdUmj';
            
            %%%%
            Eq = sum((z-target').*(z-target')); 
            
            %%%%
            E(ep)=E(ep)+Eq;
		end
    end
%     w_all(:,:,ep) = w;
%     u_all(:,:,ep) = u;
    plot(1:ep,(E/TrainNo),'-*b');
    drawnow
%     
    %%% Test
    
    Et(ep) = 0; % Error of all test feature vectors in each epoch
    st = 0;
    clear i j
	for i = 1 : class_num
        clear target2
        target2 = L_t*ones(1,class_num);
        target2(i) = H_t;
        for j = 1 : end_data(i)
            clear t2 
            st = st + 1;
            t2 = Test{i}(:,j); % Each of feature vectors for training
            Eq2 = 0;
            %%%%
            r2=zeros(1,hid_layer);
            s2=zeros(1,class_num);
            z2=zeros(1,class_num);
            y2=zeros(1,hid_layer);
            
            %%%%
            r2 = w' * t2; % (hid_layer x 1)
            y2=tanh(r2); % (hid_layer x 1)
            
            %%%%
            s2 = u' * y2; % Cx1
            z2=tanh(s2); % Cx1
            
            %%%%
            Eq2 = sum((z2-target2').*(z2-target2')); 
            
            %%%%
            Et(ep)=Et(ep)+Eq2;
            
            [m2,order2] = max(z2);
            MLP_5_class(st) = order2;
        end
	end
    if ( Et(ep)/st < min_error )
        min_error = Et(ep)/st;
        ind_min_error = ep;
        MLP_5_class_min = MLP_5_class;
    end
    hold on
    plot(1:ep,(Et/st),'-or');
    title(['Epoch ',int2str(ep)])
    xlabel('Epoch');
    ylabel('MLP Error Function');
    h = legend('Train','Test',1); 
    drawnow
end

confusion = zeros(class_num,class_num);
m = 0;
s = 0;
for mn = 1 : class_num
    for hh = 1 : end_data(mn)
        m = hh + s;
        if(MLP_5_class_min(m) == mn)
            confusion(mn,mn) = confusion(mn,mn) + 1;
        else
            confusion(mn,MLP_5_class_min(m)) = confusion(mn,MLP_5_class_min(m)) + 1;
        end
    end
    s = s + end_data(mn);
end

clear i
for i = 1 : class_num
    main_diag(i) = confusion(i,i);
end

accuracy_percent = ( sum(main_diag./end_data)/class_num ) * 100;

% confusion
% ind_min_error

% save MLP_BP_classifier w_all u_all 
