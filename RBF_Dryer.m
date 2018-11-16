clear all
clc

hid_layer = 10;
% hid_layer2 = 3;
epoch = 50;
spread = 1;

%%%%%%%%%%% Heat Exchanger
load('dryer.dat')

input_data = dryer(:,1);
output_data = dryer(:,2);

train1 = [input_data(1:750)]';
test1 = [input_data(751:1000)]';

Z_target_train1 = [output_data(1:750)]';
Z_target_test1 = [output_data(751:1000)]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net=newrb(train1,Z_target_train1,1e-2,spread,hid_layer,1); %% training of net, NEWRB(PR,T,GOAL,SPREAD,MN,DF)
estimated_Z1 = sim(net,test1); %% testing of net

Error_estimation = norm(Z_target_test1 - estimated_Z1);

figure
plot(Z_target_test1,'-b')
hold on
plot(estimated_Z1,'-r')
h = legend('Real Z','Estimated Z');