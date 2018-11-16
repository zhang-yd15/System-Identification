clear all
clc

hid_layer1 =15;
hid_layer2 =12;
epoch = 500;

%%%%%%%%%%% ball&beam

load('dryer.dat')

input_data =dryer(:,1);
output_data = dryer(:,2);
Delays =11;


train1 = [input_data(1:750)]';
test1 = [input_data(751:1000)]';

Z_target_train1 = [output_data(1:750)]';
Z_target_test1 = [output_data(751:1000)]';

for i = 0:Delays
    In(i+1,:) = train1(Delays+1-i:end-i);
end
train1 = In;
clear In
for i = 0:Delays
    In(i+1,:) = test1(Delays+1-i:end-i);
end
test1 = In;
Z_target_train1 = Z_target_train1(Delays+1:end);
Z_target_test1 = Z_target_test1(Delays+1:end);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net = newff(minmax(train1),[hid_layer1 hid_layer2  1],{'tansig' 'logsig'  'purelin'},'traingdx', 'learnpn');
net.trainParam.show = 1;
% net.trainParam.goal =0.002;
% net.trainparam.mu = .00000000001;
% net.trainparam.mu_max = 1.0000e+100000;
% net.trainparam.mu_inc = 1;
net.trainparam.min_grad = 1.00e-100;
net.trainParam.epochs = epoch;
V.P = test1;
V.T = Z_target_test1;
[net,tr] = train(net,train1,Z_target_train1,[],[],V);
estimated_Z1 = sim(net,test1);

Error_estimation = norm(Z_target_test1 - estimated_Z1);

figure
plot(Z_target_test1,'-b')
hold on
plot(estimated_Z1,'-r')
h = legend('Real Z','Estimated Z');