
% BALL and BEAM Physical system
% System identification (using third method)
% In this method ,first allowed the system to find the pure time delay
%     automatically by 'delayest' command
% But in this case the fitness was low.
% So we selected nk manually and reached a best fitness.

% *************************************************************************

voltage=dryer(:,1);
temp=dryer(:,2);
voltage_train=dryer(1:750,1);
temp_train=dryer(1:750,2);
voltage_test=dryer(751:1000,1);
temp_test=dryer(751:1000,2);
dry=iddata(temp,voltage,0.1);
dry.InputName='volt';
dry.OutputName='celsius';
ze=dry(1:750);
ze=detrend(ze);
zv=dry(751:1000);
zv=detrend(zv);
figure(1)
plot(ze(100:500))   % Plotting  some part of training data

% *************************************************************************

% Automatically pure time delay selection

nk=delayest(dry);
nk=delayest(dry)

%   nk =

%        11

NN=struc(1:5,1:5,nk);
v=arxstruc(ze,zv,NN);
nn=selstruc(v,0);
m=arx(ze,nn);
figure(2)
compare(zv,m)   % Note: while the fitness is low the system is stable

% *************************************************************************

% Manually pure time delay selection

nk=4;
NN=struc(5,4,nk);
v=arxstruc(ze,zv,NN);
nn=selstruc(v,0);
m=arx(ze,nn);
figure(3)
compare(zv,m)   % Note: while the fitness is high the system is unstable