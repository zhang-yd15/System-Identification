
% BALL and BEAM Physical system
% System identification (using second method)
% Dividing data to 3 parts
% Using part 2 for training and parts 1 and 3 for validation

voltage=dryer(:,1);
temp=dryer(:,2);
voltage_train=dryer(1:750,1);
temp_train=dryer(1:750,2);
voltage_test=dryer(751:1000,1);
temp_test=dryer(751:1000,2);
dry=iddata(temp,voltage,0.1);
dry.InputName='volt';
dry.OutputName='celsius';
datam=merge(dry(1:150),dry(151:1000));   % Dividing data
ze=getexp(datam,2);       % Data for training
ze=detrend(ze);           % Omitting offset
zv=getexp(datam,1);       % Data for validation
zv=detrend(zv);           % Omitting offset

% *************************************************************************

figure(1)
NN=[5 4 4 8];
m=armax(ze,NN);   % Using ARMAX model
compare(zv,m)

% *************************************************************************

figure(2)
NN=[5 4 4 4 8];
m=bj(ze,NN);      % Using BOX-JENKINS (BJ) model
compare(zv,m)