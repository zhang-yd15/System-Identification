clc
clear all
load('dryer.dat');
voltage=dryer(:,1);
temp=dryer(:,2);
voltage_train=dryer(1:750,1);
temp_train=dryer(1:750,2);
voltage_test=dryer(751:1000,1);
temp_test=dryer(751:1000,2);
dry=iddata(temp,voltage,0.1);
dry.InputName='volt';
dry.OutputName='celsius';

% *************************************************************************

ze=dry(1:750);       % selecting data for training.
plot(ze(300:600))     % plot of datas between 300 and 600.  
ze=detrend(ze);       % omitting offset levels.
% figure(1)
% impulse(ze,'sd',3);   % calculation of impulse response.
m1=pem(ze);           % using state-space model (Prediction Error Maximum (PEM)).
zv=dry(751:1000);    % validation data.
zv=detrend(zv);
% figure(2)
% compare(zv,m1);       % validation of system.(note that the system is unstable!).

% *************************************************************************

% figure(3)
% bode(m1);             % bode plot.
figure(4)
step(m1,ze);          % step response
xlabel('Heater supply voltage (V)')
ylabel('Air temperature (-0C)')
% m2=arx(ze,[2 2 3]);   % using ARX model for identification.
% figure(5)
% compare(zv,m1,m2);    % ARX model and PEM model comparison.
% m2=arx(ze,[4 3 3]);   % increasing model order.
% figure(6)
% compare(zv,m1,m2);    % ARX model and PEM model comparison.
% m2=arx(ze,[4 3 5]);   % increasing model delay.
% figure(7)
% compare(zv,m1,m2);    % ARX model and PEM model comparison.
% m2=arx(ze,[5 4 6]);   % increasing model order and delay.
% figure(8)
% compare(zv,m1,m2);    % ARX model and PEM model comparison.