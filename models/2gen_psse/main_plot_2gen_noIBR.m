clc;clear;

%% read simulation results
table_ = readtable('fault.xlsx');
data = table2array(table_);

tend = 15;

t = data(:,1);
mac_w = (data(:,2:3)*60+60)*2*pi;
mac_te = data(:,4:5)*100;
sexs_EFD = data(:,6:7);
gast_pm = data(:,8:9);
bus_vtm = data(:,[10,12,14,16]);


% plot
close all;
flag_sg = 1;
dev_flag = 0;


if flag_sg == 1   
    figure(1)
    clf;hold on;
    set(gcf, 'Position',  [100, 600, 400, 200])
%     plot(t,(mac_w - mac_w(:,1)*ones(1,length(mac_w(1,:))))/2/pi/60)  % relative rotor speed (ref:gen 1)
%     plot(t,mac_w/2/pi/60-1) % rotor speed deviation, pu
    plot(t,mac_w/2/pi)
    box on;
%     ylim([59.9 60.1])
    xlim([0 tend])
    title('\omega')
    
    figure(2)
    clf;hold on;
    set(gcf, 'Position',  [505, 600, 400, 200])
    plot(t,mac_te /100)
    
    box on;
%     ylim([-100 2000])
    xlim([0 tend])
    title('pe')

    
    
    figure(3)
    clf;hold on;
    set(gcf, 'Position',  [910, 600, 400, 200])
    plot(t,sexs_EFD - dev_flag*ones(length(t),1)*sexs_EFD(1,:))
    box on;
    title('EFD')
    xlim([0 tend])
%     ylim([-100 500])
    
    figure(7)
    clf;hold on;
    set(gcf, 'Position',  [910, 310, 400, 200])
    plot(t,gast_pm - dev_flag*ones(length(t),1)*gast_pm(1,:))
    box on;
    title('Pm GAST')
    xlim([0 tend])
    ylim([0 2])
    
%     figure(4)
%     clf;hold on;
%     set(gcf, 'Position',  [1315, 680, 400, 200])
%     plot(t,ieeest_vs)
%     box on;
%     title('Vs IEEEST')
%     xlim([0 tend])
    
end





figure(100)
clf;hold on;
set(gcf, 'Position',  [1400, 50, 400, 200])
plot(t,bus_vtm)
box on;
if dev_flag==1
    title('Bus voltage mag deviation')
else
    title('Bus voltage mag')
end
xlim([0 tend])
legend('Bus 1','Bus 2','Bus 3','Bus 4')
grid on
% ylim([0.8 1.5])
% ylim([0.88 1.18])
% ylim([0.94 1.05])
% ylim([0.75 1.1])

