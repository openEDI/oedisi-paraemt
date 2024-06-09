% clc;clear;

%% time step used in ParaEMT
dt = 20*50e-6;

%% read simulation results
dataV3phase = readtable('paraemt.emt_v_faultbus5_faulttype10.csv');
dataI3phase= readtable('paraemt.emt_ibranch_faultbus5_faulttype10.csv');

%% load data
t = table2array(dataV3phase(2:end,1))*dt; 
st = 1;
bus_V3 =  table2array(dataV3phase(2:end,st+1:end));
branch_I3 =  table2array(dataI3phase(2:end,st+1:end));

%% Current
branch_k=5; % Which branch
fault_t=0.5;      
k=branch_k;
Bran_num=size(branch_I3)/3; Bran_num=Bran_num(2);
% if branch_k<fault_tripline
Ia_preon=branch_I3(1:round(fault_t/dt),k);    Ia_post=branch_I3(round(fault_t/dt)+1:end,k);     Ia=[Ia_preon;Ia_post];
Ib_preon=branch_I3(1:round(fault_t/dt),k+Bran_num);   Ib_post=branch_I3(round(fault_t/dt)+1:end,k+Bran_num);   Ib=[Ib_preon;Ib_post];
Ic_preon=branch_I3(1:round(fault_t/dt),k+2*Bran_num); Ic_post=branch_I3(round(fault_t/dt)+1:end,k+2*Bran_num); Ic=[Ic_preon;Ic_post];

% Ia_save=Ia;
% Ib_save=Ib;
% Ic_save=Ic;
plot(Ia_save-Ia)
%% plot
% close all;
% tstart=0;
% tend = max(t);
% xshift = 0;
% yshift = -200;

% plot(Ia+Ib+Ic)
%% Bus Voltage
k=2; % Which bus
% bus_n=11;
% figure(11)
% clf;hold on;
% set(gcf, 'Position',  [50+xshift, 750+yshift, 400, 200])
% plot(t,bus_V3(:,k),t,bus_V3(:,k+bus_n),t,bus_V3(:,k+2*bus_n))
% legend('va','vb','vc')
% box on;
% xlabel('Time (s)')
% ylabel('Three-phase internal voltage (kV)')
% set(gcf, 'Position',  [1000, 500, 600, 300])
% title('Three phase bus voltage')
% xlim([tstart tend])

%% branch current
% figure(12)
% clf;hold on;
% set(gcf, 'Position',  [50+xshift, 750+yshift, 400, 200])
% plot(t,Ia,t,Ib,t,Ic)
% legend('ia','ib','ic')
% box on;
% xlabel('Time (s)')
% ylabel('Three-phase branch current (pu)')
% set(gcf, 'Position',  [1000, 500, 600, 300])
% title('Three phase branch current')
% xlim([tstart tend])


