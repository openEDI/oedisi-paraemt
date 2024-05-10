clc;clear;

%% time step used in ParaEMT
dt = 2*50e-6;

%% read simulation results
dataT = readtable('paraemt.emt_x.csv');
dataTbus = readtable('paraemt.emt_bus.csv');
dataV3phase = readtable('paraemt.emt_v.csv');
dataI3phase = readtable('paraemt_ibranch.csv');

%% parce the data
bus_n = 11;
bus_odr = 6;
load_n = 2;
load_odr = 4;

Num_gen=4;
gen_genrou_n = Num_gen;
exc_sexs_n = Num_gen;
gov_tgov1_n = Num_gen;
ibr_n = 10-Num_gen;

gen_genrou_odr = 18;
exc_sexs_odr = 2;
gov_tgov1_odr = 3;
gov_hygov_odr = 5;
gov_hygov_n = 0;
gov_gast_n = 0;
gov_gast_odr = 4;
pss_ieeest_odr = 10;
pss_ieeest_n = 0;

pll_odr = 3;
ibr_odr = 13;

%% load data
t = table2array(dataT(2:end,1))*dt;
st = 1;
mac_dt = table2array(dataT(2:end,st+1:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_w = table2array(dataT(2:end,st+2:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_id = table2array(dataT(2:end,st+3:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_iq = table2array(dataT(2:end,st+4:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_ifd = table2array(dataT(2:end,st+5:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_i1d = table2array(dataT(2:end,st+6:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_i1q = table2array(dataT(2:end,st+7:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_i2q = table2array(dataT(2:end,st+8:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_ed = table2array(dataT(2:end,st+9:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_eq = table2array(dataT(2:end,st+10:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_psyd = table2array(dataT(2:end,st+11:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_psyq = table2array(dataT(2:end,st+12:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_psyfd = table2array(dataT(2:end,st+13:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_psy1q = table2array(dataT(2:end,st+14:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_psy1d = table2array(dataT(2:end,st+15:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_psy2q = table2array(dataT(2:end,st+16:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_te = table2array(dataT(2:end,st+17:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));
mac_qe = table2array(dataT(2:end,st+18:gen_genrou_odr:(st+gen_genrou_odr*gen_genrou_n)));

st = 1 + gen_genrou_odr*gen_genrou_n;
sexs_v1 = table2array(dataT(2:end,st+1:exc_sexs_odr:(st+exc_sexs_odr*exc_sexs_n)));
sexs_EFD = table2array(dataT(2:end,st+2:exc_sexs_odr:(st+exc_sexs_odr*exc_sexs_n)));

st = 1 + gen_genrou_odr*gen_genrou_n + exc_sexs_odr*exc_sexs_n;
tgov1_p1 = table2array(dataT(2:end,st+1:gov_tgov1_odr:(st+gov_tgov1_odr*gov_tgov1_n)));
tgov1_p2 = table2array(dataT(2:end,st+2:gov_tgov1_odr:(st+gov_tgov1_odr*gov_tgov1_n)));
tgov1_pm = table2array(dataT(2:end,st+3:gov_tgov1_odr:(st+gov_tgov1_odr*gov_tgov1_n)));

st = 1 + gen_genrou_odr*gen_genrou_n + exc_sexs_odr*exc_sexs_n + gov_tgov1_odr*gov_tgov1_n;
hygov_xe = table2array(dataT(2:end,st+1:gov_hygov_odr:(st+gov_hygov_odr*gov_hygov_n)));
hygov_xc = table2array(dataT(2:end,st+2:gov_hygov_odr:(st+gov_hygov_odr*gov_hygov_n)));
hygov_xg = table2array(dataT(2:end,st+3:gov_hygov_odr:(st+gov_hygov_odr*gov_hygov_n)));
hygov_xq = table2array(dataT(2:end,st+4:gov_hygov_odr:(st+gov_hygov_odr*gov_hygov_n)));
hygov_pm = table2array(dataT(2:end,st+5:gov_hygov_odr:(st+gov_hygov_odr*gov_hygov_n)));

st = 1 + gen_genrou_odr*gen_genrou_n + exc_sexs_odr*exc_sexs_n + gov_tgov1_odr*gov_tgov1_n +gov_hygov_odr*gov_hygov_n;
gast_p1 = table2array(dataT(2:end,st+1:gov_gast_odr:(st+gov_gast_odr*gov_gast_n)));
gast_p2 = table2array(dataT(2:end,st+2:gov_gast_odr:(st+gov_gast_odr*gov_gast_n)));
gast_p3 = table2array(dataT(2:end,st+3:gov_gast_odr:(st+gov_gast_odr*gov_gast_n)));
gast_pm = table2array(dataT(2:end,st+4:gov_gast_odr:(st+gov_gast_odr*gov_gast_n)));

st = 1 + gen_genrou_odr*gen_genrou_n + exc_sexs_odr*exc_sexs_n + gov_tgov1_odr*gov_tgov1_n +gov_hygov_odr*gov_hygov_n +gov_gast_odr*gov_gast_n;
ieeest_y1 = table2array(dataT(2:end,st+1:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_y2 = table2array(dataT(2:end,st+2:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_y3 = table2array(dataT(2:end,st+3:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_y4 = table2array(dataT(2:end,st+4:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_y5 = table2array(dataT(2:end,st+5:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_y6 = table2array(dataT(2:end,st+6:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_y7 = table2array(dataT(2:end,st+7:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_x1 = table2array(dataT(2:end,st+8:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_x2 = table2array(dataT(2:end,st+9:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));
ieeest_vs = table2array(dataT(2:end,st+10:pss_ieeest_odr:(st+pss_ieeest_odr*pss_ieeest_n)));

bus_pll_ze = table2array(dataTbus(2:end,2:bus_odr:(bus_odr*bus_n)));
bus_pll_de = table2array(dataTbus(2:end,3:bus_odr:(bus_odr*bus_n)));
bus_pll_we = table2array(dataTbus(2:end,4:bus_odr:(bus_odr*bus_n)));
bus_vt = table2array(dataTbus(2:end,5:bus_odr:(bus_odr*bus_n)));
bus_vtm = table2array(dataTbus(2:end,6:bus_odr:(bus_odr*bus_n)));
bus_dvtm = table2array(dataTbus(2:end,7:bus_odr:(bus_odr*bus_n)));

%%   
st = 1;
bus_V3 =  table2array(dataV3phase(2:end,st+1:end));

%% Current
branch_k=1; st = 1;  fault_t=0.1+5/60;  fault_tripline = 3;  
branch_I3 =  table2array(dataI3phase(2:end,st+1:end));
k=branch_k;
Bran_num=size(branch_I3)/3; Bran_num=Bran_num(2);
% if branch_k<fault_tripline
Ia_preon=branch_I3(1:round(fault_t/dt),k);    Ia_post=branch_I3(round(fault_t/dt)+1:end,k);     Ia=[Ia_preon;Ia_post];
Ib_preon=branch_I3(1:round(fault_t/dt),k+Bran_num);   Ib_post=branch_I3(round(fault_t/dt)+1:end,k+Bran_num);   Ib=[Ib_preon;Ib_post];
Ic_preon=branch_I3(1:round(fault_t/dt),k+2*Bran_num); Ic_post=branch_I3(round(fault_t/dt)+1:end,k+2*Bran_num); Ic=[Ic_preon;Ic_post];
% else
% Ia_preon=branch_I3(1:round(fault_t/dt),k);    Ia_post=branch_I3(round(fault_t/dt)+1:end,k-1);     Ia=[Ia_preon;Ia_post];
% Ib_preon=branch_I3(1:round(fault_t/dt),k+Bran_num);   Ib_post=branch_I3(round(fault_t/dt)+1:end,k+Bran_num-1);   Ib=[Ib_preon;Ib_post];
% Ic_preon=branch_I3(1:round(fault_t/dt),k+2*Bran_num); Ic_post=branch_I3(round(fault_t/dt)+1:end,k+2*Bran_num-1); Ic=[Ic_preon;Ic_post];
% end

%% plot
close all;
tstart=0;
tend = max(t);
xshift = 0;
yshift = -200;

dev_flag = 0;

figure(1)
clf;hold on;
set(gcf, 'Position',  [1000, 500, 600, 300])
plot(t,mac_dt(:,:)-mac_dt(:,1))
box on;
xlim([tstart tend])
title('\delta')

figure(2)
clf;hold on;
set(gcf, 'Position',  [1000, 500, 600, 300])
plot(t,mac_w(:,:)/2/pi)
box on;
ylim([59.9 60.05])
xlim([tstart tend])
xlabel('Time (s)')
ylabel('Generator rotor frequency (Hz)')
title('\omega')

figure(3)
clf;hold on;
set(gcf, 'Position',  [950+xshift, 750+yshift, 400, 200])
plot(t,mac_te(:,:)-dev_flag*mac_te(1,:))
box on;
xlim([tstart tend])
xlabel('Time (s)')
ylabel('Generator electrical power (pu)')
set(gcf, 'Position',  [1000, 500, 600, 300])
title('Generator electrical power')

figure(4)
clf;hold on;
set(gcf, 'Position',  [1400+xshift, 750+yshift, 400, 200])
plot(t,mac_qe-dev_flag*mac_qe(1,:))
box on;
xlim([tstart tend])
title('qe')

figure(5)
clf;hold on;
set(gcf, 'Position',  [50+xshift, 450+yshift, 400, 200])
plot(t,sexs_EFD-dev_flag*sexs_EFD(1,:))
box on;
xlabel('Time (s)')
ylabel('Generator Efd (pu)')
set(gcf, 'Position',  [1000, 500, 600, 300])
title('Generator exciter filed voltage')
xlim([tstart tend])

figure(6)
clf;hold on;
set(gcf, 'Position',  [1400+xshift, 450+yshift, 400, 200])
plot(t,tgov1_pm-dev_flag*tgov1_pm(1,:))
box on;
xlim([tstart tend])
title('Pm GAST')

%% Bus Voltage
figure(11)
clf;hold on;
k=1;
set(gcf, 'Position',  [50+xshift, 750+yshift, 400, 200])
plot(t,bus_V3(:,k),t,bus_V3(:,k+bus_n),t,bus_V3(:,k+2*bus_n))
legend('va','vb','vc')
box on;
xlabel('Time (s)')
ylabel('Three-phase internal voltage (kV)')
set(gcf, 'Position',  [1000, 500, 600, 300])
title('Three phase bus voltage')
xlim([tstart tend])

%% branch current
figure(12)
clf;hold on;
set(gcf, 'Position',  [50+xshift, 750+yshift, 400, 200])
plot(t,Ia,t,Ib,t,Ic)
legend('ia','ib','ic')
box on;
xlabel('Time (s)')
ylabel('Three-phase branch current (pu)')
set(gcf, 'Position',  [1000, 500, 600, 300])
title('Three phase branch current')
xlim([tstart tend])

%% Network
figure(30)
clf;hold on;
set(gcf, 'Position',  [950+xshift, 450+yshift, 400, 200])
plot(t,bus_vtm-dev_flag*ones(length(t),1)*bus_vtm(1,:))
box on;
xlabel('Time (s)')
ylabel('IBR active power (pu)')
set(gcf, 'Position',  [1000, 500, 600, 300])
title('Bus voltage magnitude')
xlim([tstart tend])

