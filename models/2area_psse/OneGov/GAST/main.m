clc;clear;close all;warning off;

dataT = readtable('out_G1.xlsx');


Pgen_t = dataT(:,[1,6,7,8,9]);
Pm_t = dataT(:,[1,18,19,20,21]);
SPD_t = dataT(:,[1,22,23,24,25]);
% Qgen_t = readtable('Qgen.xlsx');
% EFD_t = readtable('EFD.xlsx');
% VOLT_t = readtable('VOLT.xlsx');


Pgen = table2array(Pgen_t);
Pm = table2array(Pm_t);
SPD = table2array(SPD_t);


% EFD = table2array(EFD_t);
% Qgen = table2array(Qgen_t);
% VOLT = table2array(VOLT_t);



%%
t_shift = 0;
t_end = 50;


figure(1)
set(gcf, 'Position',  [100, 600, 400, 200])
plot(SPD(:,1)+t_shift,SPD(:,2:end)*60+60)
xlabel('Time (s)')
title('SPD')
xlim([0 t_end])
ylim([59.7 60.08])

figure(2)
set(gcf, 'Position',  [600, 600, 400, 200])
plot(Pgen(:,1)+t_shift,Pgen(:,2:end))
xlabel('Time (s)')
title('Pgen')
xlim([0 t_end])

figure(3)
set(gcf, 'Position',  [1100, 600, 400, 200])
plot(Pm(:,1)+t_shift,Pm(:,2:end))
xlabel('Time (s)')
title('Pm')
xlim([0 t_end])

% figure(4)
% set(gcf, 'Position',  [100, 200, 400, 200])
% plot(Qgen(:,1)+t_shift,Qgen(:,2:end))
% xlabel('Time (s)')
% title('Qgen')
% xlim([0 t_end])
% 
% figure(5)
% set(gcf, 'Position',  [600, 200, 400, 200])
% plot(VOLT(:,1)+t_shift,VOLT(:,2:end))
% xlabel('Time (s)')
% title('VOLT')
% xlim([0 t_end])
% ylim([0.88 1.18])
% 
% figure(6)
% set(gcf, 'Position',  [1100, 200, 400, 200])
% plot(EFD(:,1)+t_shift,EFD(:,2:end))
% xlabel('Time (s)')
% title('EFD')
% xlim([0 t_end])




%%
% figure(100)
% subplot(2,1,1)
% plot(SPD(:,1),SPD(:,20)*60+60)
% grid on;
% xlim([0 5])
% subplot(2,1,2)
% plot(Pgen(:,1),Pgen(:,20))
% xlim([0 5])
% grid on;
% 
% figure(101)
% subplot(2,1,1)
% plot(SPD(:,1),SPD(:,24)*60+60)
% grid on;
% xlim([0 5])
% subplot(2,1,2)
% plot(Pgen(:,1),Pgen(:,24))
% xlim([0 5])
% grid on;


% figure(100)
% plot(SPD(:,20)*60,Pgen(:,20)-Pgen(1,20))
% grid on
% 
% figure(101)
% plot(SPD(:,24)*60,Pgen(:,24)-Pgen(1,24))
% grid on

%% 
% clc
% GenId = (Pm_t.Properties.VariableNames)';
% Hydro_units = [];
% for i = 1:length(GenId)
%     tempstr = char(GenId(i));
%     if tempstr(end)=='H'
%         Hydro_units = [Hydro_units;i];
%     end
% end
% 
% [Hydro_units, (Pm(end,Hydro_units) - Pm(1,Hydro_units))'];
% 
% % figure(100)
% % plot(Pm(:,1),Pm(:,Hydro_units) - ones(length(Pm(:,1)),1)*Pm(1,Hydro_units))

kk = 1804;

macbase = (Pgen(1,2:end)./Pm(1,2:end))'*100;

dPm = Pm(:,2:end) - ones(length(Pm(:,1)),1)*Pm(1,2:end);
dP = (Pgen(kk,2:end) - Pgen(1,2:end))'./(macbase/100);
df = SPD(kk,2:end);

droopgain = -SPD(:,2:end)./dPm;
% idx = find(isnan(droopgain));
% droop_SG = droopgain;
% droop_SG(idx) = [];
% 
% SPD_SG = SPD_t;
% SPD_SG(:,idx+1) = [];
% 
% 
% Id_SG = (SPD_SG.Properties.VariableNames)';
% Id_SG(1) = [];

figure(4)
set(gcf, 'Position',  [100, 200, 400, 200])
plot(SPD(:,1)+t_shift,droopgain)
xlabel('Time (s)')
title('Estimated droop')
xlim([0 t_end])
ylim([0 1])