clc;clear;

% EFD_t = readtable('EFD.xlsx');
Pgen_t = readtable('Pgen.xlsx');
% Pm_t = readtable('Pm.xlsx');
% Qgen_t = readtable('Qgen.xlsx');
SPD_t = readtable('SPD.xlsx');
VOLT_t = readtable('VOLT.xlsx');

% EFD = table2array(EFD_t);
Pgen = table2array(Pgen_t);
% Pm = table2array(Pm_t);
% Qgen = table2array(Qgen_t);
SPD = table2array(SPD_t);
VOLT = table2array(VOLT_t);



%%
tshft = 0;
figure(1)
set(gcf, 'Position',  [100, 600, 500, 400])
plot(SPD(:,1)-tshft,SPD(:,2:end)*60+60)
xlabel('Time (s)')
title('SPD')
xlim([0 10])
ylim([59.72 60.08])

figure(2)
set(gcf, 'Position',  [600, 600, 500, 400])
plot(Pgen(:,1)-tshft,Pgen(:,2:end))
xlabel('Time (s)')
title('Pgen')
xlim([0 10])
ylim([-2 90])

% figure(3)
% set(gcf, 'Position',  [1100, 600, 500, 200])
% plot(Pm(:,1),Pm(:,2:end))
% xlabel('Time (s)')
% title('Pm')
% xlim([0 20])
% 
% figure(4)
% set(gcf, 'Position',  [100, 200, 500, 200])
% plot(Qgen(:,1),Qgen(:,2:end))
% xlabel('Time (s)')
% title('Qgen')
% xlim([0 20])

figure(5)
set(gcf, 'Position',  [600, 200, 500, 400])
plot(VOLT(:,1)-tshft,VOLT(:,2:end))
xlabel('Time (s)')
title('VOLT')
xlim([0 10])
ylim([0.85 1.2])

% figure(6)
% set(gcf, 'Position',  [1100, 200, 500, 200])
% plot(EFD(:,1),EFD(:,2:end))
% xlabel('Time (s)')
% title('EFD')
% xlim([0 20])