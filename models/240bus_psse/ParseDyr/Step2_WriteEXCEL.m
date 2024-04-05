%% Read PSSE .dyr file and write data into CSV/EXCEL file
clc;clear;

%% read dynamic data in .mat
fileXLS = 'Dynamic data.xlsx';
load('dyr.mat');
load('raw.mat');
GENROU = [GENROU 60*ones(length(GENROU(:,1)),1) PF_gen_Mbase];

% GENROU
writematrix(GENROU(:,1)',fileXLS,'Sheet','GENROU','Range','B1')
writecell(GENROU_ID',fileXLS,'Sheet','GENROU','Range','B2')
writematrix(GENROU(:,2:end)',fileXLS,'Sheet','GENROU','Range','B3')
description = {'Bus';'ID';'T''do';'T''''do';'T''qo';'T''''qo';'H';'D';...
    'Xd';'Xq';'X''d';'X''q';'X''''d=X''''q';'Xl';'S(1.0)';'S(1.2)';'fbase';'mvabase'};
writecell(description,fileXLS,'Sheet','GENROU','Range','A1')

% SEXS
writematrix(SEXS(:,1)',fileXLS,'Sheet','SEXS','Range','B1')
writecell(SEXS_ID',fileXLS,'Sheet','SEXS','Range','B2')
writematrix(SEXS(:,2:end)',fileXLS,'Sheet','SEXS','Range','B3')
description = {'Bus';'ID';'TA/TB';'TB(>0)';'K';'TE';'EMIN';'EMAX'};
writecell(description,fileXLS,'Sheet','SEXS','Range','A1')

% GAST
writematrix(GAST(:,1)',fileXLS,'Sheet','GAST','Range','B1')
writecell(GAST_ID',fileXLS,'Sheet','GAST','Range','B2')
writematrix(GAST(:,2:end)',fileXLS,'Sheet','GAST','Range','B3')
description = {'Bus';'ID';'R(Speed Droop)';'T1(>0)';'T2(>0)';'T3(>0)';...
    'Ambient Temperature Load Limit';'KT';'VMAX';'VMIN';'Dturb'};
writecell(description,fileXLS,'Sheet','GAST','Range','A1')

% HYGOV
writematrix(HYGOV(:,1)',fileXLS,'Sheet','HYGOV','Range','B1')
writecell(HYGOV_ID',fileXLS,'Sheet','HYGOV','Range','B2')
writematrix(HYGOV(:,2:end)',fileXLS,'Sheet','HYGOV','Range','B3')
description = {'Bus';'ID';'R, Permanent Droop';'r, Temporary Droop';...
    'Tr(>0) Governor Time Constant';'Tf(>0) Filter Time Constant';...
    'Tg(>0) Servo Time Constant';'VELM, Gate Velocity Limit';...
    'GMAX, Maximum Gate Limit';'GMIN, Minimum Gate Limit';...
    'TW(>0) Water Time Constant';'At, Turbine Gain';...
    'Dturb, Turbine Damping';'aNL, No Load Flow';};
writecell(description,fileXLS,'Sheet','HYGOV','Range','A1')


% TGOV1
writematrix(TGOV1(:,1)',fileXLS,'Sheet','TGOV1','Range','B1')
writecell(TGOV1_ID',fileXLS,'Sheet','TGOV1','Range','B2')
writematrix(TGOV1(:,2:end)',fileXLS,'Sheet','TGOV1','Range','B3')
description = {'Bus';'ID';'R';'T1(>0)(sec)';'V MAX';'V MIN';'T2(sec)';...
    'T3(>0)(sec)';'Dt'};
writecell(description,fileXLS,'Sheet','TGOV1','Range','A1')


% IEEEST
writematrix(IEEEST(:,1)',fileXLS,'Sheet','IEEEST','Range','B1')
writecell(IEEEST_ID',fileXLS,'Sheet','IEEEST','Range','B2')
writematrix(IEEEST(:,2:end)',fileXLS,'Sheet','IEEEST','Range','B3')
description = {'Bus';'ID';'ICS, Stab. Input Code (see manual for codes)';...
    'IB, Remote Bus No. (for input codes 2,5,6)';'A1';'A2';'A3';'A4';'A5';'A6';'T1';'T2';'T3';...
    'T4';'T5';'T6';'KS';'LSMAX';'LSMIN';'VCU';'VCL'};
writecell(description,fileXLS,'Sheet','IEEEST','Range','A1')



%% add missing IBR units
% REGCA
REGCA1_old = REGCA1;
REGCA1_ID_old = REGCA1_ID;

REGCA1 = zeros(length(PF_re_bus),length(REGCA1_old(1,:))+2);
REGCA1_ID = PF_re_id;

REGCA1(:,1) = PF_re_bus;
REGCA1(:,2:end-2) = ones(length(PF_re_bus),1)*REGCA1_old(1,2:end);
REGCA1(:,end-1) = 60;
REGCA1(:,end) = PF_re_Mbase;

% REECB
REECB1_old = REECB1;
REECB1_ID_old = REECB1_ID;

REECB1 = zeros(length(PF_re_bus),length(REECB1_old(1,:)));
REECB1_ID = PF_re_id;

REECB1(:,1) = PF_re_bus;
REECB1(:,2:end) = ones(length(PF_re_bus),1)*REECB1_old(1,2:end);

% REPCA
REPCA1_old = REPCA1;
REPCA1_ID_old = REPCA1_ID;

REPCA1 = zeros(length(PF_re_bus),length(REPCA1_old(1,:)));
REPCA1_ID = PF_re_id;

REPCA1(:,1) = PF_re_bus;
REPCA1(:,2:end) = ones(length(PF_re_bus),1)*REPCA1_old(1,2:end);


% REGCA
writematrix(REGCA1(:,1)',fileXLS,'Sheet','REGCA','Range','B1')
writecell(REGCA1_ID',fileXLS,'Sheet','REGCA','Range','B2')
writematrix(REGCA1(:,2:end)',fileXLS,'Sheet','REGCA','Range','B3')
description = {'Bus';'ID';'LVPL switch';'Tg, Converter time constant, sec';...
    'Rrpwr, LVPL ramp rate limit (pu/s)';'Brkpt, LVPL voltage 2(pu)';...
    'Zerox, LVPL voltage 1(pu)';'Lvpl1, LVPL gail(pu)';'Volim';'Lvpnt1';...
    'Lvpnt0';'Iolim';'Tfltr';'Khv';'Iqrmax';'Iqrmin';'Accel';'fbase';'mvabase'};
writecell(description,fileXLS,'Sheet','REGCA','Range','A1')


% REECB
writematrix(REECB1(:,1)',fileXLS,'Sheet','REECB','Range','B1')
writecell(REECB1_ID',fileXLS,'Sheet','REECB','Range','B2')
writematrix(REECB1(:,2:end)',fileXLS,'Sheet','REECB','Range','B3')
description = {'Bus';'ID';'Input this as 0. For remote bus control use the plant controller model';...
    'PFFLAG (Power factor flag): 1 - power factor control; 0: Q control';...
    'VFLAG:  1 if Q control; 0 voltage control';...
    'QFLAG: 1 if voltage/Q control; 0 if pf/Q control';...
    'PQFLAG: 1 for P priority, 0 for Q priority';...
    'Vdip(pu)';'Vup(pu)';'Trv(s)';'dbd1(pu)';'dbd2(pu)';'Kqv(pu)';...
    'Iqhl(pu)';'Iqll(pu)';'Vref0(pu)';'Tp(s)';'QMax(pu)';'QMin(pu)';...
    'VMAX(pu)';'VMIN(pu)';'Kqp(pu)';'Kqi(pu)';'Kvp(pu)';'Kvi(pu)';'Tiq(s)';...
    'dPmax(pu/s)(>0)';'dPmin(pu/s)(<0)';'PMAX(pu)';'PMIN(pu)';'Imax(pu)';'Tpord(s)'};
writecell(description,fileXLS,'Sheet','REECB','Range','A1')


% REPCA
writematrix(REPCA1(:,1)',fileXLS,'Sheet','REPCA','Range','B1')
writecell(REPCA1_ID',fileXLS,'Sheet','REPCA','Range','B2')
writematrix(REPCA1(:,2:4)',fileXLS,'Sheet','REPCA','Range','B3')
writecell(REPCA1_branch_ID',fileXLS,'Sheet','REPCA','Range','B6')
writematrix(REPCA1(:,5:end)',fileXLS,'Sheet','REPCA','Range','B7')
description = {'Bus';'ID';'Remote bus number or 0 for local voltage control';...
    'Monitored branch FROM bus';...
    'Monitored branch TO bus';...
    'Monitored branch ID (enter within single quotes)';...
    'VCFlag, droop flag (0: with droop,1: line drop compensation)';...
    'RefFlag, flag for V or Q control(0: Q control, 1: V control)';...
    'Fflag, 0: disable frequency control, 1: enable';...
    'Tfltr';'Kp';'Ki';'Tft';'Tfv';'Vfrz';'Rc';'Xc';'Kc';'emax';'emin';...
    'dbd1';'dbd2';'Qmax';'Qmin';'kpg';'kig';'Tp';...
    'fdbd1';'fdbd2';'femax';'femin';'Pmax';'Pmin';'Tg';'Ddn';'Dup'};
writecell(description,fileXLS,'Sheet','REPCA','Range','A1')

















