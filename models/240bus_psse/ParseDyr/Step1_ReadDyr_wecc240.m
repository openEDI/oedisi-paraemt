%% Read PSSE .dyr file and write data into CSV/EXCEL file
clc;clear;

%% path and file name
filedyr = 'WECC240_dynamics_UPV_v04.dyr';



%% read file
fid = fopen(filedyr,'rt');
if fid == -1 
    fprintf('Error opening the dyr-file!\n');
end


GENROU = [];
GENROU_ID = [];
n_genrou = 0;
SEXS = [];
SEXS_ID = [];
n_sexs = 0;
GAST = [];
GAST_ID = [];
n_gast = 0;
HYGOV = [];
HYGOV_ID = [];
n_hygov = 0;
TGOV1 = [];
TGOV1_ID = [];
n_tgov1 = 0;
IEEEST = [];
IEEEST_ID = [];
n_ieeest = 0;

REGCA1 = [];
REGCA1_ID = [];
n_regca = 0;
REECB1 = [];
REECB1_ID = [];
n_reecb = 0;
REPCA1 = [];
REPCA1_ID = [];
REPCA1_branch_ID = [];
n_repca = 0;

while 1
    nextline = fgetl(fid);
    if ~ischar(nextline)
        break
    end
    s1 = regexp(nextline,   '\s+','split');
    type = char(s1(3));
    type = type(2:(length(type)-1));
    
    switch type
        case 'GENROU'
            n_genrou = n_genrou + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            temp1 = cell2num( [s1(5:end)],0);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end)],0);
            
            nextline = fgetl(fid);
            s3 = regexp(nextline,   '\s+','split');
            temp3 = cell2num( [s3(2:end-1)],0);
            
            temp = [bus temp1 temp2 temp3];
            GENROU = [GENROU;temp];
            GENROU_ID = [GENROU_ID;ID];
            
        case 'SEXS'
            n_sexs = n_sexs + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            temp1 = cell2num( [s1(5:end)],0);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end-1)],0);
            
            temp = [bus temp1 temp2];
            
            SEXS = [SEXS;temp];
            SEXS_ID = [SEXS_ID;ID];
            
        case 'GAST'
            n_gast = n_gast + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            temp1 = cell2num( [s1(5:end)],0);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end-1)],0);
            
            temp = [bus temp1 temp2];
            
            GAST = [GAST;temp];
            GAST_ID = [GAST_ID;ID];
            
        case 'HYGOV'
            n_hygov = n_hygov + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            temp1 = cell2num( [s1(5:end)],0);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end)],0);
            
            nextline = fgetl(fid);
            s3 = regexp(nextline,   '\s+','split');
            temp3 = cell2num( [s3(2:end-1)],0);
            
            temp = [bus temp1 temp2 temp3];
            HYGOV = [HYGOV;temp];
            HYGOV_ID = [HYGOV_ID;ID];
            
        case 'TGOV1'
            n_tgov1 = n_tgov1 + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            temp1 = cell2num( [s1(5:end)],0);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end-1)],0);
            
            temp = [bus temp1 temp2];
            
            TGOV1 = [TGOV1;temp];
            TGOV1_ID = [TGOV1_ID;ID];
            
        case 'IEEEST'
            n_ieeest = n_ieeest + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            temp1 = cell2num( [s1(5:end)],0);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end)],0);
            
            nextline = fgetl(fid);
            s3 = regexp(nextline,   '\s+','split');
            temp3 = cell2num( [s3(2:end)],0);
            
            nextline = fgetl(fid);
            s4 = regexp(nextline,   '\s+','split');
            temp4 = cell2num( [s4(2:end-1)],0);
            
            temp = [bus temp1 temp2 temp3 temp4];
            IEEEST = [IEEEST;temp];
            IEEEST_ID = [IEEEST_ID;ID];
            
        case 'REGCA1'
            n_regca = n_regca + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            temp1 = cell2num( [s1(5:end)],0);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end)],0);
            
            nextline = fgetl(fid);
            s3 = regexp(nextline,   '\s+','split');
            temp3 = cell2num( [s3(2:end)],0);
            
            nextline = fgetl(fid);
            s4 = regexp(nextline,   '\s+','split');
            temp4 = cell2num( [s4(2:end-1)],0);
            
            temp = [bus temp1 temp2 temp3 temp4];
            REGCA1 = [REGCA1;temp];
            REGCA1_ID = [REGCA1_ID;ID];
            
        case 'REECB1'
            n_reecb = n_reecb + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2(2:end)],0);
            
            nextline = fgetl(fid);
            s3 = regexp(nextline,   '\s+','split');
            temp3 = cell2num( [s3(2:end)],0);
            
            nextline = fgetl(fid);
            s4 = regexp(nextline,   '\s+','split');
            temp4 = cell2num( [s4(2:end)],0);
            
            nextline = fgetl(fid);
            s5 = regexp(nextline,   '\s+','split');
            temp5 = cell2num( [s5(2:end)],0);
            
            nextline = fgetl(fid);
            s6 = regexp(nextline,   '\s+','split');
            temp6 = cell2num( [s6(2:end)],0);
            
            nextline = fgetl(fid);
            s7 = regexp(nextline,   '\s+','split');
            temp7 = cell2num( [s7(2:end-1)],0);
            
            temp = [bus temp2 temp3 temp4 temp5 temp6 temp7];
            REECB1 = [REECB1;temp];
            REECB1_ID = [REECB1_ID;ID];
            
        case 'REPCA1'
            n_repca = n_repca + 1;
            bus = str2double(char(s1(2)));
            ID = s1(4);
            
            nextline = fgetl(fid);
            s2 = regexp(nextline,   '\s+','split');
            temp2 = cell2num( [s2([2 3 4 7 8 9])],0);
            branch_ID = s2(5);
%             branch_ID = char(s2(5));
%             branch_ID = branch_ID(2:end);
%             branch_ID = str2cell(convertCharsToStrings(branch_ID));
            
            nextline = fgetl(fid);
            s3 = regexp(nextline,   '\s+','split');
            temp3 = cell2num( [s3(2:end)],0);
            
            nextline = fgetl(fid);
            s4 = regexp(nextline,   '\s+','split');
            temp4 = cell2num( [s4(2:end)],0);
            
            nextline = fgetl(fid);
            s5 = regexp(nextline,   '\s+','split');
            temp5 = cell2num( [s5(2:end)],0);
            
            nextline = fgetl(fid);
            s6 = regexp(nextline,   '\s+','split');
            temp6 = cell2num( [s6(2:end)],0);
            
            nextline = fgetl(fid);
            s7 = regexp(nextline,   '\s+','split');
            temp7 = cell2num( [s7(2:end)],0);
            
            nextline = fgetl(fid);
            s8 = regexp(nextline,   '\s+','split');
            temp8 = cell2num( [s8(2:end-1)],0);
            
            temp = [bus temp2 temp3 temp4 temp5 temp6 temp7 temp8];
            REPCA1 = [REPCA1;temp];
            REPCA1_ID = [REPCA1_ID;ID];
            REPCA1_branch_ID = [REPCA1_branch_ID;branch_ID];
            
    end
end

fclose('all');

clearvars s1 s2 s3 s4 s5 s6 s7 s8 temp temp1 temp2 temp3 temp4 temp5 temp6 temp7 temp8 type nextline fid ans ID
save('dyr')

