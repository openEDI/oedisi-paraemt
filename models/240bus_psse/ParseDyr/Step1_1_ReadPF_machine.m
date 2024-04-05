%% Read PSSE .dyr file and write data into CSV/EXCEL file
clc;clear;

%% path and file name
filePF = 'PF_machine.xlsx';



%% read file
data_t = readtable(filePF);
data_t(end,:) = [];

PF_gen_status = table2array(data_t(:,4));
on_idx = find(PF_gen_status==1);

PF_gen_bus = table2array(data_t(on_idx,1));
PF_gen_id = table2cell(data_t(on_idx,3));
PF_gen_Ra = table2array(data_t(on_idx,6));
PF_gen_Xs = table2array(data_t(on_idx,7));
PF_gen_Mbase = table2array(data_t(on_idx,22));

re_idx = [];
for i = 1:length(PF_gen_bus)
    if (strcmp(char(PF_gen_id(i)),'S'))|(strcmp(char(PF_gen_id(i)),'W'))|(strcmp(char(PF_gen_id(i)),'DP'))|(strcmp(char(PF_gen_id(i)),'NW'))|(strcmp(char(PF_gen_id(i)),'SW'))
        re_idx = [re_idx;i];
    end
end


PF_re_bus = PF_gen_bus(re_idx);
PF_re_id = PF_gen_id(re_idx);
PF_re_status = PF_gen_status(re_idx);
PF_re_Ra = PF_gen_Ra(re_idx);
PF_re_Xs = PF_gen_Xs(re_idx);
PF_re_Mbase = PF_gen_Mbase(re_idx);

PF_gen_bus(re_idx) = [];
PF_gen_id(re_idx) = [];
PF_gen_status(re_idx) = [];
PF_gen_Ra(re_idx) = [];
PF_gen_Xs(re_idx) = [];
PF_gen_Mbase(re_idx) = [];
save('raw')

