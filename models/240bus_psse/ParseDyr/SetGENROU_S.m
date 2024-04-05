clc;clear;
load('dyr.mat');

fileID = fopen('setS.txt','w');

for i = 1:length(GENROU_ID)    
    fprintf(fileID,['psspy.change_plmod_con('  num2str(GENROU(i,1))  ',r"""'  char(GENROU_ID(i))  '""",r"""GENROU""",13,0.0)\n']);
    fprintf(fileID,['psspy.change_plmod_con('  num2str(GENROU(i,1))  ',r"""'  char(GENROU_ID(i))  '""",r"""GENROU""",14,0.0)\n']);
end

fclose(fileID);