clc;clear;
load('dyr.mat');

fileID = fopen('AllGov_Gast.txt','w');

for i = 1:length(GENROU_ID)    
    fprintf(fileID,['psspy.plmod_remove('  num2str(GENROU(i,1))  ',r"""'  char(GENROU_ID(i))  '""",7)\n']);
    fprintf(fileID,['psspy.add_plant_model('  num2str(GENROU(i,1))  ',r"""'  char(GENROU_ID(i))  '""",7,r"""GAST""",0,"",0,[],[],9,[0.33,0.5,0.5,3.0,1.0,2.0,1.0,0.0,0.0])\n']);
end

fclose(fileID);