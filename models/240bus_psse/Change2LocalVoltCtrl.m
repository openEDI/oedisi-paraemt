clc;clear;
table = table2array(readtable('LocalVoltCtrl.xlsx'));
fileID = fopen('LocalVoltCtrl.txt','w');
for i=1:length(table(:,1))
    fprintf(fileID,['psspy.plant_chng(' num2str(table(i,1)) ',0,[' num2str(table(i,4)) ',_f])']);
    fprintf(fileID,'\n');
end

