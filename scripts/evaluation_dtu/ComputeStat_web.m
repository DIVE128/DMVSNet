clear all
close all
format compact
clc

% script to calculate the statistics for each scan given this will currently only run if distances have been measured
% for all included scans (UsedSets)

% modify the path to evaluate your models
% dataPath='<path to datasets>';
dataPath='/data2/yexinyi/datasets/MVS/SampleSet/MVSData';
% resultsPath='be similar to that in BaseEvalMain_web.m';
resultsPath='/data2/yexinyi/code/DMVSNet/outputs_dtu/DMVSNet/pcd';

Force=0

MaxDist=20; %outlier thresshold of 20 mm

time=clock;

method_string='mvsnet';
light_string='l3'; %'l7'; l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end

% get sets used in evaluation
UsedSets=[1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118];

totalStatName=[resultsPath 'TotalStat_' method_string eval_string '.mat']
if (~exist(totalStatName,'file')||Force==1)
    nStat=length(UsedSets);

    BaseStat.nStl=zeros(1,nStat);
    BaseStat.nData=zeros(1,nStat);
    BaseStat.MeanStl=zeros(1,nStat);
    BaseStat.MeanData=zeros(1,nStat);
    BaseStat.VarStl=zeros(1,nStat);
    BaseStat.VarData=zeros(1,nStat);
    BaseStat.MedStl=zeros(1,nStat);
    BaseStat.MedData=zeros(1,nStat);

    for cStat=1:length(UsedSets) %Data set number
        
        currentSet=UsedSets(cStat);
        
        %input results name
        EvalName=[resultsPath method_string eval_string num2str(currentSet) '.mat']
        
        disp(EvalName);
        load(EvalName);
        
        Dstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
        Dstl=Dstl(Dstl<MaxDist); % discard outliers
        
        Ddata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
        Ddata=Ddata(Ddata<MaxDist); % discard outliers
        
        BaseStat.nStl(cStat)=length(Dstl);
        BaseStat.nData(cStat)=length(Ddata);
        
        BaseStat.MeanStl(cStat)=mean(Dstl);
        BaseStat.MeanData(cStat)=mean(Ddata);
        
        BaseStat.VarStl(cStat)=var(Dstl);
        BaseStat.VarData(cStat)=var(Ddata);
        
        BaseStat.MedStl(cStat)=median(Dstl);
        BaseStat.MedData(cStat)=median(Ddata);
        
        disp("acc");
        disp(mean(Ddata));
        disp("comp");
        disp(mean(Dstl));
        time=clock;
    end
    % totalStatName=[resultsPath 'TotalStat_' method_string eval_string '.mat']
    save(totalStatName,'BaseStat','time','MaxDist');
end

load(totalStatName)
disp(BaseStat);
disp("mean acc")
disp(mean(BaseStat.MeanData));
disp("mean comp")
disp(mean(BaseStat.MeanStl));

fid=fopen([resultsPath 'TotalStat_' method_string eval_string '.txt'],'w');
acc=mean(BaseStat.MeanData);
comp=mean(BaseStat.MeanStl);

overall=(acc+comp)/2;
fprintf(fid,'mean acc:%f\t',acc);
fprintf(fid,'mean comp:%f\t',comp);
fprintf(fid,'mean overall:%f\r\n',overall);


fprintf(fid,'scans\tacc  \tcmop  \r\n',UsedSets);
for cStat=1:length(UsedSets)
    fprintf(fid,'scan%d\t',UsedSets(cStat));
    fprintf(fid,'%.4f\t',BaseStat.MeanData(cStat));
    fprintf(fid,'%.4f\r\n',BaseStat.MeanStl(cStat));
end


fclose(fid);






