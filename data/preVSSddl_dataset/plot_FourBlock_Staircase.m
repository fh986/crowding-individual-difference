
clc
clear all
close all
addpath('/Users/helenhu/Documents/MATLAB/Gaze_Package')

%% set up files

mydir = '/Users/helenhu/Documents/MATLAB/Correlation/crowdingFourBlocksData';
% mydir = '/Users/helenhu/Documents/MATLAB/Stationary_Dynamic_Flies_Codes';
d = dir(sprintf('%s/*.csv',mydir));    
files = {d.name};

    
numSubj = length(files);

%%

cmap = jet(6);
linestyle = {'-.','-'};


for subj = 2:numSubj%[2 3 5]

    t = readtable([mydir filesep files{subj}]);
    
    
    
    ct = 1; 
    
    figure;clf;
    for b = 2 : 5
    
        for s = 1:2
    
            blockOfInterest = b;
            staircase = s;
    
            t_block_cond = t(contains(t.staircaseName,sprintf('%i_%i',blockOfInterest,staircase)),:);
            h(ct)=plot(10.^t_block_cond.questMeanBeforeThisTrialResponse,'Color',cmap(b-1,:),'LineWidth',2,'LineStyle',linestyle(s));
            hold on
            bouma(ct) = 10.^t_block_cond.questMeanAtEndOfTrialsLoop(end)./ (t_block_cond.targetEccentricityXDeg(end-1));
            ct = ct + 1;
    
    
        end
    end
    xlabel('Trials')
    ylim([0 20])
    ylabel('Crowding Distance (deg)')
    set(gca,'FontSize',18)
    legend(h,{'Right1';'Left1';'Right2';'Left2';'Right3';'Left3';'Right4';'Left4';})

 

end

%%




%%
% blocks = unique(t.block);
% 
% cmap = jet(4);
% 
% ct = 1 
% 
% for b = 2 : 3
% 
%     for s = 1 : 2
% 
%         blockOfInterest = b;
%         staircase = s;
% 
%         t_block_cond = t(contains(t.staircaseName,sprintf('%i_%i',blockOfInterest,staircase)),:);
%         h(ct)=plot(10.^t_block_cond.questMeanBeforeThisTrialResponse,'Color',cmap(b-1,:))
%         hold on
%         bouma(ct) = 10.^t_block_cond.questMeanAtEndOfTrialsLoop(end)./ (t_block_cond.targetEccentricityXDeg(end-1));
%         ct = ct + 1
% 
% 
%     end
% end
% legend(h,{'Left S1';'Right S1';'Left S2';'Right S2'})






