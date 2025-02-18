
clc
clear all
close all
t = readtable('GreatSilverCarrot137_correlationfull1_0001_2024-12-03_01h17.34.480.csv');


%%

plotTask = 'rsvp';

if strcmp(plotTask,'crowding')
    staircaseName_left = '3_2';
    staircaseName_right = '3_1';
    numTrials = 35;
elseif strcmp(plotTask,'letter_acuity')
    staircaseName_left = '1_2';
    staircaseName_right = '1_1'; 
    numTrials = 35;
elseif strcmp(plotTask,'rsvp')
    staircaseName = '4_1';
    numTrials = 24;
end



%%

figure;clf;
t_block_cond_left = t(contains(t.staircaseName,staircaseName_left),:);
t_block_cond_right = t(contains(t.staircaseName,staircaseName_right),:);

hold on;
plot(1:numTrials,10.^t_block_cond_left.questMeanBeforeThisTrialResponse(1:numTrials),'r-','LineWidth',2)
plot(1:numTrials,10.^t_block_cond_right.questMeanBeforeThisTrialResponse(1:numTrials),'g--','LineWidth',2)
legend('Left','Right')
set(gca,'FontSize',18)
xlabel('Trials')
ylabel('Spacing (deg)')


%%
figure;clf;
t_block_cond = t(contains(t.staircaseName,staircaseName),:);

hold on;
plot(1:numTrials,10.^t_block_cond.questMeanBeforeThisTrialResponse(1:numTrials),'r-','LineWidth',2)
legend('RSVP')
set(gca,'FontSize',18)
xlabel('Trials')
ylabel('Reading rate (s)')
xlim([1 24])
%         t_block_cond = t(contains(t.staircaseName,sprintf('%i_%i',blockOfInterest,staircase)),:);
%         h(ct)=plot(10.^t_block_cond.questMeanBeforeThisTrialResponse,'Color',cmap(b-1,:))
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






