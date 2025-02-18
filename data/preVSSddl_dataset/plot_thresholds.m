% plot thresholds


clc;
clear all;
%close all;

addpath('/Users/helenhu/Documents/MATLAB/Gaze_Package')

%% set up files

mydir = '/Users/helenhu/Documents/MATLAB/Correlation/crowdingFourBlocksData';
% mydir = '/Users/helenhu/Documents/MATLAB/Stationary_Dynamic_Flies_Codes';
d = dir(sprintf('%s/*.csv',mydir));    
files = {d.name};

    
numSubj = length(files);

%% calculate thresholds

eccentricity = 5;
numRepeats = 4;

subj_left = NaN(numSubj,numRepeats);
subj_right = NaN(numSubj,numRepeats);

for subj = 2:numSubj

    mainOutput = readtable([mydir filesep files{subj}]);

    thresholds_raw = mainOutput.questMeanAtEndOfTrialsLoop;
    rm = isnan(thresholds_raw);
    thresholds_raw(rm) = [];
    rm = [1 2];
    thresholds_raw(rm) = [];
    thresholds_raw = 10.^thresholds_raw;
    bouma_raw = thresholds_raw./eccentricity;

    subj_right(subj,:) = bouma_raw([1 3 5 7])';
    subj_left(subj,:) = bouma_raw([2 4 6 8])';



    % a couple of reasons to exclude a threshold:
    % 1. less than 30 trials were given to QUEST
    trialGivenBool = mainOutput.trialGivenToQuest;
    num_trialsNotGiven = sum(strcmp(trialGivenBool,'FALSE'));
    if num_trialsNotGiven > 5
        fprintf('TrialNotGiven: %d \n',subj)
    end

    % 2. SD of threshold is larger than 0.1
    sd_raw = mainOutput.questSDAtEndOfTrialsLoop;
    rm = isnan(sd_raw);
    sd_raw(rm) = [];
    if any(sd_raw > 0.1)
        fprintf('LargeSD: %d \n',subj)
    end
    

end

disp(subj_left)
disp(subj_right)

%%
function [] = plotHistLog(data,binEdges,xlimit,ylimit,titletxt,color)


    a = histogram(data,'BinEdges',binEdges);
    a.FaceColor = color; 
    a.EdgeColor = color;
    a.FaceAlpha = 0.6;
    a.EdgeAlpha = 0.6;
    ylim(ylimit)
    xlim(xlimit)
    set(gca,'XScale','log')
    set(gca,'FontSize',15)
    title(titletxt)
    ylabel('Frequency')


end
function [sem] = sem(data)
    n = length(data);
    std_dev = std(data);
    sem = std_dev / sqrt(n);
end

