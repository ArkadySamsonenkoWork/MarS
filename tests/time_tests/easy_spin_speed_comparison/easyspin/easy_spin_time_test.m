clear;

out = load('D:\ITC\РНФ_Курганский_2024\pythonProject\MarS\tests\time_tests\easy_spin_speed_comparison\easyspin\out_4.mat');
T = readtable('D:\ITC\РНФ_Курганский_2024\pythonProject\MarS\tests\time_tests\easy_spin_speed_comparison\easyspin\spectrum_4.csv');

Opt.Threshold = 1e-4;
Opt.GridSize = [20 0]; 

field_python = T.Var1 * 1000;
spec_python = T.Var2;

Sys = out.Sys;
Exp = out.Exp;
spec_easy_spin = pepper(Sys, Exp, Opt);
spec_easy_spin = spec_easy_spin / max(spec_easy_spin);
spec_python = spec_python / max(spec_python);
plot(field_python, spec_easy_spin, field_python, spec_python);
legend(["EasySpin", "Python"])

fields = linspace(Exp.Range(1), Exp.Range(2), Exp.nPoints) / 1e3;
writematrix([fields; spec_easy_spin], "spec_easy_spin_4.csv");

%%[mean, std, all] = timeSpectrumCalculation(Sys, Exp, 2, 5);

%% ========================================================================
%% FUNCTION 2: Spectrum Computation Time (Excludes Sample Creation)
%% ========================================================================
function [meanTime, stdTime, allTimes] = timeSpectrumCalculation(Sys, Exp, nWarmup, nIterations)
    %% Warmup iterations (discarded from timing)
    Opt.Threshold = 1e-4;
    Opt.GridSize = [20 0]; 

    for i = 1:nWarmup
        spectrum = pepper(Sys, Exp, Opt);
        clear spectrum;  % Free memory
    end
    
    %% Timed iterations
    allTimes = zeros(nIterations, 1);
    
    for i = 1:nIterations
        tic;
        spectrum = pepper(Sys, Exp, Opt);
        allTimes(i) = toc * 1000;  % Convert to milliseconds
        clear spectrum;  % Free memory
    end
    
    %% Statistics
    meanTime = mean(allTimes);
    stdTime = std(allTimes);
end