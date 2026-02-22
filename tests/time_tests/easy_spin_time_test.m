%% EasySpin Spectrum Timing Functions
% This file contains functions for measuring EPR spectrum simulation time
% in EasySpin, matching the Python pipeline for benchmarking comparison.
%
% Functions:
%   1. createSampleParams() - Returns Sys and Exp structures (fill yourself)
%   2. timeSpectrumCalculation() - Measures spectrum computation time
%   3. timeSpectrumFullPipeline() - Measures full pipeline time (including sample creation)
%
% Author: [Your Name]
% Date: [Date]
% Requires: EasySpin toolbox (https://easyspin.org/)

%% ========================================================================
%% FUNCTION 1: Sample Parameters Creation
%% ========================================================================

function [Sys, Exp] = createSampleParamsExample()
    %% Spin System Parameters (FILL THIS SECTION)
    Sys = struct();
    
    % Electron spins (2 electrons, S=1/2 each)
    Sys.S = [0.5, 0.5];
    
    % g-tensors for each electron (anisotropic)
    Sys.g = [2.02, 2.04, 2.06;  % Electron 1
             2.02, 2.04, 2.06]; % Electron 2
    
    % Exchange coupling (in MHz)
    % 1 cm^-1 ? 29979.2458 MHz
    Sys.J = 29979.2458;  % 1 cm^-1 exchange
    
    % Dipolar coupling (D and E in MHz)
    Sys.D = [100, 10];  % D=100 MHz, E=10 MHz
    
    % Line broadening (in mT)
    Sys.lwpp = [0.5, 0.5];  % Gaussian and Lorentzian broadening
    
    % Optional: Hamiltonian strain
    % Sys.HStrain = 50;  % MHz
    
    %% Experimental Parameters
    Exp = struct();
    
    % Microwave frequency (GHz)
    Exp.mwFreq = 9.8;  % X-band
    
    % Magnetic field range (mT)
    Exp.Field = [300, 400];  % 300-400 mT
    
    % Number of field points
    Exp.nPoints = 1000;
    
    % Temperature (K)
    Exp.Temp = 298;
    
    % Powder simulation (orientational averaging)
    Exp.Powder = true;
    
    % Optional: Number of crystallite orientations
    % Exp.CrystalSymmetry = 'auto';
    % Exp.Ordering = 'Z';
    
end


%% ========================================================================
%% FUNCTION 2: Spectrum Computation Time (Excludes Sample Creation)
%% ========================================================================

function [meanTime, stdTime, allTimes] = timeSpectrumCalculation(Sys, Exp, nWarmup, nIterations)

    %% Warmup iterations (discarded from timing)
    for i = 1:nWarmup
        spectrum = pepper(Sys, Exp);
        clear spectrum;  % Free memory
    end
    
    %% Timed iterations
    allTimes = zeros(nIterations, 1);
    
    for i = 1:nIterations
        tic;
        spectrum = pepper(Sys, Exp);
        allTimes(i) = toc * 1000;  % Convert to milliseconds
        clear spectrum;  % Free memory
    end
    
    %% Statistics
    meanTime = mean(allTimes);
    stdTime = std(allTimes);
    
end


