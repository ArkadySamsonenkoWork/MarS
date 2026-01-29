clear all;

% THE FIRST SPECTRA ISC-------------------------------
%{
Sys.S = [1];
Sys.g = [2.02 2.02 2.02]; 
Sys.D = [500, 100];
Sys.lw = [1.0, 1.0];
Sys.HStrain = 1;
Sys.initState = {[0.2 0.3 0.5],'xyz'};  % Tx, Ty, Tz


Exp.mwFreq = 9.8;  % GHz
Exp.Range = [300 400];  % mT
Exp.Harmonic = 0;  % no field modulation

spectra = pepper(Sys, Exp);
plot(spectra);
%}

% THE Second SPECTRA Recombination Triplets-------------------------------
%{
Sys.S = [1];
Sys.g = [2.02 2.02 2.02]; 
Sys.D = [500, 100];
Sys.lw = [1.0, 1.0];

Sys.initState = {[0 1 0],'eigen'};


Exp.mwFreq = 9.8;  % GHz
Exp.Range = [300 400];  % mT
Exp.Harmonic = 0;  % no field modulation

spectra = pepper(Sys, Exp);
plot(spectra);
%}

% THE Second SPECTRA Spin-Correlated Radical Pairs
Sys.S = [1/2 1/2];
Sys.g = [2.02, 2.03, 2.04; 2.03, 2.025, 2.025];

Sys.J = 400 ;  % MHz
Sys.dip = [-23.3333333333, -43.3333333333, 66.6666666667];  % MHz

Sys.lw = [1.0, 1.0];

Singlet = 1/sqrt(2)*[0; 1; -1; 0];  % singlet state in uncoupled basis, (|??>-|??>)/sqrt(2)
PSinglet = Singlet*Singlet'  % singlet projection operator
Sys.initState = {PSinglet,'uncoupled'};

Exp.Range = [300.0 400.0];
Exp.mwFreq = 9.8;
Exp.Harmonic = 0;

pepper(Sys,Exp);