Sys.S = [1/2 3/2];
Sys.g = [2.02 2.02 2.02; 2.02 2.07 2.3]; 
Sys.D = [0 0; unitconvert(-10, 'cm^-1->MHz'), unitconvert(-2, 'cm^-1->MHz')];
Sys.J = unitconvert(-5, 'cm^-1->MHz');
Sys.lw = [20, 100];
Sys.HStrain = 100000;


Exp.Temperature = 5;
Exp.mwRange = [0 3000];
Exp.Harmonic = 0;
Exp.Field = 10 * 1e3;
Exp.nPoints = 4000;

Opt.GridSize = 91; 

pol='circular+' ;
k = [0, pi * 0 / 180];
Exp.mwMode = {k, pol};

pepper(Sys, Exp);