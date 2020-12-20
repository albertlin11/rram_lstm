function [state, I]=rram_IV_300(V)
%close all;
%clear all;
%V=linspace(-1.5,1.5,100);
%state=1; % 1 means HRS

a=[0,1.77E-09
    0.015,1.25E-06
    0.03,2.50E-06
    0.045,3.76E-06
    0.06,5.04E-06
    0.075,6.35E-06
    0.09,7.70E-06
    0.105,9.14E-06
    0.12,1.05E-05
    0.135,1.19E-05
    0.15,1.30E-05
    0.165,1.44E-05
    0.18,1.61E-05
    0.195,1.77E-05
    0.21,1.96E-05
    0.225,2.09E-05
    0.24,2.26E-05
    0.255,2.43E-05
    0.27,2.70E-05
    0.285,2.85E-05
    0.3,2.93E-05
    0.315,3.06E-05
    0.33,3.20E-05
    0.345,3.39E-05
    0.36,3.72E-05
    0.375,3.94E-05
    0.39,3.98E-05
    0.405,4.25E-05
    0.42,4.61E-05
    0.435,4.76E-05
    0.45,4.71E-05
    0.465,4.80E-05
    0.48,5.04E-05
    0.495,5.20E-05
    0.51,5.26E-05
    0.525,5.36E-05
    0.54,5.20E-05
    0.555,5.23E-05
    0.57,5.40E-05
    0.585,5.73E-05
    0.6,5.61E-05
    0.615,6.03E-05
    0.63,6.37E-05
    0.645,7.92E-05
    0.66,9.84E-05
    0.675,0.000104996
    0.69,0.000101677
    0.705,0.000106559
    0.72,0.000127319
    0.735,0.000133673
    0.75,0.000153499
    0.765,0.000137737
    0.78,0.000130776
    0.795,0.000138958
    0.81,0.00015636
    0.825,0.000188674
    0.84,0.000212834
    0.855,0.000261016
    0.87,0.000277389
    0.885,0.000289537
    0.9,0.000315062
    0.915,0.000344917
    0.93,0.000399068
    0.945,0.000431778
    0.96,0.000999993
    0.975,0.000999992
    0.99,0.000999991
    1.005,0.000999991
    1.02,0.000999992
    1.035,0.000999991
    1.05,0.000999992
    1.065,0.000999991
    1.08,0.000999992
    1.095,0.000999992
    1.11,0.000999992
    1.125,0.000999991
    1.14,0.000999992
    1.155,0.000999991
    1.17,0.000999992
    1.185,0.000999991
    1.2,0.000999991
    1.215,0.000999992
    1.23,0.000999992
    1.245,0.000999992
    1.26,0.000999991
    1.275,0.000999992
    1.29,0.000999993
    1.305,0.000999993
    1.32,0.000999992
    1.335,0.000999992
    1.35,0.000999992
    1.365,0.000999992
    1.38,0.000999992
    1.395,0.000999993
    1.41,0.000999992
    1.425,0.000999992
    1.44,0.000999992
    1.455,0.000999991
    1.47,0.000999992
    1.485,0.000999991
    1.5,0.000999991];
b=[1.5,0.000999992
    1.485,0.000999992
    1.47,0.000999991
    1.455,0.000999993
    1.44,0.000999993
    1.425,0.000999992
    1.41,0.000999992
    1.395,0.000999991
    1.38,0.000999992
    1.365,0.000999992
    1.35,0.000999992
    1.335,0.000999993
    1.32,0.000999992
    1.305,0.000999991
    1.29,0.000999992
    1.275,0.00099999
    1.26,0.00099999
    1.245,0.000999991
    1.23,0.000999992
    1.215,0.000999991
    1.2,0.000999991
    1.185,0.000999991
    1.17,0.000999991
    1.155,0.000999991
    1.14,0.000999992
    1.125,0.000999992
    1.11,0.000999992
    1.095,0.000999991
    1.08,0.000999991
    1.065,0.000999991
    1.05,0.000999992
    1.035,0.000999992
    1.02,0.000999992
    1.005,0.000999991
    0.99,0.000999991
    0.975,0.000999992
    0.96,0.000999992
    0.945,0.000999992
    0.93,0.000999992
    0.915,0.000999991
    0.9,0.000999992
    0.885,0.000999991
    0.87,0.000999993
    0.855,0.000999992
    0.84,0.000999992
    0.825,0.000999993
    0.81,0.000999991
    0.795,0.000999992
    0.78,0.000999992
    0.765,0.000999991
    0.75,0.000999992
    0.735,0.000999992
    0.72,0.000977
    0.705,0.000956496
    0.69,0.000930406
    0.675,0.000907311
    0.66,0.000881925
    0.645,0.000858986
    0.63,0.000835178
    0.615,0.000815365
    0.6,0.000792323
    0.585,0.000769537
    0.57,0.000748283
    0.555,0.000727132
    0.54,0.000705374
    0.525,0.000683095
    0.51,0.00066245
    0.495,0.000639869
    0.48,0.000619516
    0.465,0.00059862
    0.45,0.000577594
    0.435,0.000556441
    0.42,0.000536009
    0.405,0.000514935
    0.39,0.000495115
    0.375,0.000473721
    0.36,0.00045396
    0.345,0.000434154
    0.33,0.000415074
    0.315,0.000394127
    0.3,0.000374068
    0.285,0.000355009
    0.27,0.000335894
    0.255,0.000316546
    0.24,0.000297413
    0.225,0.000278103
    0.21,0.000259331
    0.195,0.000239774
    0.18,0.000221101
    0.165,0.000202424
    0.15,0.000183848
    0.135,0.000165187
    0.12,0.000146743
    0.105,0.000128129
    0.09,0.000109485
    0.075,9.12E-05
    0.06,7.29E-05
    0.045,5.47E-05
    0.03,3.64E-05
    0.015,1.82E-05
    0,4.86E-08
    -0.015,-1.81E-05
    -0.03,-3.63E-05
    -0.045,-5.45E-05
    -0.06,-7.25E-05
    -0.075,-9.07E-05
    -0.09,-0.00010886
    -0.105,-0.000127281
    -0.12,-0.000145841
    -0.135,-0.000164394
    -0.15,-0.000182821
    -0.165,-0.000201655
    -0.18,-0.000220477
    -0.195,-0.000239431
    -0.21,-0.000258664
    -0.225,-0.00027723
    -0.24,-0.00029587
    -0.255,-0.000315202
    -0.27,-0.000333569
    -0.285,-0.000353244
    -0.3,-0.000372511
    -0.315,-0.00039239
    -0.33,-0.000411631
    -0.345,-0.000431521
    -0.36,-0.000450669
    -0.375,-0.000471216
    -0.39,-0.000490924
    -0.405,-0.000511796
    -0.42,-0.000532168
    -0.435,-0.000552905
    -0.45,-0.000571528
    -0.465,-0.000592989
    -0.48,-0.000612183
    -0.495,-0.000633808
    -0.51,-0.000651807
    -0.525,-0.000674422
    -0.54,-0.000696816
    -0.555,-0.000719457
    -0.57,-0.000741655
    -0.585,-0.000763115
    -0.6,-0.000786771
    -0.615,-0.000805985
    -0.63,-0.000822999
    -0.645,-0.000846008
    -0.66,-0.00086186
    -0.675,-0.000876954
    -0.69,-0.0008979
    -0.705,-0.000919163
    -0.72,-0.000929619
    -0.735,-0.000951223
    -0.75,-0.000954576
    -0.765,-0.000950896
    -0.78,-0.000936428
    -0.795,-0.000909923
    -0.81,-0.000865588
    -0.825,-0.00085482
    -0.84,-0.000868274
    -0.855,-0.000868308
    -0.87,-0.000858879
    -0.885,-0.0008674
    -0.9,-0.000863277
    -0.915,-0.000868623
    -0.93,-0.000849653
    -0.945,-0.000857229
    -0.96,-0.000740004
    -0.975,-0.000663759
    -0.99,-0.000611634
    -1.005,-0.000483346
    -1.02,-0.000444544
    -1.035,-0.000425425
    -1.05,-0.000412494
    -1.065,-0.000385068
    -1.08,-0.000403527
    -1.095,-0.000426862
    -1.11,-0.000443883
    -1.125,-0.000436424
    -1.14,-0.000431519
    -1.155,-0.000438768
    -1.17,-0.000428693
    -1.185,-0.000398278
    -1.2,-0.000406323
    -1.215,-0.000386104
    -1.23,-0.000414591
    -1.245,-0.000425494
    -1.26,-0.000426304
    -1.275,-0.000452139
    -1.29,-0.000449389
    -1.305,-0.000441322
    -1.32,-0.000422692
    -1.335,-0.000432313
    -1.35,-0.000449264
    -1.365,-0.000482676
    -1.38,-0.000474095
    -1.395,-0.000520022
    -1.41,-0.0005292
    -1.425,-0.00053593
    -1.44,-0.000607373
    -1.455,-0.000511989
    -1.47,-0.000505606
    -1.485,-0.000531817
    -1.5,-0.000528032];
c=[-1.5,-0.000535153
    -1.485,-0.000496172
    -1.47,-0.000479232
    -1.455,-0.000460636
    -1.44,-0.000444703
    -1.425,-0.000428412
    -1.41,-0.000415865
    -1.395,-0.00039935
    -1.38,-0.000382547
    -1.365,-0.000373478
    -1.35,-0.000361458
    -1.335,-0.000352418
    -1.32,-0.000341928
    -1.305,-0.000329613
    -1.29,-0.000319731
    -1.275,-0.000309417
    -1.26,-0.000302568
    -1.245,-0.000291904
    -1.23,-0.000283655
    -1.215,-0.000275124
    -1.2,-0.000267732
    -1.185,-0.000259817
    -1.17,-0.00025378
    -1.155,-0.000246267
    -1.14,-0.000238831
    -1.125,-0.000232194
    -1.11,-0.000225108
    -1.095,-0.000219503
    -1.08,-0.000213475
    -1.065,-0.000207704
    -1.05,-0.000201778
    -1.035,-0.000196278
    -1.02,-0.00019105
    -1.005,-0.000185609
    -0.99,-0.00018117
    -0.975,-0.000176929
    -0.96,-0.000171949
    -0.945,-0.000167359
    -0.93,-0.000162636
    -0.915,-0.000158316
    -0.9,-0.000154107
    -0.885,-0.000149852
    -0.87,-0.000145247
    -0.855,-0.000141264
    -0.84,-0.000137293
    -0.825,-0.000133322
    -0.81,-0.000129459
    -0.795,-0.000125745
    -0.78,-0.00012213
    -0.765,-0.000118502
    -0.75,-0.000114828
    -0.735,-0.000111254
    -0.72,-0.000107976
    -0.705,-0.000104586
    -0.69,-0.000101213
    -0.675,-9.82E-05
    -0.66,-9.51E-05
    -0.645,-9.19E-05
    -0.63,-8.89E-05
    -0.615,-8.61E-05
    -0.6,-8.16E-05
    -0.585,-7.88E-05
    -0.57,-7.58E-05
    -0.555,-7.30E-05
    -0.54,-7.04E-05
    -0.525,-6.76E-05
    -0.51,-6.49E-05
    -0.495,-6.22E-05
    -0.48,-5.95E-05
    -0.465,-5.66E-05
    -0.45,-5.38E-05
    -0.435,-5.15E-05
    -0.42,-4.91E-05
    -0.405,-4.68E-05
    -0.39,-4.51E-05
    -0.375,-4.35E-05
    -0.36,-4.12E-05
    -0.345,-3.86E-05
    -0.33,-3.63E-05
    -0.315,-3.38E-05
    -0.3,-3.21E-05
    -0.285,-3.01E-05
    -0.27,-2.82E-05
    -0.255,-2.64E-05
    -0.24,-2.45E-05
    -0.225,-2.26E-05
    -0.21,-2.08E-05
    -0.195,-1.89E-05
    -0.18,-1.72E-05
    -0.165,-1.55E-05
    -0.15,-1.37E-05
    -0.135,-1.22E-05
    -0.12,-1.06E-05
    -0.105,-9.06E-06
    -0.09,-7.50E-06
    -0.075,-6.05E-06
    -0.06,-4.77E-06
    -0.045,-3.56E-06
    -0.03,-2.37E-06
    -0.015,-1.12E-06];
HRS=[c; a];
LRS=flipud(b);
tt=[a;b;c];

plot(tt(:,1),tt(:,2));
Vset=0.96;
Vreset=-1.065;







for ii=1:length(V)
    
    if V(ii)>Vset
        state(ii)=0;
    elseif V(ii)< Vreset
        state(ii)=1;
    else
        if ii==1
            state(ii)=1;
        else
            state(ii)=state(ii-1);
        end
    end
    
    if state(ii)==1 % HRS
        I(ii)=interp1(HRS(:,1),HRS(:,2),V(ii));
    elseif state(ii)==0 % LRS
        I(ii)=interp1(LRS(:,1),LRS(:,2),V(ii));
    else
    end
    
end
