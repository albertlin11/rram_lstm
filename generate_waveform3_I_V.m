%function Vseq=generate_waveform(Vmin,Vmax)
close all;
clear all;
Vmin=-1.49;Vmax=1.49;
Vstart=0;
dV=0.3;

for rndseed=1:5
    close all;
    Vseq=[Vstart];
    
    %%%%%%%%%% random walk
    %rndseed=5;
    
    rng(rndseed);
    for ii=1:1e4
        Vnew=Vseq(end)+dV*(rand(1)*2-1);
        if Vnew>Vmax
            Vnew=Vmax;
        elseif Vnew < Vmin
            Vnew=Vmin;
        end
        Vseq=[Vseq Vnew];
    end
    
    
    plot(Vseq);
    
    %%%%%%%%%%%%%%% evaluate rram current
    
    [state_seq,Iseq]=rram_IV_0(Vseq);
    
    plot(Iseq);
    figure;
    plot(Vseq)
    figure;
    plot(state_seq)
    
    his_len=5;
    
    ds=[];
    ds(1,:)=zeros(1,11);
    
    jj=2;
    for ii=his_len:length(Vseq)-1
        temp=[];
        for kk=1:his_len
            temp=[temp Vseq(ii-his_len+kk+1) Iseq(ii-his_len+kk)  ];
        end
        %for kk=1:his_len
        %  temp=[temp state_seq(ii-his_len+kk-1)];
        %end
        ds(jj,:)=[temp Iseq(ii+1) ];
        jj=jj+1;
    end
    
    if sum(sum(isnan(ds)))~=0
        disp('NAN');
    end
    %ds=[zeros(size(ds,1)+0,1) ds ];
    csvwrite(['CSV_data/ds_rram_test_rnseed' num2str(rndseed)  '.csv'],ds);
end