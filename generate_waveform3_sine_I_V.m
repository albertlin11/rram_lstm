%function Vseq=generate_waveform(Vmin,Vmax)

clear all;
Vmin=-1.49;Vmax=1.49;
%Vstart=0;
%dV=0.5;
%Vseq=[Vstart];
for phi=[-1,0,1,2,3]
    close all;
    %%% sine
    A=(-Vmin+Vmax)/2;
    offset=(Vmin+Vmax)/2;
    %phi=2;
    t=linspace(0,100,10e2);
    Vseq=A*sin(t+phi)+offset;
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
    % from sine to temperautre dependent anneal sine, change 3 places
    %ds=[zeros(size(ds,1),his_len)+0    ds ];
    csvwrite(['CSV_data/ds_rram_sine' num2str(phi) '.csv'],ds)
    %csvwrite(['CSV_data/Anneal/ds0_rram_sine' num2str(phi) '.csv'],ds)
end
