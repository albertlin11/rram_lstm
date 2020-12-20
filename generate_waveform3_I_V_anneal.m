%function Vseq=generate_waveform(Vmin,Vmax)
close all;
clear all;
Vmin=-1.49;Vmax=1.49;
Vstart=0;
dV=0.3;

%an_te=300;
for an_te=[0,300,400,500]
    for rndseed=2:3  % rndseed=1 5e4, rndsee=2,3, 1e4
        %%%%%%%%%% random walk
        %rndseed=5;
        close all;
        Vseq=[Vstart];
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
        if an_te==0
            [state_seq,Iseq]=rram_IV_0(Vseq);
        elseif an_te==300
            [state_seq,Iseq]=rram_IV_300(Vseq);
        elseif an_te==400
            [state_seq,Iseq]=rram_IV_400(Vseq);
        elseif an_te==500
            [state_seq,Iseq]=rram_IV_500(Vseq);
        end
        
        plot(Iseq);
        figure;
        plot(Vseq)
        figure;
        plot(state_seq)
        
        his_len=5;
        
        ds=[];
        ds(1,:)=zeros(1,16);
        jj=2;
        for ii=his_len:length(Vseq)-1
            temp=[];
            for kk=1:his_len
                temp=[temp an_te Vseq(ii-his_len+kk+1) Iseq(ii-his_len+kk)  ];
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
        
        %ds=[zeros(size(ds,1),his_len)+an_te    ds ];
        csvwrite(['CSV_data/Anneal/ds' num2str(an_te) '_rram_rndseed' num2str(rndseed) '.csv'],ds);
    end
end