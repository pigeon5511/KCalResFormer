function generate_formal(stage,worker,workers,out,dataRoot)
% Isolated source dataset. Independent channel/data/noise streams; atomic shards.
if nargin<2, worker=1; end
if nargin<3, workers=1; end
if nargin<4, out=fileparts(mfilename('fullpath')); end
p=jsondecode(fileread(fullfile(out,'protocol.json')));
if nargin<5, dataRoot=p.data_root; end
carrier=nrCarrierConfig('NSizeGrid',10,'SubcarrierSpacing',15,'NCellID',0);
carrier.NSlot=0; fs=p.carrier.sample_rate_hz;
oi=nrOFDMInfo(carrier,'SampleRate',fs);
times=cumsum([0;double(oi.SymbolLengths(1:13)).'])/fs;
uniongrid=complex(zeros(120,14)); masks=false(120,14,4);
expected={[2],[2 11],[2 7 11],[2 5 8 11]};
for j=1:4
    pdsch=nrPDSCHConfig; pdsch.PRBSet=0:9; pdsch.SymbolAllocation=[0 14];
    pdsch.MappingType='A'; pdsch.NumLayers=1;
    pdsch.DMRS.DMRSConfigurationType=2; pdsch.DMRS.DMRSTypeAPosition=2;
    pdsch.DMRS.DMRSLength=1; pdsch.DMRS.DMRSAdditionalPosition=j-1;
    pdsch.DMRS.DMRSPortSet=0; pdsch.DMRS.NumCDMGroupsWithoutData=1;
    ind=nrPDSCHDMRSIndices(carrier,pdsch); sym=nrPDSCHDMRS(carrier,pdsch);
    mask=false(120,14); mask(ind)=true;
    assert(nnz(mask)==40*j && isequal(find(any(mask,1))-1,expected{j}));
    overlap=uniongrid(ind)~=0; assert(all(abs(uniongrid(ind(overlap))-sym(overlap))<1e-12));
    uniongrid(ind)=sym; masks(:,:,j)=mask;
end
data_mask=uniongrid==0; assert(nnz(data_mask)==1480);
if strcmp(stage,'smoke')
    g=struct('pilot_masks',masks,'pilot_symbols',single(uniongrid), ...
        'data_mask',data_mask,'subcarrier_frequencies_hz',(0:119)'*15000, ...
        'symbol_times_s',times,'layout_names','ABCD','source_layout_indices',[1 3]);
    save(fullfile(out,'geometry.mat'),'-struct','g','-v6');
    seed=p.smoke_seed_base+1;
    a=realization(carrier,uniongrid,fs,50,0,100,seed);
    b=realization(carrier,uniongrid,fs,50,0,100,seed);
    c=realization(carrier,uniongrid,fs,50,0,100,seed+1);
    assert(isequal(a.H_true,b.H_true)&&isequal(a.Y,b.Y)&&isequal(a.X,b.X));
    assert(norm(a.H_true(:)-c.H_true(:))>1e-3);
    static_nmse=sum(abs(a.Y_noiseless(:)-a.H_true(:).*a.X(:)).^2)/sum(abs(a.H_true(:)).^2);
    assert(static_nmse<1e-4,'Static coordinate/alignment smoke failed');
    eq=a.Y_noiseless./a.H_true;
    bh=cat(3,real(eq)<0,imag(eq)<0);
    bitmask=repmat(data_mask,1,1,2);
    assert(nnz(bh(bitmask)~=a.tx_bits(bitmask))==0,'Static ideal-channel BER failed');
    dynamic=realization(carrier,uniongrid,fs,300,1000,20,seed+2);
    assert(all(isfinite(dynamic.H_true(:)))&&all(isfinite(dynamic.Y(:))));
    a.snr_db=100; a.delay_spread_ns=50; a.max_doppler_hz=0;
    save(fullfile(out,'smoke_static.mat'),'-struct','a','-v6');
    toolboxInfo=ver('5g');
    t=tic;
    for i=1:100
        realization(carrier,uniongrid,fs,25+25*mod(i,12),50+50*mod(i,20),5*mod(i,6),seed+100+i);
    end
    elapsed=toc(t);
    writejson(fullfile(out,'smoke.json'),struct('status','PASS','matlab_version',version,'fiveg_toolbox_version',toolboxInfo.Version, ...
        'static_noiseless_mismatch_nmse',double(static_nmse),'static_ideal_channel_bit_errors',0, ...
        'repeat_bitwise_identical',true,'different_seed_distinct_channel',true, ...
        'benchmark_channels',100,'benchmark_seconds',elapsed,'seconds_per_channel',elapsed/100, ...
        'finite_high_doppler',true,'sample_rate_hz',fs,'pilot_counts',[40 80 120 160], ...
        'data_RE',nnz(data_mask),'bits_per_channel',2*nnz(data_mask)));
    fprintf('SMOKE PASS: %.5fs/channel, 2960 bits/channel\n',elapsed/100);
    return;
end
freeze=jsondecode(fileread(fullfile(out,'freeze.json')));
assert(strcmp(filehash(fullfile(out,'protocol.json')),freeze.protocol_sha256));
assert(strcmp(filehash([mfilename('fullpath') '.m']),freeze.generator_sha256));
assert(strcmp(stage,'source')||strcmp(stage,'replay'));
splits={'calibration','validation','train'}; job=0; count=0; clock=tic;
for si=1:numel(splits)
    split=splits{si}; spec=p.splits.(split); schedule=load(fullfile(out,['schedule_' split '.mat']));
    for shard=1:ceil(spec.channels/p.shard_size)
        job=job+1;
        if strcmp(stage,'source') && mod(job-1,workers)~=worker-1, continue; end
        lo=(shard-1)*p.shard_size+1; hi=min(shard*p.shard_size,spec.channels);
        destination=fullfile(dataRoot,split,sprintf('shard_%04d.mat',shard));
        marker=[destination '.complete.json'];
        if strcmp(stage,'replay')
            if shard~=1, continue; end
            m=load(destination); k=1;
            a=realization(carrier,uniongrid,fs,double(m.delay_spread_ns(k)),double(m.max_doppler_hz(k)),double(m.snr_db(k)),double(m.channel_seed(k)));
            assert(isequal(a.H_true,m.H_true(:,:,k))&&isequal(a.Y,m.Y(:,:,k))&&isequal(a.X,m.X(:,:,k))&&isequal(a.tx_bits,m.tx_bits(:,:,:,k)));
            fprintf('REPLAY PASS %s channel %d\n',split,m.channel_id(k));
            continue;
        end
        if isfile(marker)
            done=jsondecode(fileread(marker));
            assert(isfile(destination)&&strcmp(done.protocol_sha256,freeze.protocol_sha256));
            assert(strcmp(done.sha256,filehash(destination)),'Resume shard checksum mismatch');
            assert(done.channels==hi-lo+1&&done.first_channel_id==lo&&done.last_channel_id==hi);
            fprintf('VALIDATED SKIP %s shard %d\n',split,shard);
            continue;
        end
        assert(~isfile(destination),'Finalized shard missing completion marker; audit orphan before recovery.');
        n=hi-lo+1; s=struct;
        s.H_true=complex(zeros(120,14,n,'single')); s.X=s.H_true; s.Y=s.H_true; s.Y_noiseless=s.H_true;
        s.tx_bits=zeros(120,14,2,n,'uint8');
        s.target_energy=zeros(1,n); s.noiseless_mismatch_nmse=zeros(1,n);
        s.timing_offset=zeros(1,n,'uint16');
        names={'channel_id','condition_index','snr_db','delay_spread_ns','max_doppler_hz','role','channel_seed'};
        for field=1:numel(names), key=names{field}; s.(key)=schedule.(key)(lo:hi); end
        s.data_seed=uint32(double(s.channel_seed)+10000000); s.noise_seed=uint32(double(s.channel_seed)+20000000);
        s.noise_variance=10.^(-double(s.snr_db)/10);
        s.split_name=split; s.dataset_version=p.dataset_version; s.protocol_sha256=freeze.protocol_sha256;
        s.subcarrier_frequencies_hz=(0:119)'*15000; s.symbol_times_s=times;
        for k=1:n
            a=realization(carrier,uniongrid,fs,double(s.delay_spread_ns(k)),double(s.max_doppler_hz(k)),double(s.snr_db(k)),double(s.channel_seed(k)));
            s.H_true(:,:,k)=a.H_true; s.X(:,:,k)=a.X; s.Y(:,:,k)=a.Y;
            s.Y_noiseless(:,:,k)=a.Y_noiseless; s.tx_bits(:,:,:,k)=a.tx_bits;
            s.target_energy(k)=a.target_energy; s.noiseless_mismatch_nmse(k)=a.noiseless_mismatch_nmse;
            s.timing_offset(k)=a.timing_offset;
        end
        temp=[destination '.partial.mat']; save(temp,'-struct','s','-v6');
        movefile(temp,destination);
        complete=struct('status','COMPLETE','split',split,'shard',shard,'channels',n, ...
            'first_channel_id',lo,'last_channel_id',hi,'sha256',filehash(destination), ...
            'protocol_sha256',freeze.protocol_sha256,'generator_sha256',freeze.generator_sha256);
        writejson(marker,complete); count=count+n;
        fprintf('COMMITTED worker=%d %s shard=%d channels=%d elapsed=%.1fs\n',worker,split,shard,count,toc(clock));
        writejson(fullfile(out,sprintf('worker_%d_progress.json',worker)),struct( ...
            'worker',worker,'split',split,'shard',shard,'new_channels',count,'elapsed_seconds',toc(clock)));
        clear s;
    end
end
if strcmp(stage,'replay')
    writejson(fullfile(out,'replay.json'),struct('status','PASS','bitwise_H_X_Y_bits',true,'splits',{splits}));
else
    writejson(fullfile(out,sprintf('worker_%d_complete.json',worker)),struct('status','COMPLETE', ...
        'worker',worker,'workers',workers,'new_channels',count,'elapsed_seconds',toc(clock)));
end
end

function a=realization(carrier,uniongrid,fs,ds,fd,snr,seed)
rng(seed+10000000,'twister');
bits=uint8(randi([0 1],120,14,2));
tx=((1-2*double(bits(:,:,1)))+1i*(1-2*double(bits(:,:,2))))/sqrt(2);
mask=uniongrid~=0; tx(mask)=uniongrid(mask);
% Store sign bits for all REs; BER uses only data_mask, never pilot positions.
b0=bits(:,:,1); b1=bits(:,:,2); b0(mask)=real(tx(mask))<0; b1(mask)=imag(tx(mask))<0;
bits=cat(3,b0,b1);
wave=nrOFDMModulate(carrier,tx,'SampleRate',fs);
ch=nrTDLChannel; ch.NumTransmitAntennas=1; ch.NumReceiveAntennas=1;
ch.SampleRate=fs; ch.DelayProfile='TDL-A'; ch.DelaySpread=ds*1e-9;
ch.MaximumDopplerShift=fd; ch.RandomStream='mt19937ar with seed'; ch.Seed=seed;
filters=getPathFilters(ch); padding=size(filters,1)-1;
[rx,gains,st]=ch([wave;zeros(padding,1)]);
offset=nrPerfectTimingEstimate(gains,filters); rx=rx(1+offset:end,:);
grid=nrOFDMDemodulate(carrier,rx,'SampleRate',fs); grid=grid(:,1:14,1);
h=nrPerfectChannelEstimate(carrier,gains,filters,offset,st,'SampleRate',fs); h=h(1:120,1:14,1,1);
rng(seed+20000000,'twister'); variance=10^(-snr/10);
noise=sqrt(variance/2)*(randn(120,14)+1i*randn(120,14));
a=struct('H_true',single(h),'X',single(tx),'Y',single(grid+noise),'Y_noiseless',single(grid), ...
    'tx_bits',bits,'timing_offset',uint16(offset));
a.target_energy=mean(abs(double(a.H_true(:))).^2);
a.noiseless_mismatch_nmse=sum(abs(double(a.Y_noiseless(:))-double(a.H_true(:)).*double(a.X(:))).^2)/sum(abs(double(a.H_true(:))).^2);
release(ch);
end

function result=filehash(path)
md=java.security.MessageDigest.getInstance('SHA-256');
f=fopen(path,'rb'); assert(f>=0); cleaner=onCleanup(@()fclose(f));
while ~feof(f)
    bytes=fread(f,1048576,'*uint8'); md.update(typecast(bytes,'int8'));
end
result=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end

function writejson(path,value)
temp=[path '.tmp']; f=fopen(temp,'w'); assert(f>=0); cleaner=onCleanup(@()fclose(f));
fwrite(f,jsonencode(value),'char'); clear cleaner; movefile(temp,path,'f');
end
