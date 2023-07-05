import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import csv 
import yaml
import os 
import sqlite3

from iminuit import Minuit, cost

from sotodlib import core, tod_ops
from sotodlib.tod_ops import filters, fft_ops, apodize, detrend # FFT modules
import sotodlib.io.load_smurf as load_smurf
from sotodlib.io.load_smurf import load_file, G3tSmurf, Observations, SmurfStatus
import sotodlib.io.g3tsmurf_utils as utils
import sotodlib.site_pipeline.util as sp_util
import sotodlib.io.metadata as io_meta
import sotodlib.hwp.hwp as hwp
from sotodlib.hwp import demod
from sotodlib.hwp.g3thwp import G3tHWP
from so3g.hk import load_range, HKArchiveScanner
import sodetlib 


def _get_config(config_file):
    return yaml.safe_load(open(config_file, 'r'))


def search_wg_obs(config_file='./test_config.yaml', keyword="wg_step") :

    config2 = _get_config(config_file)

    context_yaml = _get_config(config2['context_file'])
    dbfile = context_yaml['obsdb'].replace('{base_dir}',context_yaml['tags']['base_dir'])
    table = "tags"
    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    obs_ids = cur.execute('select obs_id from %s '%(table)).fetchall()
    tags = cur.execute('select tag from %s '%(table)).fetchall()

    wg_obs_dict = {}
    wg_id_list = []


    for i, tag in enumerate(tags) :

        if tag[0] is None : continue

        if keyword in tag[0] :
            wg_obs_dict[obs_ids[i][0]] = tag[0]
            print(obs_ids[i][0], tag[0])

    return wg_obs_dict


def wg_tag2obsid(tag, config_file='./test_config.yaml', keyword="wg_step") :

    obs_dict = search_wg_obs(config_file=config_file, keyword=keyword)

    for key in obs_dict.keys() :
        if obs_dict[key] == tag :
            return key
    return None


def wg_obsid2tag(obsid, config_file='./test_config.yaml', keyword="wg_step") :

    obs_dict = search_wg_obs(config_file=config_file, keyword=keyword)

    for key in obs_dict.keys() :
        if key == obsid :
            return obs_dict[key]

    return None


def wg_init_setting_p10r1(config_file, stream_id, tag, hk_dir) : 
    
    ### Declare axis manager 
    SMURF = load_smurf.G3tSmurf.from_configs(config_file)
    session = SMURF.Session()   
    obs_list = session.query(Observations).filter(Observations.tag.like(tag), Observations.stream_id.like(stream_id)).all()
    obs_ids = [obs.obs_id for obs in obs_list]
    obs_start = [obs.start for obs in obs_list]
    obs_end = [obs.stop for obs in obs_list]
    # I guess the number of obs_XXX component is just one. But adopt the last component as the end time just in case.
    aman = SMURF.load_data(obs_start[0], obs_end[-1], stream_id = stream_id) 

    
    ### Load data
    utils.load_hwp_data(aman, config_file)
    bias_step_file = utils.get_last_bias_step(obs_ids[0], SMURF)
    bias_step_obj = np.load(bias_step_file, allow_pickle=True).item()
    
    
    ### Load and wrap wiregrid house keeping data
    wg_fields = ['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    hk_in = load_range(obs_start[0], obs_end[-1], wg_fields, data_dir=hk_dir)
    wg_time, wg_enc = hk_in['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    wg_ang = wg_enc/52000.*2.*np.pi
    wg_man = core.AxisManager()
    wg_man.wrap("wg_timestamp", wg_time, [(0, core.OffsetAxis('wg_samps', count=len(wg_time)))])
    wg_man.wrap("wg_angle", wg_ang, [(0, 'wg_samps')])
    aman.wrap("hkwg", wg_man)
    
    return aman


def wg_init_setting(start, end, config_file, hk_dir) :

    ### Declare axis manager
    SMURF = load_smurf.G3tSmurf.from_configs(config_file)
    aman = SMURF.load_data(start, end)

    ### Load data
    utils.load_hwp_data(aman, config_file)
    bias_step_file = utils.get_last_bias_step(start, SMURF)
    bias_step_obj = np.load(bias_step_file, allow_pickle=True).item()

    ### Load and wrap wiregrid house keeping data
    wg_fields = ['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    hk_in = load_range(start, end, wg_fields, data_dir=hk_dir)
    wg_time, wg_enc = hk_in['observatory.wgencoder.feeds.wgencoder_full.reference_count']
    wg_ang = wg_enc/52000.*2.*np.pi
    wg_man = core.AxisManager()
    wg_man.wrap("wg_timestamp", wg_time, [(0, core.OffsetAxis('wg_samps', count=len(wg_time)))])
    wg_man.wrap("wg_angle", wg_ang, [(0, 'wg_samps')])
    aman.wrap("hkwg", wg_man)

    return aman



def wg_demod_tod(aman) :

    ### If axis manager already has demomded components, skip the process
    if ("demodQ" in aman.keys()) or ("demodU" in aman.keys()) : return

    ### Demod
    #detrend.detrend_tod(aman,method='median')
    #apodize.apodize_cosine(aman)
    hwp.demod_tod(aman,signal='signal')

    return



def get_wg_angle(aman, threshold=0.00015, plateau_len=5000, debug=False) :
    

    wg_angle =  aman["hkwg"]["wg_angle"]
    wg_timestamp = aman["hkwg"]["wg_timestamp"]
    
    ### Get static part
    diff_wgangle = np.diff(wg_angle)
    moving = (diff_wgangle>threshold).astype(np.int32)
    switch_indices = np.where(np.diff(moving) != 0)[0] + 1
    run_lengths = np.diff(switch_indices)
    
    static_indices_start = switch_indices[:-1][run_lengths>plateau_len]
    static_indices_end = []
    for _id,run_length in enumerate(run_lengths[run_lengths>plateau_len]) : 
        static_indices_end.append(static_indices_start[_id]+run_length-1)
     
    if debug : 
        fig_check = plt.figure(figsize=[20,4])
        plt.plot(wg_timestamp[:-1][diff_wgangle>0]-1.676598e9,diff_wgangle[diff_wgangle>0])
        for _id in range(len(static_indices_start)) : 
            plt.vlines(wg_timestamp[static_indices_start[_id]]-1.676598e9,0.,0.00035,color="blue")
            plt.vlines(wg_timestamp[static_indices_end[_id]]-1.676598e9,0.,0.00035,color="red")
        plt.show()
    
    ### Generate a list that contents timestamps of wg static start/end and angle of wg
    wg_info = []
    for _id in range(len(static_indices_start)) : 
        angle = np.mean(wg_angle[static_indices_start[_id]:static_indices_end[_id]])
        wg_info.append([wg_timestamp[static_indices_start[_id]],wg_timestamp[static_indices_end[_id]],angle])
    
    return len(static_indices_start),wg_info


def funcQ(x, amp, freq, phase, offsetQ):
    return amp * np.cos ( freq * x + 2.*phase) + offsetQ


def funcU(x, amp, freq, phase, offsetU):
    return amp * np.cos ( freq * x + 2.*phase + np.pi*0.5) + offsetU


def fit_angle(aman,fitting_results,wg_info,rotate_tau=False,debug=False):
    
    ### Calculate HWP rotation speed 
    timestamp = aman["timestamps"]
    dt = (timestamp[-1]-timestamp[0])
    #hwp rotation count
    rt_count = aman["hwp_angle"]*0.5/np.pi
    num_rot = np.sum((np.diff(rt_count)>0.9).astype(np.int32))
    hwp_speed = num_rot/dt
    dspeed = np.abs(hwp_speed - 2.)
    print(f"HWP rotation speed is {hwp_speed}")
    
    ### Get compoents of wg_info then calculate polarized angle
    each_result = []
    ang_mean,Q_mean,U_mean,Q_error,U_error=[],[],[],[],[]
    fig1 = plt.figure()
    for _id in range(len(wg_info)) : 
        wg_angle = wg_info[_id][2]
        t_selection = (aman.timestamps > wg_info[_id][0]) & (aman.timestamps < wg_info[_id][1])
           
        ### time constant correction
        Q = aman.demodQ[:,t_selection]
        U = aman.demodU[:,t_selection]
        
        if debug :
            fig = plt.figure()
            plt.plot(timestamp[t_selection],Q[60])
            plt.plot(timestamp[t_selection],U[60])
            plt.show()
         
        if rotate_tau : 
            exp = np.exp(t_const*hwp_speed*2.*np.pi*4.).reshape((len(t_const), 1)) #  Need to update this later
        else : 
            exp = 1.
                
        rotated_tod = exp*(Q+1j*U)
        Q = np.real(rotated_tod)
        U = np.imag(rotated_tod)
        
        
        ### get sin curve
        ang_mean.append(wg_angle)
        Q_mean.append(np.mean(Q,axis=1))
        U_mean.append(np.mean(U,axis=1))
        Q_error.append(np.std(Q,axis=1))
        U_error.append(np.std(U,axis=1))

    
    ang_mean = np.array(ang_mean)
    Q_mean = np.array(Q_mean)
    U_mean = np.array(U_mean)
    Q_error = np.array(Q_error)
    U_error = np.array(U_error)
    
    
    ### Fitting with iMinuit
    fitted_phases,fitted_amps,fitted_offsetq,fitted_offsetu = ['phase'],['amp'],['offsetq'],['offsetu']
    error_phases,error_amps,error_offsetq,error_offsetu = ['phase_e'],['amp_e'],['offsetq_e'],['offsetu_u']
    channels,bands,subbands,bandpasses = ['channel'],['band'],['subband'],['bandpass']
    minuit_chi2 = ['chi2']
    
    for i in range(len(Q_mean[0])) : 

        if np.sum(np.isnan(Q_mean[:,i]).astype(np.int32)) > 0 : 
            print(aman["det_info"]["readout_id"][i])
            print(aman.dets.vals)
            continue 

        Qi,Ui = Q_mean[:,i],U_mean[:,i]
        Qi_e,Ui_e = Q_error[:,i],U_error[:,i]                  
            
        def fcn(amp,freq,phase,offsetQ,offsetU):
            model1 = amp * np.cos ( freq * ang_mean + 2.*phase) + offsetQ 
            model2 = amp * np.cos ( freq * ang_mean + 2.*phase + np.pi*0.5) + offsetU
            chi_squared1 =  ((model1-Qi)/Qi_e)*((model1-Qi)/Qi_e)
            chi_squared2 =  ((model2-Ui)/Ui_e)*((model2-Ui)/Ui_e)
            return np.sum(chi_squared1)+np.sum(chi_squared2)
            
        m = Minuit(fcn, amp=(np.max(Q_mean[:,i])-np.min(Q_mean[:,i]))*0.5, freq=hwp_speed, phase=0.5*np.pi, offsetQ=np.mean(Q_mean[:,i]), offsetU=np.mean(U_mean[:,i]))
        m.limits['amp'] = (0, None)
        m.limits['phase'] = (0.,2.*np.pi)
        m.limits['freq'] = (2.,2.)
        m.migrad().migrad().hesse()

        #  "ids","amplitude","amplitude_e","angle","angle_e","offset_q","offset_u","phase","chi2"
        _id = aman["det_info"]["readout_id"][i]
        _amp = m.values[0]
        _amp_e = m.errors[0]
        _ang = m.values[2]
        if _ang > np.pi : _ang -= np.pi
        _ang_e = m.errors[2]
        _off_q = m.values[3]
        _off_u = m.values[4]
        #_chi2 = m.fmin.reduced_chi2
        ndf = len(Qi) + len(Ui) - len(m.values)
        _chi2 = m.fval/ndf
           
        fitting_results.rows.append((_id,_amp,_amp_e,_ang,_ang_e,_off_q,_off_u,_chi2))
   
        if debug and (i%30 == 0) :
            print(_chi2,m.values[1])
            fig = plt.figure()
            angle = np.arange(628)*0.01
            plt.errorbar(ang_mean,Q_mean[:,i],yerr=Q_error[:,i],fmt='.',color='r')
            plt.errorbar(ang_mean,U_mean[:,i],yerr=U_error[:,i],fmt='.',color='b')
            plt.plot(angle, funcQ(angle, m.values[0], m.values[1], m.values[2], m.values[3]), ls="--", label="fittedQ",color='r')
            plt.plot(angle, funcU(angle, m.values[0], m.values[1], m.values[2], m.values[4]), ls="--", label="fittedU",color='b')
            plt.show()
            
            
    return fitting_results



def main(config_file='/homes/atakeuchi/workspace/SO/wg_dev/sat1_p10r2/test_config.yaml', query=None, obs_id="obs_1676597367_sat1_1111101", overwrite=False, min_ctime=None, max_ctime=None,logger=None,):

    verbose = 0
    # set logger
    logger = sp_util.init_logger(__name__, 'make_abs_cal_model: ')
    if verbose >= 1:
        logger.setLevel('INFO')
    if verbose >= 2:
        sotodlib.logger.setLevel('INFO')
    if verbose >= 3:
        sotodlib.logger.setLevel('DEBUG')

    # load config file
    config = _get_config(config_file)

    # load context file
    context = core.Context(config['context_file'])
    obsdb = context.obsdb
    obs = obsdb.query(f'obs_id == "{obs_id}"')[0]

    # place of house keeping data
    hk_dir = config['hk_dir']

    aman = context.get_meta(obs_id)


    # Analysis settings
    tag = 'obs,stream,'+wg_obsid2tag(obsid=obs_id, config_file=config_file)
    arrays = config['arrays']
    output_h5 = config['archive']['policy']['filename']

    if os.path.exists(config['archive']['index']):
        logger.info(f'Mapping {config["archive"]["index"]} for the archive index.')
        db = core.metadata.ManifestDb(config['archive']['index'])
    else:
        logger.info(f'Creating {config["archive"]["index"]} for the archive index.')
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(config['archive']['index'], scheme=scheme)


    fitting_results = core.metadata.ResultSet(
        keys=["dets:readout_id","amplitude","amplitude_e","angle","angle_e","offset_q","offset_u","chi2"]
    )


    for array in arrays :
        stream_id = array['stream_id']
        aman = wg_init_setting_p10r1(config_file=config_file,stream_id=stream_id,tag=tag,hk_dir=hk_dir)
        wg_demod_tod(aman)
        num_angle, wg_info = get_wg_angle(aman,debug=False)
        fitting_results = fit_angle(aman,fitting_results,wg_info,debug=False)
        del aman


    # Save outputs
    db_data = {'obs:obs_id': obs_id, 'dataset' : obs_id}
    db.add_entry(db_data, output_h5, f'{obs_id}',replace=True)
    db.to_file(config['archive']['index'])
    io_meta.write_dataset(fitting_results, output_h5, f'{obs_id}', overwrite=True)


if __name__ == '__main__' :

    #sp_util.main_launcher(main,get_parser)
    main()
