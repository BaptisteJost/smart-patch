# import os
#
# path_to_fgbuster = '/mnt/c/Users/Baptiste/fgbuster'
# os.sys.path.insert(0, os.path.realpath(path_to_fgbuster))


import numpy as np
import healpy as hp
import pylab as py
import scipy
import sys
import matplotlib.pyplot as plt
import pysm
from numba import jit
#import fgbuster as fg
from fgbuster.pysm_helpers import get_instrument, get_sky
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.separation_recipies import _get_prewhiten_factors,_A_evaluators
import fgbuster.algebra as fgal
import numdifftools
import copy
from mpl_toolkits.mplot3d import Axes3D
#P_i = list of list for component i, each list corresponds to the pixel of one patch, i.e. P_d=[patch0, patch1,...,patchn] & patchi=[pixel0,pixel1,...]
#input_set_of_betas is a list of all betas for each component and patch i.e. input_set_of_betas=[beta_d1,beta_d2,...,beta_dn,beta_s1,...,beta_sm,temp1,...,templ]
    #by convention the component are ordered as beta_dust,beta_sync,temperature_dust


#------------------------------------------------------SKY GENERATION----------------------------------------------------
nside_comparison_for_fine_tuning=False
if nside_comparison_for_fine_tuning!=True:
    nside = 8
    pixel_number=hp.nside2npix(nside)
    pysm_model='c1d1s1'
    sky = get_sky(nside, pysm_model)
    nu = np.array([40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1])

    pixel_number=hp.nside2npix(nside)

    npix=hp.nside2npix(nside)
    instrument = get_instrument(nside, 'litebird')
    self=instrument

    pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.Nside)) * (180. * 60. / np.pi) ** 2
    """sigma_pix_I/P is std of noise per pixel. It is an array of length
    equal to the number of input maps."""
    sigma_pix_I = np.sqrt(self.Sens_I ** 2 / pix_amin2)
    sigma_pix_P = np.sqrt(self.Sens_P ** 2 / pix_amin2)

    np.random.seed (seed=None)
    noise = np.random.randn(len(self.Sens_I), 3, npix)
    noise[:, 0, :] *= sigma_pix_I[:, None]
    noise[:, 1, :] *= sigma_pix_P[:, None]
    noise[:, 2, :] *= sigma_pix_P[:, None]


    freq_maps = instrument.observe(sky, write_outputs=False)[0] + noise
    freq_maps=freq_maps[:,1:,:]  #on retire la temperature (1:) car on ne prend en compte que Q&U pas I
    components = [CMB(), Dust(150.), Synchrotron(20.)]
    prewhiten_factors = _get_prewhiten_factors(instrument, freq_maps.shape)



    hp.mollview(sky.cmb(nu)[1,1,:])
    plt.savefig('true_res_11_nside=%s'%nside)
    plt.show()
    hp.mollview(sky.cmb(nu)[1,2,:])
    plt.savefig('true_res_12_nside=%s'%nside)
    plt.show()
    # sys.exit()
    # instrument = get_instrument(nside, 'litebird')
    # freq_maps = instrument.observe(sky, write_outputs=False)[0] + instrument.observe(sky, write_outputs=False)[1]
    # freq_maps=freq_maps[:,1:,:]  #on retire la temperature (1:) car on ne prend en compte que Q&U pas I
    # components = [CMB(), Dust(150.), Synchrotron(20.)]
    # prewhiten_factors = _get_prewhiten_factors(instrument, freq_maps.shape)         # correspond a N^-1/2

    print 'freq_maps shape',np.shape(freq_maps)

    # print 'pysm.components.Dust.spectral_index',hp.read_map(pysm.components.Dust.Spectral_Index)
    # hp.read_map(template('dust_beta.fits'), nside = nside, field = 0)

    #----------------------------------------------------NO MASK---------------------------------------------------------------------
    # mask_bin=[1]*pixel_number
    # masked_pixels=[]


    #----------------------------------------------------MASK-------------------------------------------------------------------------
    mask=hp.read_map('../map_test/HFI_Mask_GalPlane-apo2_2048_R2.00.fits',field=2)
    mask_bin=mask*0
    mask_bin=hp.ud_grade(mask_bin,nside)
    mask_bin[np.where(hp.ud_grade(mask,nside)!=0)[0]]=1
    masked_pixels=[np.where(hp.ud_grade(mask,nside)==0)[0]]
    print(' number of masked_pixels',len(masked_pixels[0]))

    #--------------------------------------------------------------------------------------------------------------------------
    freq_maps_save=freq_maps[:][:][:]
    if len(masked_pixels)!=0:
        freq_maps= np.zeros((len(freq_maps_save), len(freq_maps_save[0]),pixel_number-len(masked_pixels[0])))
        # print 'freq_maps test',freq_maps
        for i in range(len(freq_maps_save)):
            for j in range(len(freq_maps_save[i])):
                freq_maps[i][j]=np.delete(freq_maps_save[i][j],masked_pixels)
                # print(np.delete(freq_maps_save[i][j],masked_pixels))

        pixel_list=[]
        # print('masked_pixels=',masked_pixels[1][0])
        temp=[]
        for j in range(pixel_number):
            for i in range(len(masked_pixels[0])):
                if j==masked_pixels[0][i]:
                    temp.append(j)
            if len(temp)==0:
                pixel_list.append(j)
            temp=[]
        # print('pixel_list=',pixel_list)
        pixel_number=pixel_number-len(masked_pixels[0])
    else:
        pixel_list=range(pixel_number)

    # print 'np.transpose(freq_maps)[0] shape',np.shape(np.transpose(freq_maps)[0])
    # print 'np.diag(prewhiten_factors**2) shape',np.shape(np.diag(prewhiten_factors**2))
    # print 'np.diag(prewhiten_factors) shape',np.shape(np.diag(prewhiten_factors))
    # print 'prewhiten_factors shape',np.shape(prewhiten_factors)
    noise=noise[:,1:,:]
    # print 'np.transpose(noise)[0] shape', np.shape(np.transpose(noise)[0])
    # sys.exit()

# print 'pixel_number', len(pixel_list)
# print 'pixel_list',pixel_list
# for i in range(len(freq_maps_save)):
#     for j in range(len(freq_maps_save[i])):
#         freq_maps[i][j]=freq_maps_save[i][j]*mask_bin

# hp.mollview(freq_maps[0][0])
# plt.show()

# P_d=[range(48),range(48,96),range(96,144),range(144,192)]
# P_t=[range(48),range(48,96),range(96,144),range(144,192)]
# P_s=[range(48),range(48,96),range(96,144),range(144,192)]
# for i in range(len(P_s)):
#     try:
#         P_s[i].index(45)
#         bd_index=i
#         print('bd_index=',bd_index)
#     except:
#         pass
# print bd_index


#-----------------------------------------The A matrix and the likelihood function are computed once and for all in the program------------
# prewhitened_data = prewhiten_factors * np.transpose(freq_maps)
# A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=prewhiten_factors)
# A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)
# print('A_dB_ev',A_dB_ev([1.5,19,-3]))


# for p in range(hp.nside2npix(nside)):
#     funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
#     fun.append(funtemp)
#     last_values.append(last_valuestemp)
#     print(funtemp([1.59,19.6,-3.1]))
#     del funtemp, jactemp,last_valuestemp

#the Likelihood function is computed for each pixel and storred in the fun array where each element is the likelihood function for the corresponding pixel



def data_patch(freq_maps,P_i,pixel_list): #this function returns the data corresponding to the pixels in the input patch. Works for patches as well as for meta patches
    data=[]
    temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    index_list=[]
    for i in range(len(P_i)):
        for f in range(len(freq_maps)):
            for s in range(len(freq_maps[0])):
                for p in range(len(P_i[i])):
                    temp_list[f][s].append(freq_maps[f][s][pixel_list.index(P_i[i][p])])
        data.append(temp_list)# for p in range(len(meta_P[i])))
        temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    del temp_list
    return data

def data_pixel(freq_maps,P_i,nside,pixel_list): #this function returns the data corresponding to the pixels in the input patch. Works for patches as well as for meta patches
    data=[None]*hp.nside2npix(nside)
    temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    index_list=[]
    for i in range(len(P_i)):
        for f in range(len(freq_maps)):
            for s in range(len(freq_maps[0])):
                for p in range(len(P_i[i])):
                    temp_list[f][s].append(freq_maps[f][s][pixel_list.index(P_i[i][p])])
        data.append(temp_list)# for p in range(len(meta_P[i])))
        temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    del temp_list
    return data

def patch_map(input_set_of_betas,P_d,P_t,P_s,nside):
    ind=0
    pixel_number1=hp.nside2npix(nside)
    map_bd=[None]*pixel_number1
    map_bt=[None]*pixel_number1
    map_bs=[None]*pixel_number1
    # print 'input_set_of_betas', input_set_of_betas
    # print 'len(P_d)',len(P_d)
    for i in range(len(P_d)):
        for j in range(len(P_d[i])):
            # print 'ind',ind
            map_bd[P_d[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_t)):
        for j in range(len(P_t[i])):
            # print 'ind',ind
            map_bt[P_t[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in range(len(P_s[i])):
            map_bs[P_s[i][j]]=input_set_of_betas[ind]
        # print('index=',ind)
        ind+=1
    return map_bd,map_bt,map_bs

def joint_spectral_likelihood(input_set_of_betas,fun,P_d,P_t,P_s,pixel_list,nside,jac): #computes the joint likelihood for the whole sky taking into account the patches
    #prewhiten_factors must be defined above !
    logL_spec=0
    #first we have to create the maps of beta in the sky to link each patch with its corresponding beta value
    # ind=0
    # pixel_number1=hp.nside2npix(nside)
    # map_bd=[None]*pixel_number1
    # map_bt=[None]*pixel_number1
    # map_bs=[None]*pixel_number1
    # # print('len(P_d)=',len(P_d))
    # # print('len(P_d[0])=',len(P_d[0]))
    # # print('np.shape(input_set_of_betas)=',np.shape(input_set_of_betas))
    # # print(P_d)
    # for i in range(len(P_d)):
    #     # print('map_bd=',np.shape(map_bd))
    #     # print 'i=',i
    #     for j in P_d[i]:
    #         # print 'j=',j
    #         map_bd[j]=input_set_of_betas[ind]
    #     ind+=1
    #
    # for i in range(len(P_t)):
    #     for j in P_t[i]:
    #         map_bt[j]=input_set_of_betas[ind]
    #     ind+=1
    #
    # for i in range(len(P_s)):
    #     for j in P_s[i]:
    #         map_bs[j]=input_set_of_betas[ind]
    #     ind+=1

    # for p in pixel_list:
        # print(fun[p]([map_bd[p],map_bt[p],map_bs[p]]))
    # print 'fun[p]([map_bd[p],map_bt[p],map_bs[p]])',fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    # print 'input_set_of_betas',input_set_of_betas

    map_bd,map_bt,map_bs=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)
    # print 'input_set_of_betas',input_set_of_betas
    # print 'jac',jac(input_set_of_betas,fun,P_d,P_t,P_s,pixel_list,nside,jac)

    for p in pixel_list:
        logL_spec+=fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    del map_bd,map_bt,map_bs
    return logL_spec

def joint_spectral_likelihood_one_patch(input_set_of_betas,P_i,patch_index,pixel_list,nside):
    logL_spec=0
    map_bd,map_bt,map_bs=patch_map(input_set_of_betas,P_i,P_i,P_i,nside)

    for p in P_i[patch_index]:
        logL_spec+=fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    del map_bd,map_bt,map_bs
    logL_spec*=1e-9
    return logL_spec

def sigma_matrix_making(minimization_result,P_d,P_t,P_s,last_values): #computes the sigma matrix for each patch and creates the total sigma matrix for the whole sky
    ind=0
    pixel_number1=hp.nside2npix(nside)
    map_bd=[None]*pixel_number1
    map_bt=[None]*pixel_number1
    map_bs=[None]*pixel_number1
    fisher_tot=[[0 for j in range(len(P_d)+len(P_t)+len(P_s))]for i in range(len(P_d)+len(P_t)+len(P_s))]
    # print(fisher_tot)
    #print(fisher_tot[0][0])
    # print 'length',len(P_d)+len(P_t)+len(P_s)
    for i in range(len(P_d)):
        for j in range(len(P_d[i])):
            map_bd[P_d[i][j]]=minimization_result[ind]
        ind+=1
    for i in range(len(P_t)):
        for j in range(len(P_t[i])):
            map_bt[P_t[i][j]]=minimization_result[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in range(len(P_s[i])):
            map_bs[P_s[i][j]]=minimization_result[ind]
        ind+=1

    sigma_m=[]

    for p in pixel_list:
        # print 'p test',p
        # print 'sigma p', p
        # fun[p]([map_bd[p],map_bt[p],map_bs[p]])#[minimization_result.x[l*3],minimization_result.x[l*3+1],minimization_result.x[l*3+2]]
        u_e_v_last, A_dB_last, x_last, pw_d = last_values[p]
        # print 'u_e_v_last[0][1] sigma fisher0 pixel p:%s'%p,u_e_v_last[0][1]
        s =fgal._Wd_svd(u_e_v_last[0], pw_d[0])

        # if A_dB_ev is None:
        #     fisher = numdifftools.Hessian(fun)(minimization_result.x)
        # else:
        # print 'u_e_v_last[0][1] sigma fisher1 pixel p:%s'%p,u_e_v_last[0][1]
        fisher = fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s,
                                            A_dB_last[0], comp_of_dB)

        if np.isnan(s[0][0]) or np.isnan(s[0][1]) or np.isnan(s[0][2]) or np.isnan(s[1][0]) or np.isnan(s[1][1]) or np.isnan(s[1][2]):
            print ' /!\ NAN in S /!\ '
            print 's',s
            print 'u_e_v_last[0][1]',u_e_v_last[0][1]
            print 'pw_d[0]',pw_d[0]
        #     # print 'last_values', last_values
        #     print 'fun',fun[p]([map_bd[p],map_bt[p],map_bs[p]])
        #     print 'comp_of_dB', comp_of_dB
        #     print 'u_e_v_last[0]',u_e_v_last[0]
        #     print 'A_dB_last[0]',A_dB_last[0]


        # print([map_bd[p],map_bt[p],map_bs[p]])
        # print('u_e_v_last[0]=',u_e_v_last[0],'s=', s,'A_dB_last[0]=',A_dB_last[0],'comp_of_dB=', comp_of_dB)
        # print('numdifftools.Hessian(fun)(minimization_result.x)=',numdifftools.Hessian(fun[p])([map_bd[p],map_bt[p],map_bs[p]]))
        #
        # print('fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s, A_dB_last[0], comp_of_dB)=',fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s,A_dB_last[0], comp_of_dB))
        # if h!=0:
        #     fisher=np.zeros((3,3))
        # else:
        #     fisher=numdifftools.Hessian(fun[p])([map_bd[p],map_bt[p],map_bs[p]])
        # Sigma = np.linalg.inv(fisher)

        #sigma_m.append(Sigma)

        for i in range(len(P_d)):
            try:
                P_d[i].index(p)
                bd_index=i
            except:
                pass

        for i in range(len(P_t)):
            try:
                P_t[i].index(p)
                bt_index=i
            except:
                pass

        for i in range(len(P_s)):
            try:
                P_s[i].index(p)
                bs_index=i
            except:
                pass
        # print('fisher_tot=',fisher_tot)
        # print('Sigma[0][0]=',Sigma[0][0])
        fisher_tot[bd_index][bd_index]+=fisher[0][0]
        fisher_tot[len(P_d)+bt_index][len(P_d)+bt_index]+=fisher[1][1]
        fisher_tot[len(P_d)+len(P_t)+bs_index][len(P_d)+len(P_t)+bs_index]+=fisher[2][2]

        fisher_tot[bd_index][len(P_d)+bt_index]+=fisher[0][1]
        fisher_tot[len(P_d)+bt_index][bd_index]+=fisher[1][0]

        fisher_tot[bd_index][len(P_d)+len(P_t)+bs_index]+=fisher[0][2]
        fisher_tot[len(P_d)+len(P_t)+bs_index][bd_index]+=fisher[2][0]

        fisher_tot[len(P_d)+bt_index][len(P_d)+len(P_t)+bs_index]+=fisher[1][2]
        fisher_tot[len(P_d)+len(P_t)+bs_index][len(P_d)+bt_index]+=fisher[2][1]
    # print(fisher_tot)
    # print 'fisher_tot_shape',np.shape(fisher_tot)
    sigma_tot=np.linalg.inv(fisher_tot).T
    return(sigma_tot)

def fisher_pixel(map_bd,map_bt,map_bs,fun,pixel_list,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB,Hessian=False):
    fisher_list=[None]*hp.nside2npix(nside)
    # fisher_list_N=[None]*hp.nside2npix(nside)
    # s_q_00=[None]*hp.nside2npix(nside)
    # s_q_10=[None]*hp.nside2npix(nside)
    # s_q_20=[None]*hp.nside2npix(nside)
    # s_q_01=[None]*hp.nside2npix(nside)
    # s_q_11=[None]*hp.nside2npix(nside)
    # s_q_21=[None]*hp.nside2npix(nside)
    s_list=[None]*hp.nside2npix(nside)
    n_cmb_list=[None]*hp.nside2npix(nside)
    fisher_s_list=[None]*hp.nside2npix(nside)
    fisher_n_list=[None]*hp.nside2npix(nside)
    fisher_hess_list=[None]*hp.nside2npix(nside)
    i=0
    for p in pixel_list:
        # fun[p]([map_bd[p],map_bt[p],map_bs[p]])#[minimization_result.x[l*3],minimization_result.x[l*3+1],minimization_result.x[l*3+2]]
        # u_e_v_last, A_dB_last, x_last, pw_d = last_values[p]
        # print 'u_e_v_last[0][1] sigma fisher0 pixel p:%s'%p,u_e_v_last[0][1]

        # s =fgal._Wd_svd(u_e_v_last[0], pw_d[0])
        s=fgal.Wd(A_ev([map_bd[p],map_bt[p],map_bs[p]]),np.transpose(freq_maps)[i], np.diag(prewhiten_factors**2), return_svd=False)
        n_cmb_list[p]=fgal.Wd(A_ev([map_bd[p],map_bt[p],map_bs[p]]),np.transpose(noise)[i], np.diag(prewhiten_factors**2), return_svd=False)
        # print 's_shape', np.shape(s)
        # print 'n_cmb shape',np.shape(n_cmb_list[p])
        # print 'ind_cmb', A_ev.components.index('CMB')
        # print '_invAtNA_svd(u_e_v_last[0])',fgal._invAtNA_svd(u_e_v_last[0])
        # print 's-n',s-fgal._invAtNA_svd(u_e_v_last[0])
        # print '_invAtNA_svd(u_e_v_last[0]) shape', np.shape(fgal._invAtNA_svd(u_e_v_last[0]))
        # print 's shape',np.shape(s)
        # print 's',s

        # nnt_list[p]=fgal._invAtNA_svd(u_e_v_last[0])
        # fisher_s_list[p]=fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s, A_dB_last[0], comp_of_dB)
        # fisher_n_list[p]=fgal._fisher_logL_dB_dB_svd(u_e_v_last[0],fgal._invAtNA_svd(u_e_v_last[0]),A_dB_last[0], comp_of_dB)
        # if A_dB_ev is None:
        if Hessian == True:
            fisher_hess_list[p] = numdifftools.Hessian(fun[p])([map_bd[p],map_bt[p],map_bs[p]])
        i+=1
        # else:
        # print 'u_e_v_last[0][1] sigma fisher1 pixel p:%s'%p,u_e_v_last[0][1]

        # fisher_list[p]= fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s, A_dB_last[0], comp_of_dB)# - fgal._fisher_logL_dB_dB_svd(u_e_v_last[0],fgal._invAtNA_svd(u_e_v_last[0]),A_dB_last[0], comp_of_dB)

        # fisher_list_N[p]=
        fisher_list[p]=fgal.fisher_logL_dB_dB(A_ev([map_bd[p],map_bt[p],map_bs[p]]), s, A_dB_ev([map_bd[p],map_bt[p],map_bs[p]]), comp_of_dB, np.diag(prewhiten_factors**2), return_svd=False)
        # s_q_00[p]=s.T[0][0]
        # s_q_10[p]=s.T[1][0]
        # s_q_20[p]=s.T[2][0]
        # s_q_01[p]=s.T[0][1]
        # s_q_11[p]=s.T[1][1]
        # s_q_21[p]=s.T[2][1]
        s_list[p]=s

        # print 'fisher_list[p]',fisher_list[p]
        # print 'fisher_list_N[p]',fisher_list_N[p]
        # if np.log(1/fisher_list[p][0][0])>0:
        #     print 'fisher_list[%s][1][1]'%p,np.log(1/fisher_list[p][1][1])

        if np.isnan(s[0][0]) or np.isnan(s[0][1]) or np.isnan(s[0][2]) or np.isnan(s[1][0]) or np.isnan(s[1][1]) or np.isnan(s[1][2]):
            print ' /!\ NAN in S /!\ '
            print 'at pixel %s'%p
            print 's',s
            # print 'u_e_v_last[0][1]',u_e_v_last[0][1]
            # print 'pw_d[0]',pw_d[0]
    if Hessian==True:
        return fisher_list,s_list,n_cmb_list,fisher_s_list,fisher_n_list,fisher_hess_list
    return fisher_list,s_list,n_cmb_list,fisher_s_list,fisher_n_list

def patch_making_pbyp(input_beta_zeroth_iteration,fisher_list,pixel_number,pixel_list):

    sorted_beta_d_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k])
    sorted_beta_t_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k+pixel_number])
    sorted_beta_s_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k+pixel_number*2])
    """
    print 'sorted beta d'
    for p in range(pixel_number):
        # print 'pixel_list[sorted_beta_d_pixel_list[p]]',pixel_list[sorted_beta_d_pixel_list[p]]
        # print 'sorted_beta_d_pixel_list[p]',sorted_beta_d_pixel_list[p]
        # print 'p',p
        print input_beta_zeroth_iteration[sorted_beta_d_pixel_list[p]]

    print 'sorted beta t'
    for p in range(pixel_number):
        print input_beta_zeroth_iteration[sorted_beta_t_pixel_list[p]+pixel_number]

    print 'sorted beta s'
    for p in range(pixel_number):
        print input_beta_zeroth_iteration[sorted_beta_s_pixel_list[p]+2*pixel_number]
    """
    P_d_new=[]
    P_t_new=[]
    P_s_new=[]
    sigma_patch_list=[]

    beta2_patch_list=[]



    #-----------------------------------------------------------------------------------P_d_new--------------------------------------------------------------------

    sigma_patch_temp=0 #arbitrary
    delta_beta_temp=0.0 #arbitrary
    pixel_patch_temp=[]
    p_list=[]
    patch_list=[]
    fisher_tot=0
    p=0

    # fisher_tot+=fisher_list[sorted_beta_d_pixel_list[0]] # /!\ PROBLEME CAR ON NE PEUT PAS AVOIR DE PATCH DE TAILLE UN PIXEL (OU BIEN TOUT LES PATCHES AURONT TAILLE 1 )
    # pixel_patch_temp.append(0)
    #
    # print 'fisher_list[sorted_beta_d_pixel_list[82]]',fisher_list[82]
    # print 'len fisher',len(fisher_list)
    patchnum=1

    while p!=pixel_number:
        # print 'pp',p
        # print 'fisher_list[sorted_beta_d_pixel_list[p]]',fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]
        fisher_tot+=fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]
        pixel_patch_temp.append(pixel_list[sorted_beta_d_pixel_list[p]])
        p_list.append(p)
        sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]

        p+=1
        while sigma_patch_temp >= delta_beta_temp*delta_beta_temp:
            if p==pixel_number:
                # print 'pixel limit reached'
                # print 'p',p
                break

            # print 'p',p
            # print 'pixel_list[sorted_beta_d_pixel_list[p]]',pixel_list[sorted_beta_d_pixel_list[p]]
            fisher_tot+=fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]

            sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]
            pixel_patch_temp.append(pixel_list[sorted_beta_d_pixel_list[p]])
            p_list.append(p)
            # print '[input_beta_zeroth_iteration[x] for x in pixel_patch_temp]',[input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list]
            # print 'p_list',p_list
            # print '[sorted_beta_d_pixel_list[x] for x in p_list]',[sorted_beta_d_pixel_list[x] for x in p_list]
            # print '[input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list]',[input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list]
            delta_beta_temp=np.std([input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list])
            # print 'db^2=',delta_beta_temp*delta_beta_temp
            # print 'sigma=',sigma_patch_temp
            p+=1

        patchnum+=1
        # print 'NEW d PATCH ! patch number=',patchnum

        P_d_new.append(pixel_patch_temp)
        sigma_patch_list.append(sigma_patch_temp)
        beta2_patch_list.append(delta_beta_temp*delta_beta_temp)
        pixel_patch_temp=[]
        p_list=[]
        sigma_patch_temp=0 #arbitrary
        delta_beta_temp=0.0 #arbitrary
        fisher_tot=0
    print 'patch number for beta d =',patchnum
    #-----------------------------------------------------------------------------------P_t_new--------------------------------------------------------------------

    sigma_patch_temp=0 #arbitrary
    delta_beta_temp=0.0 #arbitrary
    pixel_patch_temp=[]
    p_list=[]
    patch_list=[]
    fisher_tot=0
    p=0
    patchnum=1
    while p!=pixel_number:
        # print 'pp',p
        # print 'fisher_list[sorted_beta_d_pixel_list[p]]',fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]
        fisher_tot+=fisher_list[pixel_list[sorted_beta_t_pixel_list[p]]]
        pixel_patch_temp.append(pixel_list[sorted_beta_t_pixel_list[p]])
        p_list.append(p)
        sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]

        p+=1
        while sigma_patch_temp >= delta_beta_temp*delta_beta_temp:
            if p==pixel_number:
                # print 'pixel limit reached'
                # print 'p',p
                break

            # print 'p',p
            # print 'pixel_list[sorted_beta_d_pixel_list[p]]',pixel_list[sorted_beta_d_pixel_list[p]]
            fisher_tot+=fisher_list[pixel_list[sorted_beta_t_pixel_list[p]]]

            sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]
            pixel_patch_temp.append(pixel_list[sorted_beta_t_pixel_list[p]])
            p_list.append(p)
            # print '[input_beta_zeroth_iteration[x] for x in pixel_patch_temp]',[input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list]
            delta_beta_temp=np.std([input_beta_zeroth_iteration[sorted_beta_t_pixel_list[x]+pixel_number] for x in p_list])
            # print 'db^2=',delta_beta_temp*delta_beta_temp
            # print 'sigma=',sigma_patch_temp
            p+=1

        patchnum+=1
        # print 'NEW PATCH t ! patch number=',patchnum
        sigma_patch_list.append(sigma_patch_temp)
        beta2_patch_list.append(delta_beta_temp*delta_beta_temp)
        P_t_new.append(pixel_patch_temp)
        pixel_patch_temp=[]
        p_list=[]
        sigma_patch_temp=0 #arbitrary
        delta_beta_temp=0.0 #arbitrary
        fisher_tot=0

    print 'patch number for beta t =',patchnum
    #-----------------------------------------------------------------------------------P_s_new--------------------------------------------------------------------

    sigma_patch_temp=0 #arbitrary
    delta_beta_temp=0.0 #arbitrary
    pixel_patch_temp=[]
    p_list=[]
    patch_list=[]
    fisher_tot=0
    p=0
    patchnum=1
    while p!=pixel_number:
        # print 'pp',p
        # print 'fisher_list[sorted_beta_d_pixel_list[p]]',fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]
        fisher_tot+=fisher_list[pixel_list[sorted_beta_s_pixel_list[p]]]
        pixel_patch_temp.append(pixel_list[sorted_beta_s_pixel_list[p]])
        p_list.append(p)
        sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]

        p+=1
        while sigma_patch_temp >= delta_beta_temp*delta_beta_temp:
            if p==pixel_number:
                # print 'pixel limit reached'
                # print 'p',p
                break

            # print 'p',p
            # print 'pixel_list[sorted_beta_d_pixel_list[p]]',pixel_list[sorted_beta_d_pixel_list[p]]
            fisher_tot+=fisher_list[pixel_list[sorted_beta_s_pixel_list[p]]]

            sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]
            pixel_patch_temp.append(pixel_list[sorted_beta_s_pixel_list[p]])
            # print pixel_patch_temp
            p_list.append(p)
            # print '[input_beta_zeroth_iteration[x] for x in pixel_patch_temp]',[input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list]
            delta_beta_temp=np.std([input_beta_zeroth_iteration[sorted_beta_s_pixel_list[x]+2*pixel_number] for x in p_list])
            # print 'db^2=',delta_beta_temp*delta_beta_temp
            # print 'sigma=',sigma_patch_temp
            p+=1

        patchnum+=1
        # print 'NEW s PATCH ! patch number=',patchnum
        sigma_patch_list.append(sigma_patch_temp)
        beta2_patch_list.append(delta_beta_temp*delta_beta_temp)
        P_s_new.append(pixel_patch_temp)
        pixel_patch_temp=[]
        p_list=[]
        sigma_patch_temp=0 #arbitrary
        delta_beta_temp=0.0 #arbitrary
        fisher_tot=0

    print 'patch number for beta d =',patchnum
    # print 'OUT OF THE LOOP'
    return P_d_new,P_t_new,P_s_new,sigma_patch_list,beta2_patch_list

# def sigma_addition(fisher_list,P_d,P_t,P_s):
#     for p in :
#         for i in range(len(P_d)):
#             try:
#                 P_d[i].index(p)
#                 bd_index=i
#             except:
#                 pass
#
#         for i in range(len(P_t)):
#             try:
#                 P_t[i].index(p)
#                 bt_index=i
#             except:
#                 pass
#
#         for i in range(len(P_s)):
#             try:
#                 P_s[i].index(p)
#                 bs_index=i
#             except:
#                 pass
#
#         fisher_tot[bd_index][bd_index]+=fisher_list[p][0][0]
#         fisher_tot[len(P_d)+bt_index][len(P_d)+bt_index]+=fisher_list[p][1][1]
#         fisher_tot[len(P_d)+len(P_t)+bs_index][len(P_d)+len(P_t)+bs_index]+=fisher_list[p][2][2]
#
#         fisher_tot[bd_index][len(P_d)+bt_index]+=fisher_list[p][0][1]
#         fisher_tot[len(P_d)+bt_index][bd_index]+=fisher_list[p][1][0]
#
#         fisher_tot[bd_index][len(P_d)+len(P_t)+bs_index]+=fisher_list[p][0][2]
#         fisher_tot[len(P_d)+len(P_t)+bs_index][bd_index]+=fisher_list[p][2][0]
#
#         fisher_tot[len(P_d)+bt_index][len(P_d)+len(P_t)+bs_index]+=fisher_list[p][1][2]
#         fisher_tot[len(P_d)+len(P_t)+bs_index][len(P_d)+bt_index]+=fisher_list[p][2][1]
#     # print(fisher_tot)
#     # print 'fisher_tot_shape',np.shape(fisher_tot)
#     sigma_tot=np.linalg.inv(fisher_tot).T
#     return(sigma_tot)

def delta_beta_matrix_making(P_d,P_t,P_s,input_beta_zeroth_iteration): #computes the spatial delta over the betas in a patch
    ind=0
    pixel_number1=hp.nside2npix(nside)
    beta_d_list=[]
    beta_t_list=[]
    map_bs=[None]*pixel_number1

    delta_beta=[]

    for i in range(len(P_d)):
        beta_d_list=[]
        for j in range(len(P_d[i])):
            beta_d_list.append(input_beta_zeroth_iteration[j])
            # print(beta_d_list)
        delta_beta.append(np.std(beta_d_list))
        # print(np.std(beta_d_list))
    # print(delta_beta)
    for i in range(len(P_t)):
        beta_t_list=[]
        for j in range(len(P_t[i])):
            beta_t_list.append(input_beta_zeroth_iteration[pixel_number+j])
            # print(beta_t_list)
        delta_beta.append(np.std(beta_t_list))
        # print(np.std(beta_t_list))
    # print(delta_beta)
    # print('pixel_numbeazeazertzze=',pixel_number)
    for i in range(len(P_s)):
        beta_s_list=[]
        # print(P_s)
        for j in range(len(P_s[i])):
            # print('index=',2*pixel_number+P_s[i][j])
            # print(pixel_number,np.shape(P_s[i]))
            # print(np.shape(input_beta_zeroth_iteration))
            beta_s_list.append(input_beta_zeroth_iteration[2*pixel_number+j])
        delta_beta.append(np.std(beta_s_list))
    # print(delta_beta)
    return(np.diag(delta_beta))

def matrix_comparison(delta_beta,sigma_matrix): #compares the delta matrix with the sigma matrix
    delta_beta2=np.dot(delta_beta,delta_beta)
    norm=np.linalg.norm(np.absolute(delta_beta2-sigma_matrix),1) #%! is the determinant the right kind of norm to choose ?
    return norm

def patch_making(input_beta_zeroth_iteration,sigma,P_d,P_t,P_s):   # patch making function using the "addition method" takes as arguments:
    #the beta of the first iteration to create the histogram, the sigma matrix of the last iteration and the patches used in the last itaration
    #it returns the new patches and the beta for each patch (computed using the mean of the boundaries for now)
    min_beta_d=np.amin(input_beta_zeroth_iteration[:pixel_number])
    min_beta_t=np.amin(input_beta_zeroth_iteration[pixel_number:2*pixel_number])
    min_beta_s=np.amin(input_beta_zeroth_iteration[2*pixel_number:3*pixel_number])
    beta_dust_slice=[np.nan]*pixel_number
    temp_dust_slice=[np.nan]*pixel_number
    beta_sync_slice=[np.nan]*pixel_number
    set_of_beta_slice=[]
    P_d_new=[]
    P_t_new=[]
    P_s_new=[]

    temp_list2=[]
    #first try with addition method
    inf_boundary_d=min_beta_d
    sup_boundary_d=min_beta_d+np.sqrt(sigma[0][0])

    #-------------------------------------------------P_d CONSTRUCTION----------------------------------------------------

    for j in range(len(P_d)-1):
        for i in range(pixel_number):
            if input_beta_zeroth_iteration[i]>=inf_boundary_d and input_beta_zeroth_iteration[i]<sup_boundary_d:
                beta_dust_slice[i]=(inf_boundary_d+sup_boundary_d)/2
                temp_list2.append(pixel_list[i])
        if len(temp_list2) >= 1:
            P_d_new.append(temp_list2)
            set_of_beta_slice.append((inf_boundary_d+sup_boundary_d)/2)
        temp_list2=[]
        inf_boundary_d=inf_boundary_d+np.sqrt(sigma[j][j])
        sup_boundary_d=sup_boundary_d+np.sqrt(sigma[j+1][j+1])
    # %! what happens if there is too much or not enough pixels ?
    #-> not enough: by the way it has been constructed if there is too much bins nothing will happen, ie bins will stay empty and hence won't appear in beta_dust_slice
    #if there is too much pixel then something needs to be done:
    temp_list2=[]
    for i in range(pixel_number):
        if np.isnan(beta_dust_slice[i]):
            temp_list2.append(pixel_list[i])
    if len(temp_list2) >=1:
        print('not enough patches for P_d')
        P_d_new.append(temp_list2)
        index_list=[]
        for j in range(len(temp_list2)):
            index_list.append(pixel_list.index(temp_list2[j]))
        min_beta_d_in_list=np.amin(np.take(input_beta_zeroth_iteration,index_list))
        max_beta_d_in_list=np.amax(np.take(input_beta_zeroth_iteration,index_list))
        set_of_beta_slice.append((min_beta_d_in_list+max_beta_d_in_list)/2)
        index_list=[]
    print('np.shape(P_d)=',np.shape(P_d))
    print('pixel_number=',pixel_number)
    #-------------------------------------------------P_t CONSTRUCTION----------------------------------------------------

    temp_list2=[]
    inf_boundary_t=min_beta_t
    sup_boundary_t=min_beta_t+np.sqrt(sigma[len(P_d)][len(P_d)])
    for j in range(len(P_t)-1):
        for i in range(pixel_number):
            if input_beta_zeroth_iteration[i+pixel_number]>=inf_boundary_t and input_beta_zeroth_iteration[i+pixel_number]<sup_boundary_t:
                temp_dust_slice[i]=(inf_boundary_t+sup_boundary_t)/2
                temp_list2.append(pixel_list[i])
        if len(temp_list2) >= 1:
            P_t_new.append(temp_list2)
            set_of_beta_slice.append((inf_boundary_t+sup_boundary_t)/2)
        temp_list2=[]
        inf_boundary_t=inf_boundary_t+np.sqrt(sigma[j+len(P_d)][j+len(P_d)])
        sup_boundary_t=sup_boundary_t+np.sqrt(sigma[j+1+len(P_d)][j+1+len(P_d)])
    temp_list2=[]
    for i in range(pixel_number):
        if np.isnan(temp_dust_slice[i]):
            temp_list2.append(pixel_list[i])

    if len(temp_list2) >=1:
        print('not enough patches for P_t')
        P_t_new.append(temp_list2)
        temp_list3=[]
        index_list=[]
        for j in range(len(temp_list2)):
            index_list.append(pixel_list.index(temp_list2[j]))
            temp_list3.append(index_list[j]+pixel_number)
        min_beta_t_in_list=np.amin(np.take(input_beta_zeroth_iteration,temp_list3))
        max_beta_t_in_list=np.amax(np.take(input_beta_zeroth_iteration,temp_list3))
        set_of_beta_slice.append((min_beta_t_in_list+max_beta_t_in_list)/2)
        index_list=[]

        del temp_list3
    print('np.shape(P_t)=',np.shape(P_t))
    print('pixel_number=',pixel_number)
    #-------------------------------------------------P_s CONSTRUCTION----------------------------------------------------
    temp_list2=[]
    inf_boundary_s=min_beta_s
    sup_boundary_s=min_beta_s+np.sqrt(sigma[len(P_d)+len(P_t)][len(P_d)+len(P_t)])
    for j in range(len(P_s)-1):
        for i in range(pixel_number):
            if input_beta_zeroth_iteration[i+2*pixel_number]>=inf_boundary_s and input_beta_zeroth_iteration[i+2*pixel_number]<sup_boundary_s:
                beta_sync_slice[i]=(inf_boundary_s+sup_boundary_s)/2
                temp_list2.append(pixel_list[i])
        if len(temp_list2) >= 1:
            P_s_new.append(temp_list2)
            set_of_beta_slice.append((inf_boundary_s+sup_boundary_s)/2)
        temp_list2=[]

        inf_boundary_s=inf_boundary_s+np.sqrt(sigma[j+len(P_d)+len(P_t)][j+len(P_d)+len(P_t)])
        sup_boundary_s=sup_boundary_s+np.sqrt(sigma[j+1+len(P_d)+len(P_t)][j+1+len(P_d)+len(P_t)])

    temp_list2=[]
    for i in range(pixel_number):
        if np.isnan(beta_sync_slice[i]):
            temp_list2.append(pixel_list[i])
            # print i
    if len(temp_list2) >=1:
        print('not enough patches for P_s')
        P_s_new.append(temp_list2)
        temp_list3=[]
        index_list=[]
        for j in range(len(temp_list2)):
            index_list.append(pixel_list.index(temp_list2[j]))
            temp_list3.append(index_list[j]+2*pixel_number)
        min_beta_s_in_list=np.amin(np.take(input_beta_zeroth_iteration,temp_list3))
        max_beta_s_in_list=np.amax(np.take(input_beta_zeroth_iteration,temp_list3))
        print('inf_boundary_s_in_list',min_beta_s_in_list)
        print('sup_boundary_s_in_list',max_beta_s_in_list)
        set_of_beta_slice.append((min_beta_s_in_list+max_beta_s_in_list)/2)
        index_list=[]
        del temp_list3
    temp_list2=[]
    del temp_list2
    print('np.shape(P_s)=',np.shape(P_s))
    print('pixel_number=',pixel_number)
    return P_d_new,P_t_new,P_s_new,set_of_beta_slice

def iteration(iteration_number,input_beta_zeroth_iteration,P_d_new,P_t_new,P_s_new,set_of_beta_slice):
    print('iteration_number=',iteration_number)
    print('iteration_number shape=',iteration_number.shape)
    print('iteration_number[0]=',iteration_number[0])
    iteration_number=int(iteration_number[0])
    print('iteration_number_int=',iteration_number)
    for i in range(iteration_number):
        print('set_of_beta_slice=',set_of_beta_slice)
        minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,set_of_beta_slice,(fun,P_d_new,P_t_new,P_s_new,nside))

        sigma=sigma_matrix_making(minimization_result,P_d_new,P_t_new,P_s_new,freq_maps)
        # for j in range(len(sigma)):
        #     print(np.sqrt(sigma[j][j]))

        delta_b=delta_beta_matrix_making(P_d_new,P_t_new,P_s_new,input_beta_zeroth_iteration)
        # for j in range(len(delta_b)):
        #     print(np.sqrt(delta_b[j][j]))
        comp.append(matrix_comparison(delta_b,sigma))
        P_d_new,P_t_new,P_s_new,set_of_beta_slice=patch_making(input_beta_zeroth_iteration,sigma,P_d_new,P_t_new,P_s_new)
    print('set_of_beta_slice=',set_of_beta_slice)
    return comp[iteration_number]

def patch_making_mean(input_beta_zeroth_iteration,sigma,minimization_result,P_d,P_t,P_s):
    min_beta_d=np.amin(input_beta_zeroth_iteration[:pixel_number])
    min_beta_t=np.amin(input_beta_zeroth_iteration[pixel_number:2*pixel_number])
    min_beta_s=np.amin(input_beta_zeroth_iteration[2*pixel_number:3*pixel_number])
    beta_dust_slice=[np.nan]*pixel_number
    temp_dust_slice=[np.nan]*pixel_number
    beta_sync_slice=[np.nan]*pixel_number
    set_of_beta_slice=[]
    P_d_new=[]
    P_t_new=[]
    P_s_new=[]

    temp_list2=[]
    #first try with addition method
    inf_boundary_d=min_beta_d
    sup_boundary_d=min_beta_d+np.sqrt(sigma[0][0])

    #-------------------------------------------------P_d CONSTRUCTION----------------------------------------------------

    for j in range(len(P_d)):
        for i in range(pixel_number):
            if input_beta_zeroth_iteration[i]>minimization_result[j]-sigma[j][j]/2 and input_beta_zeroth_iteration[i]<minimization_result[j]+sigma[j][j]/2:
                beta_dust_slice[i]=minimization_result[j]
                temp_list2.append(i)
        if len(temp_list2) >= 1:
            P_d_new.append(temp_list2)
            set_of_beta_slice.append(minimization_result[j])
    for i in range(len(P_d)-1):
        for j in range(len(P_d[i])):
            for k in range(len(P_d[i+1])):
                if P_d_new[i][j]==P_d_new[i+1][k]:
                    temp_list4.append(P_d_new[i][j])

        #use remove !! le pixel ne sera present que une seule fois par liste anyway
        temp_list2=[]
    # %! what happens if there is too much or not enough pixels ?
    #-> not enough: by the way it has been constructed if there is too much bins nothing will happen, ie bins will stay empty and hence won't appear in beta_dust_slice
    #if there is too much pixel then something needs to be done:
    temp_list2=[]
    for i in range(pixel_number):
        if np.isnan(beta_dust_slice[i]):
            temp_list2.append(i)
    if len(temp_list2) >=1:
        P_d_new.append(temp_list2)
        min_beta_d_in_list=np.amin(np.take(input_beta_zeroth_iteration,temp_list2))
        max_beta_d_in_list=np.amax(np.take(input_beta_zeroth_iteration,temp_list2))
        set_of_beta_slice.append((min_beta_d_in_list+max_beta_d_in_list)/2)

    temp_list2=[]

    return 0

def return_fun(pixel,freq_maps,prewhiten_factors,A_ev):
    # print 'A_ev 3param',A_ev([1.5,20,-3])
    # print 'A_ev 2param',A_ev([1.5,-3])
    return lambda x: -fgal.logL(A_ev(x), np.transpose(freq_maps)[pixel], np.diag(prewhiten_factors**2 ) )


def return_jac(pixel,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB):
    return lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[pixel], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)





# def return_fun_half(pixel):
#     return lambda x: -(2.)*fgal.logL(A_ev(x), np.transpose(freq_maps)[pixel], np.diag(prewhiten_factors**2))

#--------------------------------------TEST------------------------------------------
#

# map_bd=freq_maps[0][1]*0
# map_bs=freq_maps[0][1]*0
# map_t=freq_maps[0][1]*0



#meta patch initialisation: one patch per pixel
#meta_P=[[i] for i in range(len(freq_maps[0][0]))]
# meta_P=[range(24),range(23,47),[47]]
# input_set_of_betas=[1.59,19.6,-3.1,1.59,19.6,-3.1,1.59,19.6,-3.1]


# print(meta_P)
# print(input_set_of_betas)



# for i in range(len(P_s)):
#     map_bs[P_s[i]]=-3.1
#     input_set_of_betas.append(-3.1)
# for i in range(len(P_t)):
#     map_t[P_t[i]]=19.6
#     input_set_of_betas.append(19.6)




# A_ev_test, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=prewhiten_factors)
#
# A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)
# prewhitened_data = prewhiten_factors * freq_maps.T
# print(prewhiten_factors.shape)
# print(freq_maps.T.shape)
# print('prewhitened_data test=',prewhitened_data.shape)



# fun, jac, last_values = fgal._build_bound_inv_logL_and_logL_dB(A_ev_test, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
# #fun prend en argument les beta sur le nombre de pixel considere dans prewhitened_data
# minimization_check=scipy.optimize.minimize(fun,[1.59,19.6,-3.1])
#
#print(minimization_check)
# data=data_patch(freq_maps,meta_P)
# print(joint_spectral_likelihood(input_set_of_betas,P_d,P_t,P_s,freq_maps))

if nside_comparison_for_fine_tuning!=True:
    map_beta_dust_pysm=hp.ud_grade(hp.read_map('../map_test/pysm_maps/dust_beta.fits', field=0),nside)
    map_temp_dust_pysm=hp.ud_grade(hp.read_map('../map_test/pysm_maps/dust_temp.fits', field=0),nside)
    map_beta_sync_pysm=hp.ud_grade(hp.read_map('../map_test/pysm_maps/sync_beta.fits', field=0),nside)
    map_beta_dust_pysm[masked_pixels]=None
    map_temp_dust_pysm[masked_pixels]=None
    map_beta_sync_pysm[masked_pixels]=None



zeroth_iteration=True
joint_likelihood_test=True
minimization_jsl=True

plot_histo_residu_normal_and_jsl=True
plot_joint_likelihood_test=True #f
plot_signal_to_noise=True #f
plot_histo_residu=True #f
likelihood_comparison=False
jac_comparison=False
plot_res_to_noise=True
likelihood_comparison_3comp=True

if nside_comparison_for_fine_tuning==True:
    if plot_joint_likelihood_test==True and minimization_jsl==True:
        fig_res_jsl, ax_res_jsl  = plt.subplots(2, 2,  tight_layout=True)

    if plot_histo_residu_normal_and_jsl==True and minimization_jsl==True:
        fig_res_norm_and_jsl , ax_res_norm_and_jsl = plt.subplots(2, 2,  tight_layout=True)

    if plot_histo_residu==True and zeroth_iteration==True:
        fig_res_norm , ax_res_norm = plt.subplots(2, 2,  tight_layout=True)

    if plot_signal_to_noise==True and zeroth_iteration==True:
        fig_signal_to_noise , ax_signal_to_noise = plt.subplots(2, 2,  tight_layout=True)

    if plot_signal_to_noise==True and zeroth_iteration==True:
        fig_res_to_noise , ax_res_to_noise = plt.subplots(2, 2,  tight_layout=True)


    if plot_signal_to_noise==True and joint_likelihood_test==True and minimization_jsl==True:
        fig_signal_to_noise_jsl , ax_signal_to_noise_jsl = plt.subplots(2, 2,  tight_layout=True)

    if plot_res_to_noise==True and joint_likelihood_test==True and minimization_jsl==True:
        fig_res_to_noise_jsl , ax_res_to_noise_jsl = plt.subplots(2, 2,  tight_layout=True)


if nside_comparison_for_fine_tuning==True:
    # fig, axs = plt.subplots(2, 2,  tight_layout=True)
    for nside in [2]:

        # fig_comp_sep, axs_comp_sep = plt.subplots(2, 2,  tight_layout=True)
        pysm_model='c1d1s1'
        sky = get_sky(nside, pysm_model)
        np.random.seed (seed=None)
        map_bd_list=[]
        map_bt_list=[]
        map_bs_list=[]
        map_bd_list_comp_sep=[]
        map_bt_list_comp_sep=[]
        map_bs_list_comp_sep=[]
        map_bd_list_jsl_test=[]
        map_bt_list_jsl_test=[]
        map_bs_list_jsl_test=[]


        for iter in range(1):
            pixel_number=hp.nside2npix(nside)

            npix=hp.nside2npix(nside)
            instrument = get_instrument(nside, 'litebird')
            self=instrument

            pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.Nside)) * (180. * 60. / np.pi) ** 2
            """sigma_pix_I/P is std of noise per pixel. It is an array of length
            equal to the number of input maps."""
            sigma_pix_I = np.sqrt(self.Sens_I ** 2 / pix_amin2)
            sigma_pix_P = np.sqrt(self.Sens_P ** 2 / pix_amin2)

            np.random.seed (seed=None)
            noise = np.random.randn(len(self.Sens_I), 3, npix)
            noise[:, 0, :] *= sigma_pix_I[:, None]
            noise[:, 1, :] *= sigma_pix_P[:, None]
            noise[:, 2, :] *= sigma_pix_P[:, None]


            freq_maps = instrument.observe(sky, write_outputs=False)[0] + noise
            freq_maps=freq_maps[:,1:,:]  #on retire la temperature (1:) car on ne prend en compte que Q&U pas I
            components = [CMB(), Dust(150.), Synchrotron(20.)]
            prewhiten_factors = _get_prewhiten_factors(instrument, freq_maps.shape)         # correspond a N^-1/2



            # hp.mollview(noise[0][0])
            # plt.savefig('../test_noise/noise_nside=%s_iter=%s.png'%(nside,iter))
            # plt.show()


            mask=hp.read_map('../map_test/HFI_Mask_GalPlane-apo2_2048_R2.00.fits',field=2)
            mask_bin=mask*0
            mask_bin=hp.ud_grade(mask_bin,nside)
            mask_bin[np.where(hp.ud_grade(mask,nside)!=0)[0]]=1
            masked_pixels=[np.where(hp.ud_grade(mask,nside)==0)[0]]
            print 'masked_pixels',masked_pixels
            """
            masking pixel #0 :
            """
            # masked_pixels[0]=np.append(masked_pixels[0],0)

            print 'masked_pixels',masked_pixels

            print(' number of masked_pixels',len(masked_pixels[0]))
            freq_maps_save=freq_maps[:][:][:]
            print 'freq_maps SHAPE', np.shape(freq_maps)
            print 'freq_maps', freq_maps[:][:][1]
            # sys.exit()
            # freq_maps_save=np.copy(freq_maps)
            if len(masked_pixels)!=0:
                freq_maps= np.zeros((len(freq_maps_save), len(freq_maps_save[0]),pixel_number-len(masked_pixels[0])))
                # print 'freq_maps test',freq_maps

                for i in range(len(freq_maps_save)):
                    for j in range(len(freq_maps_save[i])):
                        freq_maps[i][j]=np.delete(freq_maps_save[i][j],masked_pixels)

                # print '1',freq_maps
                # freq_maps=np.delete(freq_maps_save,masked_pixels,2)
                # print '2',freq_maps-freq_maps1
                # sys.exit()
                        # print(np.delete(freq_maps_save[i][j],masked_pixels))
                # freq_maps=np.transpose(freq_maps)
                print 'FREQ MAPS SHAPE',np.shape(freq_maps)
                pixel_list=[]
                # print('masked_pixels=',masked_pixels[1][0])
                temp=[]
                for j in range(pixel_number):
                    for i in range(len(masked_pixels[0])):
                        if j==masked_pixels[0][i]:
                            temp.append(j)
                    if len(temp)==0:
                        pixel_list.append(j)
                    temp=[]
                # print('pixel_list=',pixel_list)
                pixel_number=pixel_number-len(masked_pixels[0])
            else:
                pixel_list=range(pixel_number)

            print pixel_list
            # print '0',freq_maps[:][:][0]
            # print '1',freq_maps[:][:][1]
            # print '2',freq_maps[:][:][2]
            # sys.exit()
            #-----pixel 0 removal-----
            # pixel_list=pixel_list[1:]
            # pixel_number=pixel_number-1


            map_beta_dust_pysm=hp.ud_grade(hp.read_map('../map_test/pysm_maps/dust_beta.fits', field=0),nside)
            map_temp_dust_pysm=hp.ud_grade(hp.read_map('../map_test/pysm_maps/dust_temp.fits', field=0),nside)
            map_beta_sync_pysm=hp.ud_grade(hp.read_map('../map_test/pysm_maps/sync_beta.fits', field=0),nside)
            map_beta_dust_pysm[masked_pixels]=None
            map_temp_dust_pysm[masked_pixels]=None
            map_beta_sync_pysm[masked_pixels]=None





            P_d=[]
            P_s=[]
            P_t=[]
            # for i in range(len(freq_maps[0][1])): P_d.append([pixel_list[i]])
            # for i in range(len(freq_maps[0][1])): P_s.append([pixel_list[i]])
            # for i in range(len(freq_maps[0][1])): P_t.append([pixel_list[i]])
            for i in range(pixel_number): P_d.append([pixel_list[i]])
            for i in range(pixel_number): P_s.append([pixel_list[i]])
            for i in range(pixel_number): P_t.append([pixel_list[i]])

            input_set_of_betas=[]
            # for i in range(len(freq_maps[0][0])):
            #     input_set_of_betas.append(1.54)
            # for i in range(len(freq_maps[0][0])):
            #     input_set_of_betas.append(19.6)
            # for i in range(len(freq_maps[0][0])):
            #     input_set_of_betas.append(-3.1)
            for i in range(pixel_number):
                input_set_of_betas.append(1.54)
            for i in range(pixel_number):
                input_set_of_betas.append(19.6)
            for i in range(pixel_number):
                input_set_of_betas.append(-3.1)




            # input_set_of_betas_rand=[]
            # for i in range(len(freq_maps[0][0])):
            #     input_set_of_betas_rand.append(1.54+np.random.uniform(-1.54/10.0,1.54/10.0))
            # for i in range(len(freq_maps[0][0])):
            #     input_set_of_betas_rand.append(19.6+np.random.uniform(-19.6/10.0,19.6/10.0))
            # for i in range(len(freq_maps[0][0])):
            #     input_set_of_betas_rand.append(-3.1+np.random.uniform(3.1/10.0,-3.1/10.0))



            A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=None)#prewhiten_factors)
            print 'params',params
            print 'len params', len(params)
            # print 'params shape',np.shape(params)
            # x0=np.array([1.55,19.6,-3.1])

            if len(params)==3:
                print '3 PARAMS !!'
                jac_list=[]
                # bounds=((1.0, 5.0),(15,25),(-3.5,-2.0))
                if pysm_model=='c1d1s1':
                    bounds=((1.45,1.7),(1.0,100),(-7.5,0.5)) #GOOOOOD ONES FOR C1D1S1 DON'T TOUCH!!!!!!!!!!
                if pysm_model=='d0s0':
                    bounds=((0.5,2.5),(1.0,75),(-7.5,0.5)) # Not the best one yet, needs some more fine tuning
                # bounds=((0.1, 10.0),(0.1,100),(-10,-0.1))
                A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)

                # for p in range(pixel_number):
                    # jac_list.append(lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False))
                    # jac_list.append(return_jac(p,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB))
                    # if p==0:
                        # print 'A_ev([1.55,19.6,-3.1])',A_ev([1.55,19.6,-3.1])
                        # print 'np.transpose(freq_maps)[0]',np.transpose(freq_maps)[0]
                        # print 'np.diag(prewhiten_factors**2)',np.diag(prewhiten_factors**2)
                        # print 'A_dB_ev',A_dB_ev([1.55,19.6,-3.1])
                        # print 'comp_of_dB',comp_of_dB
                # jac_list=[None]*hp.nside2npix(nside)
                # for p in pixel_list:
                #     jac_list[p]=lambda x :fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[pixel_list.index (p)], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)
                # sys.exit()


                # print 'dpixel_nulber=',np.transpose(freq_maps)[pixel_number-1]
                # print 'd0=',np.transpose(freq_maps)[0]
                # print 'jac0 =',jac_list[0]([1.54,19.6,-3.1])
                # print 'd50=',np.transpose(freq_maps)[50]
                # print 'jac50 =',jac_list[50]([1.54,19.6,-3.1])
                # print 'jac50 =',jac_list[50]([1.54,19.6,-3.1])[0]
                # sys.exit()

                fun=[None]*hp.nside2npix(nside)
                last_values=[None]*hp.nside2npix(nside)
                for l in range(pixel_number):
                    fun[pixel_list[l]] = return_fun(l,freq_maps,prewhiten_factors,A_ev)
                    jac_list.append(return_jac(l,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB))
                    # if l==2:
                    #     sys.exit()
                # print 'fun0 =',fun[pixel_list[0]]([1.54,19.6,-3.1])
                # print 'fun50 =',fun[pixel_list[50]]([1.54,19.6,-3.1])

                # fig = plt.figure()
                # ax3D = fig.add_subplot(111, projection='3d')
                pixel_test=0
                # beta_d_range=np.arange(1.5, 1.6, 0.1/1000.)
                # temp_d_range=np.arange(19,21,2./10.)
                # beta_s_range=np.arange(-4,-2, 2./10.)
                #
                # X1,Y1=np.meshgrid(beta_d_range,temp_d_range)
                # X2,Z2=np.meshgrid( beta_d_range,beta_s_range)
                # Y3,Z3=np.meshgrid(temp_d_range,beta_s_range)
                #
                # funxy=[[fun[pixel_test]([x,y,-3]) for x in beta_d_range] for y in temp_d_range  ]
                # # fig = plt.figure()
                # # ax3D = fig.add_subplot(111, projection='3d')
                #
                #
                #
                # X, Y = np.meshgrid(beta_d_range, temp_d_range)
                # # Z = [[fun[pixel_test]([x,y,-3]) for x in beta_d_range] for y in temp_d_range  ]
                # Z= np.zeros((len(beta_d_range),len(temp_d_range)))
                # Zj= np.zeros((len(beta_d_range),len(temp_d_range)))
                # for x in range(len(beta_d_range)):
                #     for y in range(len(temp_d_range)):
                #         Z[x,y]=fun[pixel_test]([beta_d_range[x],temp_d_range[y],-3.1])
                #         Zj[x,y]=jac_list[pixel_test]([beta_d_range[x],temp_d_range[y],-3.1])[0]
                #
                #
                # # plt.plot(beta_d_range,Z)
                # # plt.show()
                # # plt.plot(beta_d_range,Zj)
                # # plt.show()
                #
                # fun_list=np.zeros(len(beta_d_range))
                # for x in range(len(beta_d_range)):
                #     fun_list[x]=fun[pixel_test]([beta_d_range[x],20,-3])
                #
                # max_fun=max(fun_list)
                # fun_list_plot=np.zeros(len(beta_d_range))
                # for x in range(len(beta_d_range)):
                #     fun_list_plot[x]=np.exp(fun_list[x]-max_fun)
                # plt.plot(beta_d_range,fun_list_plot)
                # plt.show()

                # plt.title('fun of beta d & temp')
                # plt.contourf(X, Y, Z,100)
                # plt.colorbar()
                # plt.show()
                #
                # # fig = plt.figure()
                # # ax3D = fig.add_subplot(111, projection='3d')
                #
                # X, Y = np.meshgrid(beta_d_range, beta_s_range)
                # Z = [[fun[pixel_test]([x,20,z]) for x in beta_d_range] for z in beta_s_range ]
                # plt.title('fun of beta d & beta s')
                # plt.contourf(X, Y, Z, 100)
                # plt.colorbar()
                # plt.show()
                #
                #
                #
                # # fig = plt.figure()
                # # ax3D = fig.add_subplot(111, projection='3d')
                # X, Y = np.meshgrid(temp_d_range, beta_s_range)
                # Z = [[fun[pixel_test]([1.5,y,z]) for y in temp_d_range] for z in beta_s_range ]
                # plt.title('fun of temp & beta s')
                # plt.contourf(X, Y, Z, 100)
                # plt.colorbar()
                # plt.show()


                # funxy=fun[pixel_test]([beta_d_range , temp_d_range , -3])
                # print funxy
                # funxz=[[fun[pixel_test]([x,20,z]) for x in beta_d_range] for z in beta_s_range ]
                # funyz=[[fun[pixel_test]([1.5,y,z]) for y in temp_d_range] for z in beta_s_range ]
                # print np.shape(np.array(funxy))
                # print np.shape(beta_d_range)
                # # print funxy
                # fig = plt.figure()
                # ax3D = fig.add_subplot(111, projection='3d')
                # ax3D.plot_surface(beta_d_range,temp_d_range,np.array(funxy),label='fun of beta d & temp')
                # plt.show()
                # fig = plt.figure()
                # ax3D = fig.add_subplot(111, projection='3d')
                # ax3D.plot_surface(beta_d_range,beta_s_range,np.array(funxz),label='fun of beta d & beta s')
                # plt.show()
                # fig = plt.figure()
                # ax3D = fig.add_subplot(111, projection='3d')
                # ax3D.plot_surface(temp_d_range,beta_s_range,np.array(funyz),label='fun of temp & beta s')
                # plt.show()

                # sys.exit()
                if zeroth_iteration==True:
                    comp=[]
                    minimization_result_pixel=[]
                    input_beta_zeroth_iteration=[]
                    input_beta_zeroth_iteration_comp_sep=[]
                    comp_sep_result_pixel=[]
                    # x0=np.array([1.55,19.6,-3])
                    # print x0
                    # res= fgal.comp_sep(A_ev, np.transpose(freq_maps)[0], np.diag(prewhiten_factors**2 ), A_dB_ev, comp_of_dB,x0)
                    #
                    # print 'res comp sep=',res.x
                    # print 'fun pix 0=',fun[0](x0)
                    # print 'jac pix0=',jac_list[0](x0)
                    # print 'jac pix0 inv=', [1/jac_list[0](x0)[0],1/jac_list[0](x0)[1],1/jac_list[0](x0)[2]]
                    # fun_build, jac_build, last_values_build = fgal._build_bound_inv_logL_and_logL_dB(A_ev, np.transpose(freq_maps)[0], np.diag(prewhiten_factors**2 ), A_dB_ev, comp_of_dB)
                    # print 'jac build=',jac_build(x0)
                    for p in range(pixel_number):
                        # print 'MINIMIZATION !!!!!'

                        # def fun_print(x):
                        #     print (x)
                        #     return fun[pixel_list[p]](x)

                        # minimization_result_pixel.append(scipy.optimize.minimize(fun_print,np.array([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]]),\
                        # bounds=bounds, tol=1e-18, jac=jac_list[p],method='L-BFGS-B', options = {'disp':True,\
                        # 'stepmx':0.01,\
                        # 'eps': 1e-7,\
                        #  'ftol':1e-12,\
                        # 'xtol':1e-12,\
                        # 'gtol':1e-12,\
                        # 'maxiter': 10000
                        # }).x)
                        # print 'jac at [1.54,20,-3]',jac_list[0]([1.54,20,-3])
                        # minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],np.array([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]]),\
                        # jac=jac_list[p], bounds=bounds, tol=1e-18).x)

                        minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],np.array([input_set_of_betas[p],input_set_of_betas[p+len(P_d)],input_set_of_betas[p+len(P_d)+len(P_t)]]),\
                        jac=jac_list[p], tol=1e-18,bounds=bounds).x)

                        # bounds=((1.0, 5.0),(15,25),(-3.5,-2.0))
                        # iter_while=0
                        # while minimization_result_pixel[-1][0]<=5.0 or minimization_result_pixel[-1][0]>=1.0 or minimization_result_pixel[-1][1]<=15.0 or minimization_result_pixel[-1][1]>=25 or minimization_result_pixel[-1][2]<=-3.5 or minimization_result_pixel[-1][2]>=-2.0:
                        #     print 'wrong pixel=',pixel_list[p]
                        #     print 'wrong minimization=',minimization_result_pixel[-1]
                        #     new_init=np.array([input_set_of_betas[p]+np.random.uniform(-input_set_of_betas[p]/10.0,input_set_of_betas[p]/10.0) , input_set_of_betas[p+pixel_number]+np.random.uniform(-input_set_of_betas[p+pixel_number]/10.0 , input_set_of_betas[p+pixel_number]/10.0),\
                        #     input_set_of_betas[p+2*pixel_number]+np.random.uniform(input_set_of_betas[p+2*pixel_number]/10.0,-input_set_of_betas[p+2*pixel_number]/10.0)])
                        #     print 'new_init=',new_init
                        #     minimization_result_pixel[-1]=scipy.optimize.minimize(fun[pixel_list[p]],new_init,\
                        #     jac=jac_list[p], tol=1e-18).x
                        #
                        #     iter_while+=1
                        #     if iter_while==10:
                        #         print 'too much loop'
                        #
                        #         break

                        if minimization_result_pixel[-1][1]>=1e6:
                            # beta_d_range=np.arange(1.5, 1.6, 0.1/1000.)
                            # beta_t_range=np.arange(1e5,1e9,10000)
                            # beta_t_range=np.arange(10,150,140/1000.)
                            beta_t_range=np.logspace(0.0,8,100)
                            print 'beta_t_range',beta_t_range
                            # beta_t_range=np.arange(minimization_result_pixel[-1][1]-100,minimization_result_pixel[-1][1]+100,200/1000.)
                            # temp_d_range=np.arange(19,21,2./10.)
                            print 'minimization_result_pixel[-1]',minimization_result_pixel[-1]
                            print 'fun',fun[pixel_list[p]](minimization_result_pixel[-1])
                            # beta_s_range=np.arange(-4,-2, 2./10.)
                            fun_list=np.zeros(len(beta_t_range))
                            for x in range(len(beta_t_range)):
                                fun_list[x]=(-1./2.)*fun[pixel_list[p]]([1.54,beta_t_range[x],-3.1])

                            max_fun=max(fun_list)
                            fun_list_plot=np.zeros(len(beta_t_range))
                            for x in range(len(beta_t_range)):
                                fun_list_plot[x]=np.exp(fun_list[x]-max_fun)

                            plt.plot(beta_t_range,fun_list_plot)
                            plt.xscale('log')
                            plt.show()
                            sys.exit()

                        # comp_sep_result_pixel.append(fgal.comp_sep(A_ev, np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2 ), A_dB_ev, comp_of_dB,x0).x)
                        # minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],np.array([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]])).x)
                        # print 'minimization_result_pixel',minimization_result_pixel
                        # print 'fun pix 0 (miniresults)=',fun[0](minimization_result_pixel[0])
                        # print 'fun pix 0=',fun[0]([1.54,20,-3])
                        # print 'jac_list[p](init)',jac_list[p]([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]])
                        # print 'jac_build(init)',jac_build([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]])

                        # sys.exit()
                    for i in range(len(P_d)):
                        input_beta_zeroth_iteration.append(minimization_result_pixel[i][0])
                        # print comp_sep_result_pixel[i][0]
                        # input_beta_zeroth_iteration_comp_sep.append( comp_sep_result_pixel[i][0] )
                    for i in range(len(P_t)):
                        input_beta_zeroth_iteration.append(minimization_result_pixel[i][1])
                        # input_beta_zeroth_iteration_comp_sep.append(comp_sep_result_pixel[i][1])
                    for i in range(len(P_s)):
                        input_beta_zeroth_iteration.append(minimization_result_pixel[i][2])
                        # input_beta_zeroth_iteration_comp_sep.append(comp_sep_result_pixel[i][2])
                    minimization_result_pixel=[]
                    comp_sep_result_pixel=[]

                    map_bd,map_bt,map_bs=patch_map(input_beta_zeroth_iteration,P_d,P_t,P_s,nside)
                    # map_bd_comp_sep,map_bt_comp_sep,map_bs_comp_sep=patch_map(input_beta_zeroth_iteration_comp_sep,P_d,P_t,P_s,nside)

                    # print 'fun pixel_test', fun[pixel_test]([map_bd[pixel_test],map_bt[pixel_test],map_bs[pixel_test]])
                    # print 'betas pixel_test',[map_bd[pixel_test],map_bt[pixel_test],map_bs[pixel_test]]
                    # print 'map_bt',map_bt
                    # sys.exit()
                    # print 'map_bt',map_bt
                    fisher_list,s_list,nnt_list,fisher_s_list,fisher_n_list,fisher_hess_list=fisher_pixel(map_bd,map_bt,map_bs,fun,pixel_list,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB,Hessian=True)

                    map_bd_list.append(map_bd)
                    map_bt_list.append(map_bt)
                    map_bs_list.append(map_bs)

                    # map_bd_list_comp_sep.append(map_bd_comp_sep)
                    # map_bt_list_comp_sep.append(map_bt_comp_sep)
                    # map_bs_list_comp_sep.append(map_bs_comp_sep)

                if joint_likelihood_test==True:

                    # def return_jac_super_list(A_ev,freq_maps,prewhiten_factors,A_dB_ev,comp_of_dB,input_set_of_betas,P_d,P_t,P_s,nside,pixel_list,pixel_number):
                    #     jac_super_list=[]
                    #     for p in range(len(P_d)):
                    #         jac_super_list.append(lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)[0])
                    #     for p in range(len(P_t)):
                    #         jac_super_list.append(lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)[1])
                    #     for p in range(len(P_s)):
                    #         jac_super_list.append(lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)[2])
                    #     # map_bd_init,map_bt_init,map_bs_init=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)
                    #
                    #     print [jac([input_set_of_betas[jac_super_list.index(jac)%pixel_number],input_set_of_betas[pixel_number+jac_super_list.index(jac)%pixel_number],input_set_of_betas[2*pixel_number+jac_super_list.index(jac)%pixel_number]]) for jac in jac_super_list]
                    #
                    #     return lambda y : [ jac( [ y[jac_super_list.index(jac)%len(P_d)] , y[pixel_number+jac_super_list.index(jac)%len(P_t)] , y[2*pixel_number+jac_super_list.index(jac)%len(P_t)] ] )  for jac in jac_super_list ]
                    import time
                    def return_jac_super_list(input_set_of_betas,fun,P_d,P_t,P_s,pixel_list,nside,jac): # /!\!!!!!!  NE MARCHE QUE DANS CE CAS PRECIS OU IL Y A AUTANT D'ELEMENT DANS P_i QUE DE PIXEL !!!!
                        map_bd,map_bt,map_bs=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)
                        # jac_super_list=[]
                        # for p in range(pixel_number):
                        #     # jac_super_list.append(lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)[0])
                        #
                        #     # start = time.time()
                        #     jac_super_list.append( return_jac(p,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                        #                                 ( [input_set_of_betas[p] , input_set_of_betas[ len(P_d) + p ] , input_set_of_betas[len(P_d)+len(P_t) + p ]] ) [0] )
                        # #     # end=time.time()
                        # #     # print 'time in loop#%s ='%p,end - start
                        # #
                        # for p in range(len(P_t)):
                        # #     # jac_super_list.append(lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)[1])
                        #     jac_super_list.append( return_jac(p,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                        #                                 ( [input_set_of_betas[p] , input_set_of_betas[ len(P_d) + p ] , input_set_of_betas[len(P_d)+len(P_t)  + p ]] ) [1] )
                        # for p in range(len(P_s)):
                        # #     # jac_super_list.append(lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[p], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)[2])
                        #     jac_super_list.append( return_jac(p,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                        #                                 ( [input_set_of_betas[p] , input_set_of_betas[ len(P_d) + p ] , input_set_of_betas[len(P_d)+len(P_t)  + p ]] ) [2] )
                        # map_bd_init,map_bt_init,map_bs_init=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)

                        # print [jac([input_set_of_betas[jac_super_list.index(jac)%pixel_number],input_set_of_betas[pixel_number+jac_super_list.index(jac)%pixel_number],input_set_of_betas[2*pixel_number+jac_super_list.index(jac)%pixel_number]]) for jac in jac_super_list]

                        # return np.array([ jac( [ input_set_of_betas[jac_super_list.index(jac)%len(P_d)] , input_set_of_betas[pixel_number+jac_super_list.index(jac)%len(P_t)] , \
                        #                             input_set_of_betas[2*pixel_number+jac_super_list.index(jac)%len(P_t)] ] )  for jac in jac_super_list ])


                        jac_super_temp=0
                        jac_super_list=[]
                        for patch in range(len(P_d)):
                            for p in P_d[patch]:
                                jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                                            ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [0]
                            jac_super_list.append(jac_super_temp)
                            jac_super_temp=0
                        for patch in range(len(P_t)):
                            for p in P_t[patch]:
                                jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                                            ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [1]
                            jac_super_list.append(jac_super_temp)
                            jac_super_temp=0
                        for patch in range(len(P_s)):
                            for p in P_s[patch]:
                                jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                                            ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [2]
                            jac_super_list.append(jac_super_temp)
                            jac_super_temp=0


                        return np.array(jac_super_list)


                    # super_jac = return_jac_super_list(A_ev,freq_maps,prewhiten_factors,A_dB_ev,comp_of_dB,input_set_of_betas,P_d,P_t,P_s,nside,pixel_list,pixel_number)

                    input_true_pysm_beta=[]
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_beta_dust_pysm[l])
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_temp_dust_pysm[l])
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_beta_sync_pysm[l])
                    jac=[]
                    print 'input_true_pysm_beta',input_true_pysm_beta
                    # sys.exit()

                    import itertools

                    point_number=250
                    list_min_khi_jsl_dust=[]
                    list_min_khi_jsl_temp=[]
                    list_min_khi_jsl_sync=[]
                    beta_d_range=np.linspace(0.5,2.5,point_number)
                    beta_t_range=np.linspace(15,35,point_number)
                    beta_s_range=np.linspace(-4.0,-2.0,point_number)
                    khi_list_jsl_dust=np.zeros(point_number)
                    khi_list_jsl_temp=np.zeros(point_number)
                    khi_list_jsl_sync=np.zeros(point_number)
                    print pixel_list
                    for patch in range(len(P_d)):
                        for x in range(point_number):
                            # print 'input =',list(itertools.chain.from_iterable([input_true_pysm_beta[:patch],[beta_d_range[x]],input_true_pysm_beta[patch:]]))
                            # if patch ==0:
                            #     khi_list_jsl_dust[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas[:patch+1],[beta_d_range[x]],input_set_of_betas[patch+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)
                            #     print 'khi',khi_list_jsl_dust[x]
                                # khi_list_jsl_dust[x]=0
                            # else:
                            # print 'fun,P_d,P_t,P_s,pixel_list,nside,jac',fun,P_d,P_t,P_s,pixel_list,nside,jac
                            khi_list_jsl_dust[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_beta_zeroth_iteration[:patch],[beta_d_range[x]],input_beta_zeroth_iteration[patch+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                            # print 'khi',khi_list_jsl_dust[x]
                            # if x==0:
                                # print 'fun,P_d,P_t,P_s,pixel_list,nside,jac',fun,P_d,P_t,P_s,pixel_list,nside,jac
                        # print 'khi_list_jsl_dust',khi_list_jsl_dust
                        # print np.argmin(khi_list_jsl_dust)

                        list_min_khi_jsl_dust.append(beta_d_range[np.argmin(khi_list_jsl_dust)])
                        print list_min_khi_jsl_dust[-1]
                        khi_list_jsl_dust=np.zeros(point_number)
                        # sys.exit()
                    # for x in range(point_number):
                    #     patch=0
                    #     khi_list_jsl_dust[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_beta_zeroth_iteration[:patch],[beta_d_range[x]],input_beta_zeroth_iteration[patch:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)
                    # print beta_d_range[np.argmin(khi_list_jsl_dust)]
                    # list_min_khi_jsl_dust[0]=beta_d_range[np.argmin(khi_list_jsl_dust)]
                    # khi_list_jsl_dust=np.zeros(point_number)
                        # print np.argmin(khi_list_jsl_dust)
                        # if patch==2:
                        #     sys.exit()
                    for patch in range(len(P_t)):
                        for x in range(point_number):
                            khi_list_jsl_temp[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_beta_zeroth_iteration[:len(P_d)+patch],[beta_t_range[x]],input_beta_zeroth_iteration[len(P_d)+patch+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)
                        list_min_khi_jsl_temp.append(beta_t_range[np.argmin(khi_list_jsl_temp)])
                        khi_list_jsl_temp=np.zeros(point_number)
                    for patch in range(len(P_s)):
                        for x in range(point_number):
                            khi_list_jsl_sync[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_beta_zeroth_iteration[:len(P_d)+len(P_t)+patch],[beta_s_range[x]],input_beta_zeroth_iteration[len(P_d)+len(P_t)+patch+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)
                        list_min_khi_jsl_sync.append(beta_s_range[np.argmin(khi_list_jsl_sync)])
                        khi_list_jsl_sync=np.zeros(point_number)

                    init_jsl = list_min_khi_jsl_dust + list_min_khi_jsl_temp + list_min_khi_jsl_sync
                    print 'init jsl=',init_jsl
                    # print 'init jsl shape=',np.shape(init_jsl)
                    # print 'input shape',np.shape(input_set_of_betas)
                    # sys.exit()

                    map_bd_arg,map_bt_arg,map_bs_arg=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)
                    start = time.time()
                    jac0=return_jac_super_list(input_set_of_betas,fun,P_d,P_t,P_s,pixel_list,nside,jac)
                    end=time.time()
                    print 'time super_jac exec',end - start
                    # sys.exit()

                    print 'super_jac true',return_jac_super_list(input_true_pysm_beta,fun,P_d,P_t,P_s,pixel_list,nside,jac)



                    # sys.exit()
                    super_bounds=[]
                    if pysm_model=='c1d1s1':
                        # bounds=((1.45,1.7),(1.0,100),(-7.5,0.5)) #GOOOOOD ONES FOR C1D1S1 DON'T TOUCH!!!!!!!!!!
                        for p in range(pixel_number):
                            super_bounds.append((1.45,1.7))
                            # super_bounds.append((1.45,1.65))
                        for p in range(pixel_number):
                            super_bounds.append((1.0,100))
                            # super_bounds.append((10.0,100))
                        for p in range(pixel_number):
                            super_bounds.append((-7.5,0.5))

                    # print 'return_jac_super_list', return_jac_super_list(input_true_pysm_beta,fun,P_d,P_t,P_s,pixel_list,nside)
                    # sys.exit()
                    if minimization_jsl==True:
                        input_set_of_betas_jsl_test=scipy.optimize.minimize(joint_spectral_likelihood,init_jsl,(fun,P_d,P_t,P_s,pixel_list,nside,return_jac_super_list), tol=1e-18,bounds=super_bounds,jac=return_jac_super_list)
                        print 'minimization',input_set_of_betas_jsl_test
                        input_set_of_betas_jsl_test=input_set_of_betas_jsl_test.x
                        map_bd_jsl_test,map_bt_jsl_test,map_bs_jsl_test=patch_map(input_set_of_betas_jsl_test,P_d,P_t,P_s,nside)
                        # sys.exit()
                        print 'super_jac minimization',return_jac_super_list(input_set_of_betas_jsl_test,fun,P_d,P_t,P_s,pixel_list,nside,jac)
                        map_bd_list_jsl_test.append(map_bd_jsl_test)
                        map_bt_list_jsl_test.append(map_bt_jsl_test)
                        map_bs_list_jsl_test.append(map_bs_jsl_test)

                if likelihood_comparison==True and minimization_jsl==True:
                    figl, axl = plt.subplots(2,  tight_layout=True)

                    beta_t_range=np.linspace(15,35,500)
                    fun_list_normal=np.zeros(len(beta_t_range))
                    fun_list_jsl=np.zeros(len(beta_t_range))
                    khi_list_normal=np.zeros(len(beta_t_range))
                    khi_list_jsl=np.zeros(len(beta_t_range))

                    input_true_pysm_beta=[]
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_beta_dust_pysm[l])
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_temp_dust_pysm[l])
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_beta_sync_pysm[l])

                    pixel=np.argmax(np.abs ( np.array(input_set_of_betas_jsl_test[len(P_d):len(P_d)+len(P_t)]) - np.array(input_true_pysm_beta[len(P_d):len(P_d)+len(P_t)])) )
                    # pixel=np.argmin( np.array(input_set_of_betas_jsl_test[len(P_d):len(P_d)+len(P_t)]) )
                    print 'pixel=',pixel_list[pixel]



                    print '[map_bd[pixel_list[pixel]],map_bt[pixel_list[pixel]],map_bs[pixel_list[pixel]]]',[map_bd[pixel_list[pixel]],map_bt[pixel_list[pixel]],map_bs[pixel_list[pixel]]]
                    print '[input_set_of_betas_jsl_test[pixel],input_set_of_betas_jsl_test[pixel_number+pixel],input_set_of_betas_jsl_test[2*pixel_number+pixel]',[input_set_of_betas_jsl_test[pixel],input_set_of_betas_jsl_test[len(P_d)+pixel],input_set_of_betas_jsl_test[len(P_d)+len(P_t) +pixel]]
                    print 'true betas pixel=',[map_beta_dust_pysm[pixel_list[pixel]] , map_temp_dust_pysm[pixel_list[pixel]] , map_beta_sync_pysm[pixel_list[pixel]] ]

                    for x in range(len(beta_t_range)):
                        # fun_list_normal[x]=(-1./2.)*fun[pixel_list[pixel]]([map_bd[pixel_list[pixel]],beta_t_range[x],map_bs[pixel_list[pixel]]])
                        # fun_list_jsl[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_jsl_test[:pixel_number+pixel],[beta_t_range[x]],input_set_of_betas_jsl_test[pixel_number+pixel:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        # fun_list_normal[x]=(-1./2.)*fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])
                        # fun_list_jsl[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:pixel_number+pixel],[beta_t_range[x]],input_true_pysm_beta[pixel_number+pixel:]])),P_d,P_t,P_s,pixel_list,nside)

                        khi_list_normal[x]=fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])
                        khi_list_jsl[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:len(P_d)+pixel],[beta_t_range[x]],input_true_pysm_beta[len(P_d)+pixel+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        fun_list_normal[x]=(-1./2.)*fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])
                        fun_list_jsl[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:len(P_d)+pixel],[beta_t_range[x]],input_true_pysm_beta[len(P_d)+pixel+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        # fun_list_normal[x]=(-1./2.)*fun[pixel_list[pixel]]([1.54,beta_t_range[x],-3.1])
                        # fun_list_jsl[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas[:pixel_number+pixel],[beta_t_range[x]],input_set_of_betas[pixel_number+pixel:]])),P_d,P_t,P_s,pixel_list,nside)
                    # print 'input',list(itertools.chain.from_iterable([input_set_of_betas[:pixel_number+pixel],[beta_t_range[x]],input_set_of_betas[pixel_number+pixel:]]))
                    max_fun_normal=max(fun_list_normal)
                    max_fun_jsl=max(fun_list_jsl)

                    min_fun_normal=min(fun_list_normal)
                    min_fun_jsl=min(fun_list_jsl)


                    max_khi_normal=max(khi_list_normal)
                    max_khi_jsl=max(khi_list_jsl)

                    min_khi_normal=min(khi_list_normal)
                    min_khi_jsl=min(khi_list_jsl)

                    print 'max_fun_normal',max_fun_normal
                    print 'max_fun_jsl',max_fun_jsl

                    fun_list_normal_plot=np.zeros(len(beta_t_range))
                    fun_list_jsl_plot=np.zeros(len(beta_t_range))

                    for x in range(len(beta_t_range)):
                        fun_list_normal_plot[x]=np.exp(fun_list_normal[x]-max_fun_normal)
                        fun_list_jsl_plot[x]=np.exp(fun_list_jsl[x]-max_fun_jsl)
                    # max_fun_list_jsl_plot=max(fun_list_jsl_plot)
                    #
                    # norm_fun_list_jsl_plot=np.zeros(len(beta_t_range))
                    # for x in range(len(beta_t_range)):
                    #     norm_fun_list_jsl_plot[x]=fun_list_jsl_plot[x]/max_fun_list_jsl_plot

                    # axl.plot(beta_t_range,fun_list_normal_plot,label='normal')
                    # axl.plot(beta_t_range,fun_list_jsl_plot,label='jsl')
                    axl[0].plot(beta_t_range,khi_list_normal-min_khi_normal,label='khi2 normal')
                    axl[0].plot(beta_t_range,khi_list_jsl-min_khi_jsl,label='khi2 jsl')
                    axl[0].set_title('khi2 normal vs jsl as a function of T')
                    axl[0].set_xlabel('T')
                    axl[0].legend()

                    axl[1].plot(beta_t_range,fun_list_normal_plot,label='likelihood normal')
                    axl[1].plot(beta_t_range,fun_list_jsl_plot,label='likelihood jsl')
                    axl[1].set_title('likelihood normal vs jsl as a function of T')
                    axl[1].set_xlabel('T')
                    axl[1].legend()


                    # plt.xscale('log')
                    figl.show()

                if likelihood_comparison_3comp==True:
                    fig3l, ax3l = plt.subplots(2, 2, tight_layout=True)
                    point_number=250
                    beta_d_range=np.linspace(0.5,2.5,point_number)
                    beta_t_range=np.linspace(15,35,point_number)
                    beta_s_range=np.linspace(-4.0,-2.0,point_number)
                    # fun_list_normal_dust=np.zeros(len(beta_t_range))
                    # fun_list_jsl_dust=np.zeros(len(beta_t_range))
                    khi_list_normal_dust=np.zeros(len(beta_t_range))
                    khi_list_jsl_dust=np.zeros(len(beta_t_range))

                    # fun_list_normal_temp=np.zeros(len(beta_t_range))
                    # fun_list_jsl_temp=np.zeros(len(beta_t_range))
                    khi_list_normal_temp=np.zeros(len(beta_t_range))
                    khi_list_jsl_temp=np.zeros(len(beta_t_range))
                    #
                    # fun_list_normal_sync=np.zeros(len(beta_t_range))
                    # fun_list_jsl_sync=np.zeros(len(beta_t_range))
                    khi_list_normal_sync=np.zeros(len(beta_t_range))
                    khi_list_jsl_sync=np.zeros(len(beta_t_range))

                    input_true_pysm_beta=[]
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_beta_dust_pysm[l])
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_temp_dust_pysm[l])
                    for l in pixel_list:
                        input_true_pysm_beta.append(map_beta_sync_pysm[l])
                    if minimization_jsl==True:
                        pixel=np.argmax(np.abs ( np.array(input_set_of_betas_jsl_test[len(P_d):len(P_d)+len(P_t)]) - np.array(input_true_pysm_beta[len(P_d):len(P_d)+len(P_t)])) )
                    # pixel=np.argmin( np.array(input_set_of_betas_jsl_test[len(P_d):len(P_d)+len(P_t)]) )
                    else:
                        pixel=0
                        # pixel=np.argmax(np.abs ( np.array(input_set_of_betas_jsl_test[len(P_d):len(P_d)+len(P_t)]) - np.array(input_true_pysm_beta[len(P_d):len(P_d)+len(P_t)])) )
                    print 'pixel=',pixel
                    #
                    #
                    print 'NORMAL',[map_bd[pixel_list[pixel]],map_bt[pixel_list[pixel]],map_bs[pixel_list[pixel]]]
                    if minimization_jsl==True:
                        print 'JSL',[input_set_of_betas_jsl_test[pixel],input_set_of_betas_jsl_test[len(P_d)+pixel],input_set_of_betas_jsl_test[len(P_d)+len(P_t)+pixel]]
                    print 'true betas pixel=',[map_beta_dust_pysm[pixel_list[pixel]] , map_temp_dust_pysm[pixel_list[pixel]] , map_beta_sync_pysm[pixel_list[pixel]] ]

                    import itertools
                    # print '[map_bd[pixel_list[pixel]],map_bt[pixel_list[pixel]],map_bs[pixel_list[pixel]]]',[map_bd[pixel_list[pixel]],map_bt[pixel_list[pixel]],map_bs[pixel_list[pixel]]]
                    # print '[input_set_of_betas_jsl_test[pixel],input_set_of_betas_jsl_test[pixel_number+pixel],input_set_of_betas_jsl_test[2*pixel_number+pixel]',[input_set_of_betas_jsl_test[pixel],input_set_of_betas_jsl_test[len(P_d)+pixel],input_set_of_betas_jsl_test[len(P_d)+len(P_t) +pixel]]
                    # print 'true betas pixel=',[map_beta_dust_pysm[pixel_list[pixel]] , map_temp_dust_pysm[pixel_list[pixel]] , map_beta_sync_pysm[pixel_list[pixel]] ]
                    for x in range(point_number):
                        # fun_list_normal[x]=(-1./2.)*fun[pixel_list[pixel]]([map_bd[pixel_list[pixel]],beta_t_range[x],map_bs[pixel_list[pixel]]])
                        # fun_list_jsl[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_jsl_test[:pixel_number+pixel],[beta_t_range[x]],input_set_of_betas_jsl_test[pixel_number+pixel:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        # fun_list_normal[x]=(-1./2.)*fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])
                        # fun_list_jsl[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:pixel_number+pixel],[beta_t_range[x]],input_true_pysm_beta[pixel_number+pixel:]])),P_d,P_t,P_s,pixel_list,nside)

                        khi_list_normal_dust[x]=fun[pixel_list[pixel]]([beta_d_range[x],map_temp_dust_pysm[pixel_list[pixel]],map_beta_sync_pysm[pixel_list[pixel]]])
                        khi_list_jsl_dust[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:pixel],[beta_d_range[x]],input_true_pysm_beta[pixel+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        # fun_list_normal_dust[x]=(-1./2.)*fun[pixel_list[pixel]]([beta_d_range[x],map_temp_dust_pysm[pixel_list[pixel]],map_beta_sync_pysm[pixel_list[pixel]]])
                        # fun_list_jsl_dust[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:pixel],[beta_d_range[x]],input_true_pysm_beta[pixel:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)



                        khi_list_normal_temp[x]=fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])
                        khi_list_jsl_temp[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:len(P_d)+pixel],[beta_t_range[x]],input_true_pysm_beta[len(P_d)+pixel+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        # fun_list_norma_templ[x]=(-1./2.)*fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])
                        # fun_list_jsl_temp[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:len(P_d)+pixel],[beta_t_range[x]],input_true_pysm_beta[len(P_d)+pixel:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        khi_list_normal_sync[x]=fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],map_temp_dust_pysm[pixel_list[pixel]],beta_s_range[x]])
                        khi_list_jsl_sync[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:len(P_d)+len(P_t)+pixel],[beta_s_range[x]],input_true_pysm_beta[len(P_d)+len(P_t)+pixel+1:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        # fun_list_normal_sync[x]=(-1./2.)*fun[pixel_list[pixel]]([map_beta_dust_pysm[pixel_list[pixel]],map_temp_dust_pysm[pixel_list[pixel],beta_s_range[x]])
                        # fun_list_jsl_sync[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_true_pysm_beta[:len(P_d)+len(P_t)+pixel],[beta_s_range[x]],input_true_pysm_beta[len(P_d)+len(P_t)+pixel:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac)

                        # fun_list_normal[x]=(-1./2.)*fun[pixel_list[pixel]]([1.54,beta_t_range[x],-3.1])
                        # fun_list_jsl[x]=(-1./2.)*joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas[:pixel_number+pixel],[beta_t_range[x]],input_set_of_betas[pixel_number+pixel:]])),P_d,P_t,P_s,pixel_list,nside)
                    # print 'input',list(itertools.chain.from_iterable([input_set_of_betas[:pixel_number+pixel],[beta_t_range[x]],input_set_of_betas[pixel_number+pixel:]]))
                    # max_fun_normal_dust=max(fun_list_normal_dust)
                    # max_fun_jsl_dust=max(fun_list_jsl_dust)
                    #
                    # max_fun_normal_temp=max(fun_list_normal_temp)
                    # max_fun_jsl_temp=max(fun_list_jsl_temp)
                    #
                    # max_fun_normal_sync=max(fun_list_normal_sync)
                    # max_fun_jsl_sync=max(fun_list_jsl_sync)
                    #
                    #
                    # min_fun_normal=min(fun_list_normal)
                    # min_fun_jsl=min(fun_list_jsl)


                    min_khi_normal_dust=min(khi_list_normal_dust)
                    min_khi_jsl_dust=min(khi_list_jsl_dust)

                    min_khi_normal_temp=min(khi_list_normal_temp)
                    min_khi_jsl_temp=min(khi_list_jsl_temp)

                    min_khi_normal_sync=min(khi_list_normal_sync)
                    min_khi_jsl_sync=min(khi_list_jsl_sync)

                    # min_khi_normal=min(khi_list_normal)
                    # min_khi_jsl=min(khi_list_jsl)

                    # print 'max_fun_normal',max_fun_normal
                    # print 'max_fun_jsl',max_fun_jsl

                    # fun_list_normal_plot_dust=np.zeros(len(beta_d_range))
                    # fun_list_jsl_plot_dust=np.zeros(len(beta_d_range))
                    #
                    # fun_list_normal_plot_temp=np.zeros(len(beta_t_range))
                    # fun_list_jsl_plot_temp=np.zeros(len(beta_t_range))
                    #
                    # fun_list_normal_plot_sync=np.zeros(len(beta_s_range))
                    # fun_list_jsl_plot_sync=np.zeros(len(beta_s_range))
                    #
                    # for x in range(len(beta_t_range)):
                    #     fun_list_normal_plot_dust[x]=np.exp(fun_list_normal_dust[x]-max_fun_normal_dust)
                    #     fun_list_jsl_plot_dust[x]=np.exp(fun_list_jsl_dust[x]-max_fun_jsl_dust)
                    #
                    #     fun_list_normal_plot_temp[x]=np.exp(fun_list_normal_temp[x]-max_fun_normal_temp)
                    #     fun_list_jsl_plot_temp[x]=np.exp(fun_list_jsl_temp[x]-max_fun_jsl_temp)
                    #
                    #     fun_list_normal_plot_sync[x]=np.exp(fun_list_normal_sync[x]-max_fun_normal_sync)
                    #     fun_list_jsl_plot_sync[x]=np.exp(fun_list_jsl_sync[x]-max_fun_jsl_sync)
                    # max_fun_list_jsl_plot=max(fun_list_jsl_plot)
                    #
                    # norm_fun_list_jsl_plot=np.zeros(len(beta_t_range))
                    # for x in range(len(beta_t_range)):
                    #     norm_fun_list_jsl_plot[x]=fun_list_jsl_plot[x]/max_fun_list_jsl_plot

                    # axl.plot(beta_t_range,fun_list_normal_plot,label='normal')
                    # axl.plot(beta_t_range,fun_list_jsl_plot,label='jsl')
                    ax3l[0][0].plot(beta_d_range,khi_list_normal_dust-min_khi_normal_dust,label='khi2 normal')
                    ax3l[0][0].plot(beta_d_range,khi_list_jsl_dust-min_khi_jsl_dust,label='khi2 jsl')
                    ax3l[0][0].set_title('khi2 normal vs jsl as a function of beta d')
                    ax3l[0][0].set_xlabel('T')
                    ax3l[0][0].legend()

                    ax3l[0][1].plot(beta_t_range,khi_list_normal_temp-min_khi_normal_temp,label='khi2 normal')
                    ax3l[0][1].plot(beta_t_range,khi_list_jsl_temp-min_khi_jsl_temp,label='khi2 jsl')
                    ax3l[0][1].set_title('khi2 normal vs jsl as a function of T')
                    ax3l[0][1].set_xlabel('T')
                    ax3l[0][1].legend()


                    ax3l[1][0].plot(beta_s_range,khi_list_normal_sync-min_khi_normal_sync,label='khi2 normal')
                    ax3l[1][0].plot(beta_s_range,khi_list_jsl_sync-min_khi_jsl_sync,label='khi2 jsl')
                    ax3l[1][0].set_title('khi2 normal vs jsl as a function of beta s')
                    ax3l[1][0].set_xlabel('T')
                    ax3l[1][0].legend()


                    # plt.xscale('log')
                    fig3l.show()


                    # sys.exit()

                if jac_comparison==True:

                    figj, axj = plt.subplots(2, 2,  tight_layout=True)
                    pixel=np.argmax(np.abs ( np.array(input_set_of_betas_jsl_test[len(P_d):len(P_d)+len(P_t)]) - np.array(input_true_pysm_beta[len(P_d):len(P_d)+len(P_t)])) )
                    # pixel=1
                    # pixel=np.argmin( np.array(input_set_of_betas_jsl_test[len(P_d):len(P_d)+len(P_t)]) )
                    print 'pixel=',pixel
                    beta_t_range=np.linspace(1,100,500)
                    # jac_list_normal=np.zeros(len(beta_t_range))
                    jac_list_jsl=np.zeros(len(beta_t_range))



                    import itertools
                    jac_list_normal=[]
                    jac_list_jsl=[]

                    print 'NORMAL',[map_bd[pixel_list[pixel]],map_bt[pixel_list[pixel]],map_bs[pixel_list[pixel]]]
                    if minimization_jsl==True:
                        print 'JSL',[input_set_of_betas_jsl_test[pixel],input_set_of_betas_jsl_test[len(P_d)+pixel],input_set_of_betas_jsl_test[len(P_d)+len(P_t)+pixel]]
                    print 'true betas pixel=',[map_beta_dust_pysm[pixel_list[pixel]] , map_temp_dust_pysm[pixel_list[pixel]] , map_beta_sync_pysm[pixel_list[pixel]] ]
                    for x in range(len(beta_t_range)):
                        # print 'jac_list[pixel]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])',jac_list[pixel]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]])
                        jac_list_normal.append(jac_list[pixel]([map_beta_dust_pysm[pixel_list[pixel]],beta_t_range[x],map_beta_sync_pysm[pixel_list[pixel]]]))
                        # jac_list_normal.append([1,2,3])
                        jac_list_jsl.append(return_jac_super_list(list(itertools.chain.from_iterable([input_true_pysm_beta[:len(P_d)+pixel],[beta_t_range[x]],input_true_pysm_beta[len(P_d)+pixel:]])),fun,P_d,P_t,P_s,pixel_list,nside,jac))

                    jac_list_normal_0=[]
                    jac_list_jsl_0=[]
                    for element in jac_list_normal:
                        jac_list_normal_0.append(element[0])
                    for element in jac_list_jsl:
                        jac_list_jsl_0.append(element[0])

                    jac_list_normal_1=[]
                    jac_list_jsl_1=[]
                    for element in jac_list_normal:
                        jac_list_normal_1.append(element[1])
                    for element in jac_list_jsl:
                        jac_list_jsl_1.append(element[1])

                    jac_list_normal_2=[]
                    jac_list_jsl_2=[]
                    for element in jac_list_normal:
                        jac_list_normal_2.append(element[2])
                    for element in jac_list_jsl:
                        jac_list_jsl_2.append(element[2])

                    axj[0][0].plot(beta_t_range,jac_list_normal_0,label='jac normal')
                    axj[0][0].plot(beta_t_range,jac_list_jsl_0,label='jac jsl')
                    axj[0][1].plot(beta_t_range,jac_list_normal_1,label='jac normal')
                    axj[0][1].plot(beta_t_range,jac_list_jsl_1,label='jac jsl')
                    axj[1][0].plot(beta_t_range,jac_list_normal_2,label='ja] normal')
                    axj[1][0].plot(beta_t_range,jac_list_jsl_2,label='jac jsl')
                    axj[0][0].set_title('jac[0] normal vs jsl as 2a function of T')
                    axj[0][1].set_title('jac[1] normal vs jsl as a function of T')
                    axj[1][0].set_title('jac[2] normal vs jsl as a function of T')
                    axj[0][0].set_xlabel('T')
                    axj[0][1].set_xlabel('T')
                    axj[1][0].set_xlabel('T')
                    axj[0][0].legend()
                    axj[0][1].legend()
                    axj[1][0].legend()
                    # plt.xscale('log')
                    figj.show()


            if len(params)==2:
                print '2 PARAMS !!'
                bounds=((1.0, 5.0),(-3.5,-2.0))
                A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)
                fun=[None]*hp.nside2npix(nside)
                last_values=[None]*hp.nside2npix(nside)
                for l in range(pixel_number):
                    fun[pixel_list[l]] = return_fun(l,freq_maps,prewhiten_factors)

                comp=[]
                minimization_result_pixel=[]
                input_beta_zeroth_iteration=[]
                for p in range(pixel_number):
                    minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],([input_set_of_betas[p],input_set_of_betas[p+2*pixel_number]]),bounds=bounds,tol=1e-15).x)
                for i in range(pixel_number):
                    input_beta_zeroth_iteration.append(minimization_result_pixel[i][0])
                for i in range(pixel_number):
                    input_beta_zeroth_iteration.append(20)
                for i in range(pixel_number):
                    input_beta_zeroth_iteration.append(minimization_result_pixel[i][1])
                minimization_result_pixel=[]

                map_bd,map_bt,map_bs=patch_map(input_beta_zeroth_iteration,P_d,P_t,P_s,nside)
                # print 'map_bt',map_bt
                # fisher_list,s_q_00,s_q_10,s_q_20,s_q_01,s_q_11,s_q_21,s_list,nnt_list,fisher_s_list,fisher_n_list,fisher_hess_list=fisher_pixel(map_bd,map_bt,map_bs,fun,pixel_list,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB,Hessian=True)

                map_bd_list.append(map_bd)
                map_bt_list.append(map_bt)
                map_bs_list.append(map_bs)




            if plot_res_to_noise==True and zeroth_iteration==True:
                if pysm_model=='c1d1s1':
                    cmap = plt.get_cmap('jet_r')
                    color = cmap(1./nside)

                    ax_res_to_noise[0][0].hist([ (map_bd[x] - map_beta_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)#bins = np.logspace(-3,3,num=100)
                    ax_res_to_noise[0][1].hist([ (map_bt[x] - map_temp_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)
                    ax_res_to_noise[1][0].hist([ (map_bs[x] - map_beta_sync_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)

                    ax_res_to_noise[0][0].axvline(x=np.mean([ (map_bd[x] - map_beta_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)
                    ax_res_to_noise[0][1].axvline(x=np.mean([ (map_bt[x] - map_temp_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)
                    ax_res_to_noise[1][0].axvline(x=np.mean([ (map_bs[x] - map_beta_sync_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)



                if pysm_model=='d0s0':
                    ax_res_to_noise[0][0].hist([ (map_bd[x] - 1.54 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise d histogram iteration#=%s'%iter)#bins = np.logspace(-3,3,num=100)
                    ax_res_to_noise[0][1].hist([ (map_bt[x] - 20 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise t histogram iteration#=%s'%iter)
                    ax_res_to_noise[1][0].hist([ (map_bs[x] - (-3) ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise s histogram iteration#=%s'%iter)

                    ax_res_to_noise[0][0].axvline(x=np.mean([ (map_bd[x] - 1.54 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ]),ymin=0,ymax=npix/2)
                    ax_res_to_noise[0][1].axvline(x=np.mean([ (map_bt[x] - 20 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ]),ymin=0,ymax=npix/2)
                    ax_res_to_noise[1][0].axvline(x=np.mean([ (map_bs[x] - (-3) ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ]),ymin=0,ymax=npix/2)

                ax_res_to_noise[0][0].legend()
                ax_res_to_noise[0][1].legend()
                ax_res_to_noise[1][0].legend()

                ax_res_to_noise[0][0].set_title('residuals_norm/noise dust')
                ax_res_to_noise[0][1].set_title('residuals_norm/noise temp')
                ax_res_to_noise[1][0].set_title('residuals_norm/noise sync')

            if plot_res_to_noise==True and zeroth_iteration==True:
                if pysm_model=='c1d1s1':
                    cmap = plt.get_cmap('jet_r')
                    color = cmap(1./nside)

                    ax_signal_to_noise[0][0].hist([ (map_bd[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)#bins = np.logspace(-3,3,num=100)
                    ax_signal_to_noise[0][1].hist([ (map_bt[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)
                    ax_signal_to_noise[1][0].hist([ (map_bs[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)

                    ax_signal_to_noise[0][0].axvline(x=np.mean([ (map_bd[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)
                    ax_signal_to_noise[0][1].axvline(x=np.mean([ (map_bt[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)
                    ax_signal_to_noise[1][0].axvline(x=np.mean([ (map_bs[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)



                if pysm_model=='d0s0':
                    ax_signal_to_noise[0][0].hist([ (map_bd[x]  ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise d histogram iteration#=%s'%iter)#bins = np.logspace(-3,3,num=100)
                    ax_signal_to_noise[0][1].hist([ (map_bt[x]  ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise t histogram iteration#=%s'%iter)
                    ax_signal_to_noise[1][0].hist([ (map_bs[x]  ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise s histogram iteration#=%s'%iter)

                    ax_signal_to_noise[0][0].axvline(x=np.mean([ (map_bd[x]  ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ]),ymin=0,ymax=npix/2)
                    ax_signal_to_noise[0][1].axvline(x=np.mean([ (map_bt[x]  ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ]),ymin=0,ymax=npix/2)
                    ax_signal_to_noise[1][0].axvline(x=np.mean([ (map_bs[x]  ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ]),ymin=0,ymax=npix/2)

                ax_signal_to_noise[0][0].legend()
                ax_signal_to_noise[0][1].legend()
                ax_signal_to_noise[1][0].legend()

                ax_signal_to_noise[0][0].set_title('signal_norm/noise dust')
                ax_signal_to_noise[0][1].set_title('signal_norm/noise temp')
                ax_signal_to_noise[1][0].set_title('signal_norm/noise sync')
                # axs[0][0].set_xscale('log')
                # axs[0][1].set_xscale('log')
                # axs[1][0].set_xscale('log')


                # plt.gca().set_xscale("log")
                # plt.gca().set_xscale("log")
                # plt.gca().set_xscale("log")

            if plot_signal_to_noise==True and joint_likelihood_test==True and minimization_jsl==True:
                if pysm_model=='c1d1s1':
                    cmap = plt.get_cmap('jet_r')
                    color = cmap(1./nside)

                    ax_signal_to_noise_jsl[0][0].hist([ (map_bd_jsl_test[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)#bins = np.logspace(-3,3,num=100)
                    ax_signal_to_noise_jsl[0][1].hist([ (map_bt_jsl_test[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)
                    ax_signal_to_noise_jsl[1][0].hist([ (map_bs_jsl_test[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ] ,pixel_number, histtype='step', label='%s'%nside,normed=True)#,color=color)

                    ax_signal_to_noise_jsl[0][0].axvline(x=np.mean([ (map_bd_jsl_test[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)
                    ax_signal_to_noise_jsl[0][1].axvline(x=np.mean([ (map_bt_jsl_test[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)
                    ax_signal_to_noise_jsl[1][0].axvline(x=np.mean([ (map_bs_jsl_test[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean %s'%nside,color=color)

                ax_signal_to_noise_jsl[0][0].legend()
                ax_signal_to_noise_jsl[0][1].legend()
                ax_signal_to_noise_jsl[1][0].legend()

                ax_signal_to_noise_jsl[0][0].set_title('signal_jsl/noise dust')
                ax_signal_to_noise_jsl[0][1].set_title('signal_jsl/noise temp')
                ax_signal_to_noise_jsl[1][0].set_title('signal_jsl/noise sync')

            if plot_res_to_noise==True and joint_likelihood_test==True and minimization_jsl==True:
                if pysm_model=='c1d1s1':
                    cmap = plt.get_cmap('jet_r')
                    color = cmap(1./nside)

                    ax_res_to_noise_jsl[0][0].hist([ (map_bd_jsl_test[x] - map_beta_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ] ,pixel_number, histtype='step', label='nside %s'%nside,normed=True)#,color=color)#bins = np.logspace(-3,3,num=100)
                    ax_res_to_noise_jsl[0][1].hist([ (map_bt_jsl_test[x] - map_temp_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ] ,pixel_number, histtype='step', label='nside %s'%nside,normed=True)#,color=color)
                    ax_res_to_noise_jsl[1][0].hist([ (map_bs_jsl_test[x] - map_beta_sync_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ] ,pixel_number, histtype='step', label='nside %s'%nside,normed=True)#,color=color)

                    ax_res_to_noise_jsl[0][0].axvline(x=np.mean([ (map_bd_jsl_test[x] - map_beta_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean nside %s'%nside,color=color)
                    ax_res_to_noise_jsl[0][1].axvline(x=np.mean([ (map_bt_jsl_test[x] - map_temp_dust_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean nside %s'%nside,color=color)
                    ax_res_to_noise_jsl[1][0].axvline(x=np.mean([ (map_bs_jsl_test[x] - map_beta_sync_pysm[x] ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ]),ymin=0,ymax=npix/2, label='mean nside %s'%nside,color=color)



                if pysm_model=='d0s0':
                    ax_res_to_noise_jsl[0][0].hist([ (map_bd_jsl_test[x] - 1.54 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise d histogram iteration#=%s'%iter)#bins = np.logspace(-3,3,num=100)
                    ax_res_to_noise_jsl[0][1].hist([ (map_bt_jsl_test[x] - 20 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise t histogram iteration#=%s'%iter)
                    ax_res_to_noise_jsl[1][0].hist([ (map_bs_jsl_test[x] - (-3) ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ] ,pixel_number, histtype='step', label='signal/noise s histogram iteration#=%s'%iter)

                    ax_res_to_noise_jsl[0][0].axvline(x=np.mean([ (map_bd_jsl_test[x] - 1.54 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[0][0])) for x in pixel_list ]),ymin=0,ymax=npix/2)
                    ax_res_to_noise_jsl[0][1].axvline(x=np.mean([ (map_bt_jsl_test[x] - 20 ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[1][1])) for x in pixel_list ]),ymin=0,ymax=npix/2)
                    ax_res_to_noise_jsl[1][0].axvline(x=np.mean([ (map_bs_jsl_test[x] - (-3) ) /  np.array( np.sqrt(np.linalg.inv(fisher_list[x])[2][2])) for x in pixel_list ]),ymin=0,ymax=npix/2)

                ax_res_to_noise_jsl[0][0].legend()
                ax_res_to_noise_jsl[0][1].legend()
                ax_res_to_noise_jsl[1][0].legend()

                ax_res_to_noise_jsl[0][0].set_title('residuals_jsl/noise dust')
                ax_res_to_noise_jsl[0][1].set_title('residuals_jsl/noise temp')
                ax_res_to_noise_jsl[1][0].set_title('residuals_jsl/noise sync')

            del instrument,self
        # print 'map_bd_list',map_bd_list[:][1]
        #----------------------------C1D1S1:
        # print 'np.mean(map_bd_list,axis=0)',np.mean([[map_bd_list[x][y] for y in pixel_list] for x in range(len(map_bd_list))],axis=0)

        if plot_joint_likelihood_test==True and joint_likelihood_test==True and minimization_jsl==True:
            if pysm_model=='c1d1s1':
                ax_res_jsl[0][0].hist([np.mean([[map_bd_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bd_list_jsl_test))],axis=0)[pixel_list.index(z)] - map_beta_dust_pysm[z] for z in pixel_list],pixel_number/10)
                ax_res_jsl[0][1].hist([np.mean([[map_bt_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bt_list_jsl_test))],axis=0)[pixel_list.index(z)] - map_temp_dust_pysm[z] for z in pixel_list],pixel_number/10)
                ax_res_jsl[1][0].hist([np.mean([[map_bs_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bs_list_jsl_test))],axis=0)[pixel_list.index(z)] - map_beta_sync_pysm[z] for z in pixel_list],pixel_number/10)

            #-----------------------------S0D0 :
            if pysm_model=='d0s0':
                ax_res_jsl[0][0].hist([np.mean([[map_bd_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bd_list_jsl_test))],axis=0)[pixel_list.index(z)] - 1.54 for z in pixel_list],pixel_number/10)
                ax_res_jsl[0][1].hist([np.mean([[map_bt_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bt_list_jsl_test))],axis=0)[pixel_list.index(z)] - 20 for z in pixel_list],pixel_number/10)
                ax_res_jsl[1][0].hist([np.mean([[map_bs_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bs_list_jsl_test))],axis=0)[pixel_list.index(z)] - (-3) for z in pixel_list],pixel_number/10)


            ax_res_jsl[0][0].set_title('mean_beta_jsl - beta_true dust')
            ax_res_jsl[0][1].set_title('mean_beta_jsl - beta_true temp')
            ax_res_jsl[1][0].set_title('mean_beta_jsl - beta_true sync')

            fig_res_jsl.show()

        if plot_histo_residu==True and zeroth_iteration==True:
            if pysm_model=='c1d1s1':
                ax_res_norm[0][0].hist([np.mean([[map_bd_list[x][y] for y in pixel_list] for x in range(len(map_bd_list))],axis=0)[pixel_list.index(z)] - map_beta_dust_pysm[z] for z in pixel_list],pixel_number/10)
                ax_res_norm[0][1].hist([np.mean([[map_bt_list[x][y] for y in pixel_list] for x in range(len(map_bt_list))],axis=0)[pixel_list.index(z)] - map_temp_dust_pysm[z] for z in pixel_list],pixel_number/10)
                ax_res_norm[1][0].hist([np.mean([[map_bs_list[x][y] for y in pixel_list] for x in range(len(map_bs_list))],axis=0)[pixel_list.index(z)] - map_beta_sync_pysm[z] for z in pixel_list],pixel_number/10)

            #-----------------------------S0D0 :
            if pysm_model=='d0s0':
                ax_res_norm[0][0].hist([np.mean([[map_bd_list[x][y] for y in pixel_list] for x in range(len(map_bd_list))],axis=0)[pixel_list.index(z)] - 1.54 for z in pixel_list],pixel_number/10)
                ax_res_norm[0][1].hist([np.mean([[map_bt_list[x][y] for y in pixel_list] for x in range(len(map_bt_list))],axis=0)[pixel_list.index(z)] - 20 for z in pixel_list],pixel_number/10)
                ax_res_norm[1][0].hist([np.mean([[map_bs_list[x][y] for y in pixel_list] for x in range(len(map_bs_list))],axis=0)[pixel_list.index(z)] - (-3) for z in pixel_list],pixel_number/10)


            ax_res_norm[0][0].set_title('mean_beta_norm - beta_true dust')
            ax_res_norm[0][1].set_title('mean_beta_norm - beta_true temp')
            ax_res_norm[1][0].set_title('mean_beta_norm - beta_true sync')

            fig_res_norm.show()

        if plot_histo_residu_normal_and_jsl==True and joint_likelihood_test==True and minimization_jsl==True:
            if pysm_model=='c1d1s1':
                ax_res_norm_and_jsl[0][0].hist([np.mean([[map_bd_list[x][y] for y in pixel_list] for x in range(len(map_bd_list))],axis=0)[pixel_list.index(z)] - map_beta_dust_pysm[z] for z in pixel_list],bins = np.linspace(-0.2, 0.2, num=20),label='normal method',histtype='step')
                ax_res_norm_and_jsl[0][1].hist([np.mean([[map_bt_list[x][y] for y in pixel_list] for x in range(len(map_bt_list))],axis=0)[pixel_list.index(z)] - map_temp_dust_pysm[z] for z in pixel_list],bins = np.linspace(-8, 8, num=20),label='normal method',histtype='step')
                ax_res_norm_and_jsl[1][0].hist([np.mean([[map_bs_list[x][y] for y in pixel_list] for x in range(len(map_bs_list))],axis=0)[pixel_list.index(z)] - map_beta_sync_pysm[z] for z in pixel_list],bins = np.linspace(-2, 2, num=20),label='normal method',histtype='step')
                ax_res_norm_and_jsl[0][0].hist([np.mean([[map_bd_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bd_list_jsl_test))],axis=0)[pixel_list.index(z)] - map_beta_dust_pysm[z] for z in pixel_list],bins = np.linspace(-0.2, 0.2, num=20),label='jsl',histtype='step')
                ax_res_norm_and_jsl[0][1].hist([np.mean([[map_bt_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bt_list_jsl_test))],axis=0)[pixel_list.index(z)] - map_temp_dust_pysm[z] for z in pixel_list],bins = np.linspace(-8, 8, num=20),label='jsl',histtype='step')
                ax_res_norm_and_jsl[1][0].hist([np.mean([[map_bs_list_jsl_test[x][y] for y in pixel_list] for x in range(len(map_bs_list_jsl_test))],axis=0)[pixel_list.index(z)] - map_beta_sync_pysm[z] for z in pixel_list],bins = np.linspace(-2, 2, num=20),label='jsl',histtype='step')

                ax_res_norm_and_jsl[0][0].set_title('mean_beta - beta_true dust')
                ax_res_norm_and_jsl[0][1].set_title('mean_beta - beta_true temp')
                ax_res_norm_and_jsl[1][0].set_title('mean_beta - beta_true sync')
                ax_res_norm_and_jsl[0][0].legend()
                ax_res_norm_and_jsl[0][1].legend()
                ax_res_norm_and_jsl[1][0].legend()

            fig_res_norm_and_jsl.show()

    if plot_signal_to_noise==True and zeroth_iteration==True:

        fig_signal_to_noise.show()
        fig_res_to_noise.show()

    if plot_res_to_noise==True and joint_likelihood_test==True and minimization_jsl==True:
        fig_res_to_noise_jsl.show()
    if plot_signal_to_noise==True and joint_likelihood_test==True and minimization_jsl==True:
        fig_signal_to_noise_jsl.show()
        # plt.title('residuals/noise nside=%s'%nside)


        # axs_comp_sep[0][0].hist([np.mean([[map_bd_list_comp_sep[x][y] for y in pixel_list] for x in range(len(map_bd_list_comp_sep))],axis=0)[pixel_list.index(z)] - 1.54 for z in pixel_list],pixel_number,label='mean_beta - beta_true dust')
        # axs_comp_sep[0][1].hist([np.mean([[map_bt_list_comp_sep[x][y] for y in pixel_list] for x in range(len(map_bt_list_comp_sep))],axis=0)[pixel_list.index(z)] - 20 for z in pixel_list],pixel_number,label='mean_beta - beta_true temp')
        # axs_comp_sep[1][0].hist([np.mean([[map_bs_list_comp_sep[x][y] for y in pixel_list] for x in range(len(map_bs_list_comp_sep))],axis=0)[pixel_list.index(z)] - (-3) for z in pixel_list],pixel_number,label='mean_beta - beta_true sync')
        #
        #
        # axs_comp_sep[0][0].set_title('mean_beta - beta_true dust comp_sep')
        # axs_comp_sep[0][1].set_title('mean_beta - beta_true temp comp_sep')
        # axs_comp_sep[1][0].set_title('mean_beta - beta_true sync comp_sep')
        # plt.title('residuals/noise nside=%s'%nside)

        # plt.show(fig_comp_sep)
    plt.show()
    sys.exit()


# nside_comparison_jsl=False
# if nside_comparison_jsl==True:
    # sys.exit()



#-----------------------------------------------ZEROTH ITERATION----------------------------------------------------------------------
minimization=True
plot_FOM=False



P_d=[]
P_s=[]
P_t=[]
h=0
for i in range(len(freq_maps[0][1])): P_d.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_s.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_t.append([pixel_list[i]])


#beta initialisation
input_set_of_betas=[]
for i in range(len(freq_maps[0][0])):
    input_set_of_betas.append(1.59)
for i in range(len(freq_maps[0][0])):
    input_set_of_betas.append(19.6)
for i in range(len(freq_maps[0][0])):
    input_set_of_betas.append(-3.1)


# bounds=((0.5,2.5),(0.1,None),(-8,4))
# bounds=((0,10),(0,25),(-10,0))
# bounds=((0,None),(0,None),(None,0))
# bounds=((1.0, 5.0),(15,25),(-3.5,-2.0))
# bounds = []
# bounds.append( (1.0, 5.0) )
# bounds.append( (1.0, 40.0) )
# bounds.append( (-3.5, -2.0) )
if pysm_model=='c1d1s1':
    bounds=((1.45,1.7),(1.0,100),(-7.5,0.5)) #GOOOOOD ONES FOR C1D1S1 DON'T TOUCH!!!!!!!!!!
if pysm_model=='d0s0':
    bounds=((0.5,2.5),(1.0,75),(-7.5,0.5)) # Not the best one yet, needs some more fine tuning

fun=[]
last_values=[]
data=data_patch(freq_maps,P_d,pixel_list)
A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=None)#prewhiten_factors)
print 'param',params

A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)


fun=[None]*hp.nside2npix(nside)
last_values=[None]*hp.nside2npix(nside)
jac_list=[]

for l in range(pixel_number):
     # prewhitened_data = prewhiten_factors * np.transpose(freq_maps)[l]
     # print 'np.transpose(freq_maps)[%s]'%l,np.transpose(freq_maps)[l]
     # fun[pixel_list[l]] = lambda x: -fgal.logL(A_ev(x), np.transpose(freq_maps)[l],np.diag(prewhiten_factors**2))#np.transpose(freq_maps)[l], np.diag(prewhiten_factors**2))
     # fun_test.append(lambda x: -fgal.logL(A_ev(x), np.transpose(freq_maps)[l],np.diag(prewhiten_factors**2)))
     fun[pixel_list[l]] = return_fun(l,freq_maps,prewhiten_factors,A_ev)
     jac_list.append(return_jac(l,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB))
     # print 'fun',fun
     # print 'fun[pixel_list[l]]',fun[pixel_list[l]]([input_set_of_betas[150],input_set_of_betas[150+pixel_number],input_set_of_betas[150+2*pixel_number]])
     # last_values[pixel_list[l]]=last_valuestemp
     # print 'uev pixel:%s'%l,last_values[0]
     # del funtemp#, jactemp,last_valuestemp



# for p in range(hp.nside2npix(nside)):
#     funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
#     fun.append(funtemp)
#     last_values.append(last_valuestemp)
#     print(funtemp([1.59,19.6,-3.1]))

comp=[]
minimization_result_pixel=[]
input_beta_zeroth_iteration=[]
for p in range(pixel_number):
    # minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],([input_set_of_betas[p],input_set_of_betas[p+len(P_d)],input_set_of_betas[p+len(P_d)+len(P_t)]]),bounds=bounds).x)
    minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],np.array([input_set_of_betas[p],input_set_of_betas[p+len(P_d)],input_set_of_betas[p+len(P_d)+len(P_t)]]),\
    jac=jac_list[p], tol=1e-18,bounds=bounds).x)
    """
    for i in range(3):
        if ((minimization_result_pixel[-1][i] == bounds[i][0]) or\
                     (minimization_result_pixel[-1][i] == bounds[i][1])):
            beta_loc = copy.deepcopy(minimization_result_pixel[-1])
            beta_v = np.arange(bounds[i][0], bounds[i][1], 1e-3)
            logL = beta_v*0.0
            for j in range(len(beta_v)):
                beta_loc[i] = beta_v[j]*1.0
                logL[j] = fun[pixel_list[p]](beta_loc)
            plt.figure()
            plt.title('pixel = '+str(pixel_list[p])+' / beta = '+str(i))
            plt.plot(beta_v, logL, 'k-')
            plt.show()
    """
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][0])
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][1])
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][2])
minimization_result_pixel=[]


map_bd,map_bt,map_bs=patch_map(input_beta_zeroth_iteration,P_d,P_t,P_s,nside)
fisher_list,s_list,n_cmb_list,fisher_s_list,fisher_n_list,fisher_hess_list=fisher_pixel(map_bd,map_bt,map_bs,fun,pixel_list,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB,Hessian=True)

#--------------------------------------------------------------beta residuals histograms with sigma display-----------------------------------------------------------------

plot_export_for_report=True
if plot_export_for_report!=True:
    plt.hist([np.array(map_bd[x])-map_beta_dust_pysm[x] for x in pixel_list],pixel_number,histtype='step', label='beta dust residuals')
    plt.hist( np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list]) ,pixel_number , histtype='step', label='sigma_d histogram')
    plt.legend()
    plt.show()

    plt.hist([np.array(map_bt[x])-map_temp_dust_pysm[x] for x in pixel_list],pixel_number,histtype='step', label='temp dust residuals')
    plt.hist( np.array([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list]) ,pixel_number , histtype='step', label='sigma_t histogram')
    plt.legend()
    plt.show()

    plt.hist([np.array(map_bs[x])-map_beta_sync_pysm[x] for x in pixel_list],pixel_number,histtype='step', label='beta sync residuals')
    plt.hist( np.array([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list]) ,pixel_number , histtype='step', label='sigma_s histogram')
    plt.legend()
    plt.show()

from textwrap import wrap

if plot_export_for_report==True:
    plt.hist([np.array(map_bd[x])-map_beta_dust_pysm[x] for x in pixel_list],pixel_number/10,histtype='step', label='beta dust residuals')
    plt.xlabel('beta estimate - true beta')
    plt.ylabel('pixel number')
    plt.title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Beta Dust',60)))
    # plt.title("histogram of the difference between the estimation"\
                    # " and the true value of beta dust", ha='center')
    plt.legend()
    plt.savefig('histo_diff_beta_true_bd_0th_iter_nside=%s'%nside)
    plt.close()

    plt.hist([np.array(map_bt[x])-map_temp_dust_pysm[x] for x in pixel_list],pixel_number/10,histtype='step', label='temp dust residuals')
    plt.xlabel('beta estimate - true beta')
    plt.ylabel('pixel number')
    plt.title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Dust Temperature')))
    plt.legend()
    plt.savefig('histo_diff_beta_true_temp_0th_iter_nside=%s'%nside)
    plt.close()

    plt.hist([np.array(map_bs[x])-map_beta_sync_pysm[x] for x in pixel_list],pixel_number/10,histtype='step', label='beta sync residuals')
    plt.xlabel('beta estimate - true beta')
    plt.ylabel('pixel number')
    plt.title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Beta Synchrotron')))
    plt.legend()
    plt.savefig('histo_diff_beta_true_bs_0th_iter_nside=%s'%nside)
    plt.close()

    hp.mollview(np.array(map_bd) , unit='beta dust')#,min=bounds[0][0],max=bounds[0][1])
    plt.title('Beta Dust Map STEP 1')
    plt.savefig('map_iter0_bd_nside=%s'%nside)
    plt.close()

    hp.mollview(np.array(map_bt) , unit='dust temperature in K')#,min=bounds[1][0],max=bounds[1][1])
    plt.title('Dust Temperature Map STEP 1')
    plt.savefig('map_iter0_t_nside=%s'%nside)
    plt.close()

    hp.mollview(np.array(map_bs) , unit='beta synchrotron')#,min=bounds[2][0],max=bounds[2][1])
    plt.title('Beta Synchrotron Map STEP 1')
    plt.savefig('map_iter0_bs_nside=%s'%nside)
    plt.close()

    plt.hist( np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list]) ,pixel_number , histtype='step', label='sigma_d histogram')
    plt.xlabel('Statistical Error')
    plt.ylabel('pixel number')
    plt.title('Statistical Error on Beta Dust : STEP 2')
    plt.savefig('error_iter0_bd_nside=%s'%nside)
    plt.close()
    plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list]),pixel_number, histtype='step', label='sigma_t histogram')
    plt.xlabel('Statistical Error')
    plt.ylabel('pixel number')
    plt.title('Statistical Error on Dust Temperature : STEP 2')
    plt.savefig('error_iter0_t_nside=%s'%nside)
    plt.close()
    plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list]),pixel_number, histtype='step', label='sigma_s histogram')
    plt.xlabel('Statistical Error')
    plt.ylabel('pixel number')
    plt.title('Statistical Error on Beta Synchrotron : STEP 2')
    plt.savefig('error_iter0_bs_nside=%s'%nside)
    plt.close()






#--------------------------------------------------------------beta residuals map with sigma map-----------------------------------------------------------------

# map_bd_sigma,map_bt_sigma,map_bs_sigma=patch_map(    ,P_d,P_t,P_s,nside)



map_beta_dust_residuals=[np.nan]*hp.nside2npix(nside)
map_bd_sigma=[np.nan]*hp.nside2npix(nside)
map_temp_dust_residuals=[np.nan]*hp.nside2npix(nside)
map_bt_sigma=[np.nan]*hp.nside2npix(nside)
map_beta_sync_residuals=[np.nan]*hp.nside2npix(nside)
map_bs_sigma=[np.nan]*hp.nside2npix(nside)

for x in pixel_list:
    map_beta_dust_residuals[x]=np.array(map_bd[x])-map_beta_dust_pysm[x]
    map_temp_dust_residuals[x]=np.array(map_bt[x])-map_temp_dust_pysm[x]
    map_beta_sync_residuals[x]=np.array(map_bs[x])-map_beta_sync_pysm[x]
    map_bd_sigma[x]=np.array(np.sqrt(np.linalg.inv(fisher_list[x])[0][0]))
    map_bt_sigma[x]=np.array(np.sqrt(np.linalg.inv(fisher_list[x])[1][1]))
    map_bs_sigma[x]=np.array(np.sqrt(np.linalg.inv(fisher_list[x])[2][2]))
# print map_beta_dust_residuals
if plot_export_for_report!=True:
    hp.mollview( np.log(np.array(map_beta_dust_residuals)),sub=(2,2,1) )
    plt.title('beta dust residuals')
    hp.mollview(np.log(np.array(map_bd_sigma)),sub=(2,2,2))
    plt.title('sigma beta dust')
    plt.show()

    hp.mollview( np.log(np.array(map_temp_dust_residuals)),sub=(2,2,1))
    plt.title('temps dust residuals')
    hp.mollview(np.log(np.array(map_bt_sigma)),sub=(2,2,2))
    plt.title('sigma temp dust')
    plt.show()

    hp.mollview( np.log(np.array(map_beta_sync_residuals)),sub=(2,2,1))
    plt.title('beta sync residuals')
    hp.mollview(np.log(np.array(map_bs_sigma)),sub=(2,2,2))
    plt.title('sigma beta sync')
    plt.show()



#---------------------------------------------------------------New patches with pbyp method-------------------------------------------------------------------

P_d_pbyp , P_t_pbyp , P_s_pbyp , sigma_patch_list , beta2_patch_list = patch_making_pbyp(input_beta_zeroth_iteration,fisher_list,pixel_number,pixel_list)
# print '[np.sqrt(np.linalg.inv(fisher_list[x])[1][1]',sorted([np.sqrt(np.linalg.inv(fisher_list[x])[1][1])for x in pixel_list]) [-1]
# print 'P_d_test shape',np.shape(P_d_test)
# print 'P_t_test shape',np.shape(P_t_test)
# print 'P_s_test shape',np.shape(P_s_test)
map_bd_new_sigma,map_bt_new_sigma,map_bs_new_sigma=patch_map(sigma_patch_list,P_d_pbyp,P_t_pbyp,P_s_pbyp,nside)
map_bd_new_beta2,map_bt_new_beta2,map_bs_new_beta2=patch_map(beta2_patch_list,P_d_pbyp,P_t_pbyp,P_s_pbyp,nside)


if plot_export_for_report==True:
    index_list=[]
    temp_list5=[]
    binsd=np.linspace(np.amin(input_beta_zeroth_iteration[:len(P_d)]),np.amax(input_beta_zeroth_iteration[:len(P_d)]),pixel_number)
    for j in range(len(P_d_pbyp)):
        for l in range(len(P_d_pbyp[j])):
            index_list.append(pixel_list.index(P_d_pbyp[j][l]))
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[index_list]],binsd,label='patch # %s'%j)
        index_list=[]
    # plt.legend()
    plt.title('Histogram Representing the Adaptive Patch for Beta Dust')
    plt.savefig('patch_histo_bd_nside=%s'%nside)
    plt.close()

    binst=np.linspace(np.amin(input_beta_zeroth_iteration[len(P_d):len(P_d)+len(P_t)]),np.amax(input_beta_zeroth_iteration[len(P_d):len(P_d)+len(P_t)]),pixel_number)
    for j in range(len(P_t_pbyp)):
        for l in range(len(P_t_pbyp[j])):
            index_list.append(pixel_list.index(P_t_pbyp[j][l]))
            temp_list5.append(index_list[l]+len(P_d))
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[temp_list5]],binst,label='patch # %s'%j)
        temp_list5=[]
        index_list=[]
    # plt.legend()
    plt.title('Histogram Representing the Adaptive Patch for Dust Temperature')
    plt.savefig('patch_histo_bt_nside=%s'%nside)
    plt.close()

    binss=np.linspace(np.amin(input_beta_zeroth_iteration[len(P_d)+len(P_t):len(P_d)+len(P_t)+len(P_s)]),np.amax(input_beta_zeroth_iteration[len(P_d)+len(P_t):len(P_d)+len(P_t)+len(P_s)]),pixel_number)
    for j in range(len(P_s_pbyp)):
        for l in range(len(P_s_pbyp[j])):
            index_list.append(pixel_list.index(P_s_pbyp[j][l]))
            temp_list5.append(index_list[l]+len(P_d)+len(P_t))
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[temp_list5]],binss,label='patch # %s'%j)
        temp_list5=[]
        index_list=[]
    # plt.legend()
    plt.title('Histogram Representing the Adaptive Patch for Beta Synchrotron')
    plt.savefig('patch_histo_bs_nside=%s'%nside)
    plt.close()



#---------------------------------------------------------new beta estimation over patches and corresponding map display------------------------------------
input_set_of_betas_pbyp=[]
for i in range(len(P_d_pbyp)):
    input_set_of_betas_pbyp.append(np.mean([map_bd[x] for x in P_d_pbyp[i]]))
for i in range(len(P_t_pbyp)):
    input_set_of_betas_pbyp.append(np.mean([map_bt[x] for x in P_t_pbyp[i]]))
for i in range(len(P_s_pbyp)):
    input_set_of_betas_pbyp.append(np.mean([map_bs[x] for x in P_s_pbyp[i]]))

import itertools
jac=[]
point_number=25
list_min_khi_jsl_dust=[]
list_min_khi_jsl_temp=[]
list_min_khi_jsl_sync=[]
beta_d_range=np.linspace(0.5,2.5,point_number)
beta_t_range=np.linspace(15,35,point_number)
beta_s_range=np.linspace(-4.0,-2.0,point_number)
khi_list_jsl_dust=np.zeros(point_number)
khi_list_jsl_temp=np.zeros(point_number)
khi_list_jsl_sync=np.zeros(point_number)

print pixel_list
for patch in range(len(P_d_pbyp)):
    for x in range(point_number):
        khi_list_jsl_dust[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_pbyp[:patch],[beta_d_range[x]],input_set_of_betas_pbyp[patch+1:]])),fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,jac)
    list_min_khi_jsl_dust.append(beta_d_range[np.argmin(khi_list_jsl_dust)])
    # print list_min_khi_jsl_dust[-1]
    khi_list_jsl_dust=np.zeros(point_number)

for patch in range(len(P_t_pbyp)):
    for x in range(point_number):
        khi_list_jsl_temp[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_pbyp[:len(P_d_pbyp)+patch],[beta_t_range[x]],input_set_of_betas_pbyp[len(P_d_pbyp)+patch+1:]])),fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,jac)
    list_min_khi_jsl_temp.append(beta_t_range[np.argmin(khi_list_jsl_temp)])
    khi_list_jsl_temp=np.zeros(point_number)

for patch in range(len(P_s_pbyp)):
    for x in range(point_number):
        khi_list_jsl_sync[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_pbyp[:len(P_d_pbyp)+len(P_t_pbyp)+patch],[beta_s_range[x]],input_set_of_betas_pbyp[len(P_d_pbyp)+len(P_t_pbyp)+patch+1:]])),fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,jac)
    list_min_khi_jsl_sync.append(beta_s_range[np.argmin(khi_list_jsl_sync)])
    khi_list_jsl_sync=np.zeros(point_number)

init_jsl = list_min_khi_jsl_dust + list_min_khi_jsl_temp + list_min_khi_jsl_sync
print 'init jsl=',init_jsl



if minimization==True:

    def return_jac_super_list(input_set_of_betas,fun,P_d,P_t,P_s,pixel_list,nside,jac): # /!\!!!!!!  NE MARCHE QUE DANS CE CAS PRECIS OU IL Y A AUTANT D'ELEMENT DANS P_i QUE DE PIXEL !!!!
        map_bd,map_bt,map_bs=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)
        jac_super_temp=0
        jac_super_list=[]
        for patch in range(len(P_d)):
            for p in P_d[patch]:
                jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                            ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [0]
            jac_super_list.append(jac_super_temp)
            jac_super_temp=0
        for patch in range(len(P_t)):
            for p in P_t[patch]:
                jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                            ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [1]
            jac_super_list.append(jac_super_temp)
            jac_super_temp=0
        for patch in range(len(P_s)):
            for p in P_s[patch]:
                jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                            ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [2]
            jac_super_list.append(jac_super_temp)
            jac_super_temp=0

        return np.array(jac_super_list)

    super_bounds=[]
    for p in range(len(P_d_pbyp)):
        super_bounds.append((1.45,1.7))
    for p in range(len(P_t_pbyp)):
        super_bounds.append((1.0,100))
    for p in range(len(P_s_pbyp)):
        super_bounds.append((-7.5,0.5))

    print 'input_set_of_betas_pbyp shape',np.shape(input_set_of_betas_pbyp)

    new_set_of_beta_pbyp=scipy.optimize.minimize(joint_spectral_likelihood,init_jsl,(fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,return_jac_super_list), tol=1e-18,bounds=super_bounds,jac=return_jac_super_list).x
    print 'new_set_of_beta_test',new_set_of_beta_pbyp
    map_bd_pbyp,map_bt_pbyp,map_bs_pbyp=patch_map(new_set_of_beta_pbyp,P_d_pbyp,P_t_pbyp,P_s_pbyp,nside)


    # else:
    #     meta_P=np.multidim_intersect(P_d,np.multidim_intersect(P_s,P_t))
    #     for i in range(len(meta_P)):
    #         print 'optimisation=',scipy.optimize.minimize(joint_spectral_likelihood_one_patch,input_set_of_betas,(meta_P,i,pixel_list,nside)).x
    # azer
    if plot_export_for_report!=True:
        hp.mollview(np.array(map_bd_pbyp),sub=(2,2,1), unit='beta dust')#,min=bounds[0][0],max=bounds[0][1])
        plt.title('beta dust map after joint likelihood minimization over pbyp patches')
        hp.mollview(np.array(map_bt_pbyp),sub=(2,2,2), unit='dust temperature in K')#,min=bounds[1][0],max=bounds[1][1])
        plt.title('dust temp map after joint likelihood minimization over pbyp patches')
        hp.mollview(np.array(map_bs_pbyp),sub=(2,2,3), unit='beta synchrotron')#,min=bounds[2][0],max=bounds[2][1])
        plt.title('beta sync map after joint likelihood minimization over pbyp patches')
        plt.show()
    if plot_export_for_report==True:
        hp.mollview(np.array(map_bd_pbyp), unit='beta dust')#,min=bounds[0][0],max=bounds[0][1])
        plt.title('beta dust map last step')
        plt.savefig('map_pbyp_bd_nside=%s'%nside)
        plt.close()
        hp.mollview(np.array(map_bt_pbyp), unit='dust temperature in K')#,min=bounds[1][0],max=bounds[1][1])
        plt.title('dust temp map last step')
        plt.savefig('map_pbyp_bt_nside=%s'%nside)
        plt.close()
        hp.mollview(np.array(map_bs_pbyp), unit='beta synchrotron')#,min=bounds[2][0],max=bounds[2][1])
        plt.title('beta sync map last step')
        plt.savefig('map_pbyp_bs_nside=%s'%nside)
        plt.close()
        # plt.show()

    #----------------------------------------------------------------histogram comparing beta zeroth iteration and beta from joint likelihood pbyp----------------
    if plot_export_for_report==True:

        plt.hist(np.array([map_bd[x] - map_beta_dust_pysm[x] for x in pixel_list]),histtype='step', label='np approach', bins = np.linspace(-0.2, 0.2, num=20))
        plt.hist(np.array([map_bd_pbyp[x] - map_beta_dust_pysm[x] for x in pixel_list]),histtype='step', label='adaptive approach',bins = np.linspace(-0.2, 0.2, num=20))
        plt.xlabel('beta estimate - true beta')
        plt.ylabel('pixel number')
        plt.legend()
        plt.title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Beta Dust for Both Methods')))
        plt.savefig('histo_bd_comp_methode_nside=%s'%nside)
        plt.close()


        plt.hist(np.array([map_bt[x] - map_temp_dust_pysm[x] for x in pixel_list]),histtype='step', label='np approach' ,bins = np.linspace(-5, 5, num=20))
        plt.hist(np.array([map_bt_pbyp[x] - map_temp_dust_pysm[x] for x in pixel_list]),histtype='step', label='adaptive approach',bins = np.linspace(-5, 5, num=20))
        plt.xlabel('beta estimate - true beta')
        plt.ylabel('pixel number')
        plt.legend()
        plt.title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Dust Temperature for Both Methods')))
        plt.savefig('histo_bt_comp_methode_nside=%s'%nside)
        plt.close()

        plt.hist(np.array([map_bs[x] - map_beta_sync_pysm[x] for x in pixel_list]),histtype='step', label='np approach',bins = np.linspace(-0.2, 0.2, num=20))
        plt.hist(np.array([map_bs_pbyp[x] - map_beta_sync_pysm[x] for x in pixel_list]),histtype='step', label='adaptive approach', bins = np.linspace(-0.2, 0.2, num=20))
        plt.xlabel('beta estimate - true beta')
        plt.ylabel('pixel number')
        plt.legend()
        plt.title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Beta Synchrotron for Both Methods')))
        plt.savefig('histo_bs_comp_methode_nside=%s'%nside)
        plt.close()

    if plot_export_for_report!=True:
        fig_res_comp , ax_res_comp = plt.subplots(2, 2,  tight_layout=True)
        ax_res_comp[0][0].hist(np.array([map_bd[x] - map_beta_dust_pysm[x] for x in pixel_list]),histtype='step', label='np approach') #bins = np.linspace(-0.2, 0.2, num=20)
        ax_res_comp[0][0].hist(np.array([map_bd_pbyp[x] - map_beta_dust_pysm[x] for x in pixel_list]),histtype='step', label='adaptive approach')#,bins = np.linspace(-0.2, 0.2, num=20)
        ax_res_comp[0][0].legend()
        ax_res_comp[0][0].set_title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Beta Dust for Both Methods')))

        # plt.show()


        ax_res_comp[0][1].hist(np.array([map_bt[x] - map_temp_dust_pysm[x] for x in pixel_list]),histtype='step', label='np approach')#,binsd) #,bins = np.linspace(-5, 5, num=20)
        ax_res_comp[0][1].hist(np.array([map_bt_pbyp[x] - map_temp_dust_pysm[x] for x in pixel_list]),histtype='step', label='adaptive approach')#,bins = np.linspace(-5, 5, num=20)
        ax_res_comp[0][1].set_title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Dust Temperature for Both Methods')))

        # plt.show()

        ax_res_comp[1][0].hist(np.array([map_bs[x] - map_beta_sync_pysm[x] for x in pixel_list]),histtype='step', label='np approach')#,binsd) #,bins = np.linspace(-0.2, 0.2, num=20)
        ax_res_comp[1][0].hist(np.array([map_bs_pbyp[x] - map_beta_sync_pysm[x] for x in pixel_list]),histtype='step', label='adaptive approach')#,bins = np.linspace(-0.2, 0.2, num=20)
        ax_res_comp[1][0].legend()
        ax_res_comp[1][0].set_title('\n'.join(wrap('Histogram of the Difference between the Estimation and the True Value of Beta Synchrotron for Both Methods')))

        fig_res_comp.show()
        plt.show()
print 'max normal res for beta dust=',np.amax(np.array([np.abs(map_bd[x] - map_beta_dust_pysm[x]) for x in pixel_list]))
print 'max pbyp_jsl res for beta dust=',np.amax(np.array([np.abs(map_bd_pbyp[x] - map_beta_dust_pysm[x]) for x in pixel_list]))
print ''
print 'max normal res for temp dust=',np.amax(np.array([np.abs(map_bt[x] - map_temp_dust_pysm[x]) for x in pixel_list]))
print 'max pbyp_jsl res for temp dust=',np.amax(np.array([np.abs(map_bt_pbyp[x] - map_temp_dust_pysm[x]) for x in pixel_list]))
print ''
print 'max normal res for beta sync=',np.amax(np.array([np.abs(map_bs[x] - map_beta_sync_pysm[x]) for x in pixel_list]))
print 'max pbyp_jsl res for beta sync=',np.amax(np.array([np.abs(map_bs_pbyp[x] - map_beta_sync_pysm[x]) for x in pixel_list]))
# sys.exit()
#---------------------------------------------------------FOM
plot_FOM=True
if minimization==True and plot_FOM==True:
    fisher_list_smart,s_list_smart,n_cmb_lis_smartt,fisher_s_list_smart,fisher_n_list_smart,fisher_hess_list_smart=fisher_pixel(map_bd_pbyp,map_bt_pbyp,map_bs_pbyp,fun,pixel_list,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB,Hessian=True)

    plt.hist( np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list]) ,pixel_number , histtype='step', label='np approach')
    plt.hist( np.array([np.sqrt(np.linalg.inv(fisher_list_smart[x])[0][0]) for x in pixel_list]) ,pixel_number , histtype='step', label='adaptive approach')
    plt.xlabel('Statistical Error on Beta Dust')
    plt.ylabel('pixel number')
    plt.legend()
    plt.title('Histogram of the Statistical Error on Beta Dust for Both Methods')
    plt.savefig('error_comparison_bd_nside=%s'%nside)
    plt.close()
    plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list]),pixel_number, histtype='step', label='np approach')
    plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list_smart[x])[1][1]) for x in pixel_list]),pixel_number, histtype='step', label='adaptive approach')
    plt.xlabel('Statistical Error on Dust Temperature')
    plt.ylabel('pixel number')
    plt.legend()
    plt.title('Histogram of the Statistical Error on Dust Temperature for Both Methods')
    plt.savefig('error_comparison_t_nside=%s'%nside)
    plt.close()
    plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list]),pixel_number, histtype='step', label='np approach')
    plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list_smart[x])[2][2]) for x in pixel_list]),pixel_number, histtype='step', label='adaptive approach')
    plt.xlabel('Statistical Error on Beta Synchrotron')
    plt.ylabel('pixel number')
    plt.legend()
    plt.title('Histogram of the Statistical Error on Beta Synchrotron for Both Methods')
    plt.savefig('error_comparison_bs_nside=%s'%nside)
    plt.close()

    s_smart_00=[None]*hp.nside2npix(nside)
    s_smart_10=[None]*hp.nside2npix(nside)
    s_idiot_00=[None]*hp.nside2npix(nside)
    s_idiot_10=[None]*hp.nside2npix(nside)
    delta_smart_00=[None]*hp.nside2npix(nside)
    delta_smart_01=[None]*hp.nside2npix(nside)
    delta_idiot_00=[None]*hp.nside2npix(nside)
    delta_idiot_01=[None]*hp.nside2npix(nside)
    n_cmb_smart=[None]*hp.nside2npix(nside)
    i=0
    for p in pixel_list:
        s_smart=fgal.Wd(A_ev([map_bd_pbyp[p],map_bt_pbyp[p],map_bs_pbyp[p]]),np.transpose(freq_maps)[i], np.diag(prewhiten_factors**2), return_svd=False)
        n_cmb_smart[p]=fgal.Wd(A_ev([map_bd_pbyp[p],map_bt_pbyp[p],map_bs_pbyp[p]]),np.transpose(noise)[i], np.diag(prewhiten_factors**2), return_svd=False)
        s_smart_00[p]=s_smart[0][0]
        s_smart_10[p]=s_smart[1][0]
        s_idiot_00[p]=s_list[p][0][0]
        s_idiot_10[p]=s_list[p][1][0]
        delta_smart_00[p]=-(np.array ( s_smart_00)[p] - n_cmb_smart[p][0][0] ) + sky.cmb(nu)[1,1,p] #-nnt_list[i][0][0]
        delta_smart_01[p]=-(np.array(s_smart_10)[p] - n_cmb_smart[p][1][0] ) + sky.cmb(nu)[1,2,p] #-nnt_list[i][1][1]
        delta_idiot_00[p]=-(np.array(s_idiot_00)[p] - n_cmb_list[p][0][0] ) + sky.cmb(nu)[1,1,p] #-nnt_list[i][0][0]
        delta_idiot_01[p]=-(np.array(s_idiot_10)[p] - n_cmb_list[p][1][0] ) + sky.cmb(nu)[1,2,p] #-nnt_list[i][1][1]
        i+=1
    # smart00max=np.argmax([ -(np.array(s_smart_00)[p] - n_cmb_smart[p][0][0] ) + sky.cmb(nu)[1,1,p] for p in pixel_list])
    # idiot00max=np.argmax([ -(np.array(s_idiot_00)[p] - n_cmb_list[p][0][0] ) + sky.cmb(nu)[1,1,p] for p in pixel_list])
    #
    # print fgal.Wd(A_ev([map_bd_pbyp[smart00max],map_bd_pbyp[smart00max],map_bd_pbyp[smart00max]]),np.transpose(freq_maps)[pixel_list.index(smart00max)], np.diag(prewhiten_factors**2), return_svd=False)
    # print fgal.Wd(A_ev([map_bd_pbyp[smart00max],map_bd_pbyp[smart00max],map_bd_pbyp[smart00max]]),np.transpose(noise)[pixel_list.index(smart00max)], np.diag(prewhiten_factors**2), return_svd=False)
    #
    # print fgal.Wd(A_ev([map_bd_pbyp[idiot00max],map_bd_pbyp[idiot00max],map_bd_pbyp[idiot00max]]),np.transpose(freq_maps)[pixel_list.index(idiot00max)], np.diag(prewhiten_factors**2), return_svd=False)
    # print fgal.Wd(A_ev([map_bd_pbyp[idiot00max],map_bd_pbyp[idiot00max],map_bd_pbyp[idiot00max]]),np.transpose(noise)[pixel_list.index(idiot00max)], np.diag(prewhiten_factors**2), return_svd=False)

    hp.mollview((np.array(s_smart_00)))
    plt.title('s_smart_00')
    plt.savefig('map_smart_Q_nside=%s'%nside)
    plt.show()
    hp.mollview((np.array(s_smart_10)))
    plt.title('s_smart_10')
    plt.savefig('map_smart_U_nside=%s'%nside)
    plt.show()

    hp.mollview((np.array(s_idiot_00)))
    plt.title('s_idiot_00')
    plt.savefig('map_idiot_Q_nside=%s'%nside)
    plt.show()
    hp.mollview((np.array(s_idiot_10)))
    plt.title('s_idiot_10')
    plt.savefig('map_idiot_U_nside=%s'%nside)
    plt.show()

    plt.hist(np.array([delta_smart_00[p] for p in pixel_list]), histtype='step', label='C.M.B. residuals in Q parameter  with adaptive approach', bins = np.linspace(-0.09, 0.09, num=20))
    plt.hist(np.array([delta_idiot_00[p] for p in pixel_list]), histtype='step', label='C.M.B. residuals in Q parameter  with np approach', bins = np.linspace(-0.09, 0.09, num=20))
    plt.xlabel('C.M.B. residuals')
    plt.ylabel('pixel number')
    plt.title('C.M.B. residuals comparison for Q polarization')
    plt.legend()
    plt.savefig('histo_residuals_Q_nside=%s'%nside)
    plt.show()

    plt.hist(np.array([delta_smart_01[p] for p in pixel_list]), histtype='step', label='C.M.B. residuals in U parameter  with adaptive approach', bins = np.linspace(-0.09, 0.09, num=20))
    plt.hist(np.array([delta_idiot_01[p] for p in pixel_list]), histtype='step', label='C.M.B. residuals in U parameter  with np approach', bins = np.linspace(-0.09, 0.09, num=20))
    plt.xlabel('C.M.B. residuals')
    plt.ylabel('pixel number')
    plt.title('C.M.B. residuals comparison for U polarization')
    plt.legend()
    plt.savefig('histo_residuals_U_nside=%s'%nside)
    plt.show()


    # hist.

    # for i in pixel_list:
        #
        #
        # (sky.cmb(nu)[1,1,:])
        # (sky.cmb(nu)[1,2,:])


#----------------------------------------------------------------histogram with bins corresponding to pbyp patches----------------------------
patch_bin=False
if minimization==True and patch_bin==True:
    index_list=[]
    temp_list5=[]
    binsd=np.linspace(np.amin(input_beta_zeroth_iteration[:len(P_d)]),np.amax(input_beta_zeroth_iteration[:len(P_d)]),pixel_number)
    for j in range(len(P_d_pbyp)):
        for l in range(len(P_d_pbyp[j])):
            index_list.append(pixel_list.index(P_d_pbyp[j][l]))
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[index_list]],binsd,label='patch # %s'%j)
        index_list=[]
    plt.legend()
    plt.title('B_d histogram pbyp patches')
    plt.show()

    binst=np.linspace(np.amin(input_beta_zeroth_iteration[len(P_d):len(P_d)+len(P_t)]),np.amax(input_beta_zeroth_iteration[len(P_d):len(P_d)+len(P_t)]),pixel_number)
    for j in range(len(P_t_pbyp)):
        for l in range(len(P_t_pbyp[j])):
            index_list.append(pixel_list.index(P_t_pbyp[j][l]))
            temp_list5.append(index_list[l]+len(P_d))
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[temp_list5]],binst,label='patch # %s'%j)
        temp_list5=[]
        index_list=[]
    plt.legend()
    plt.title('B_t histogram pbyp patches')
    plt.show()

    binss=np.linspace(np.amin(input_beta_zeroth_iteration[len(P_d)+len(P_t):len(P_d)+len(P_t)+len(P_s)]),np.amax(input_beta_zeroth_iteration[len(P_d)+len(P_t):len(P_d)+len(P_t)+len(P_s)]),pixel_number)
    for j in range(len(P_s_pbyp)):
        for l in range(len(P_s_pbyp[j])):
            index_list.append(pixel_list.index(P_s_pbyp[j][l]))
            temp_list5.append(index_list[l]+len(P_d)+len(P_t))
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[temp_list5]],binss,label='patch # %s'%j)
        temp_list5=[]
        index_list=[]
    plt.legend()
    plt.title('B_s histogram pbyp patches')
    plt.show()




#-----------------------------------------------------------------Display maps of sigma and delta beta squared-----------------------------
plot_sigma_delta_map=False
if plot_sigma_delta_map==True:
    for i in range(len(map_bd_new_sigma)):
        if map_bd_new_sigma[i]==None:
            map_bd_new_sigma[i]=0
            map_bd_new_beta2[i]=0

    for i in range(len(map_bt_new_sigma)):
        if map_bt_new_sigma[i]==None:
            map_bt_new_sigma[i]=0
            map_bt_new_beta2[i]=0

    for i in range(len(map_bs_new_sigma)):
        if map_bs_new_sigma[i]==None:
            map_bs_new_sigma[i]=0
            map_bs_new_beta2[i]=0

    hp.mollview( np.log(np.array(map_bd_new_sigma)) ,sub=(2,2,1))
    plt.title('map_bd_new_sigma')
    hp.mollview(np.log(np.array(map_bd_new_beta2)),sub=(2,2,2))
    plt.title('map_bd_new_deltabeta2')
    plt.show()

    hp.mollview(np.log(np.array(map_bt_new_sigma)),sub=(2,2,1))
    plt.title('map_bt_new_sigma')
    hp.mollview(np.log(np.array(map_bt_new_beta2)),sub=(2,2,2))
    plt.title('map_bt_new_deltabeta2')
    plt.show()

    hp.mollview(np.log(np.array(map_bs_new_sigma)),sub=(2,2,1))
    plt.title('map_bs_new_sigma')
    hp.mollview(np.log(np.array(map_bs_new_beta2)),sub=(2,2,2))
    plt.title('map_bs_new_deltabeta2')
    plt.show()



# print 's_q_d',s_q_d
# hp.mollview(np.array(s_q_00))
# plt.title('s[0][0]')
# plt.show()
#
#
# hp.mollview(np.array(s_q_10))
# plt.title('s[1][0]')
# plt.show()
#
#
# hp.mollview(np.array(s_q_20))
# plt.title('s[2][0]')
# plt.show()
#
# hp.mollview(np.array(s_q_01))
# plt.title('s[0][1]')
# plt.show()
#
# hp.mollview(np.array(s_q_11))
# plt.title('s[1][1]')
# plt.show()
#
# hp.mollview(np.array(s_q_21))
# plt.title('s[2][1]')
# plt.show()
# #
# delta00=[None]*hp.nside2npix(nside)
# delta01=[None]*hp.nside2npix(nside)
# for i in pixel_list:
#     delta00[i]=-(np.array(s_q_00)[i]-nnt_list[i][0][0])+sky.cmb(nu)[1,1,i]
#     delta01[i]=-(np.array(s_q_01)[i]-nnt_list[i][1][1])+sky.cmb(nu)[1,2,i]
#
# hp.mollview(np.array(delta00))
# plt.title('DELTA s[0][0]')
# plt.show()
#
# hp.mollview(np.array(delta01))
# plt.title('DELTA s[0][1]')
# plt.show()


# print 'u_e_v_last[0][1] sigma fisher2',last_values[0][0]

# sigma=sigma_matrix_making(input_beta_zeroth_iteration,P_d,P_t,P_s,last_values)        UNCOMMENT !!!


# print 'u_e_v_last[0][1] sigma fisher3',last_values[0][0]
# print'sigma_t_iter0='
# for j in range(len(P_t)):
#     if np.sqrt(sigma[j+len(P_d)][j+len(P_d)])>=1:
#         print 'p',j
#         print np.sqrt(sigma[j+len(P_d)][j+len(P_d)])
# print('sigma=')
# for i in range(len(sigma)):
#     print(np.sqrt(sigma[i][i]))
# print('shape sigma=',np.shape(sigma))
# print(np.shape(input_beta_zeroth_iteration))

a=delta_beta_matrix_making(P_d,P_t,P_s,input_beta_zeroth_iteration)
# print('delta=',a)

# comp.append(matrix_comparison(a,sigma))     UNCOMMENT !!!

# print('matrix comparison=',comp[0])
# print('input_beta_zeroth_iteration=',input_beta_zeroth_iteration)
map_d1,map_t1,map_s1=patch_map(input_beta_zeroth_iteration,P_d,P_t,P_s,nside)
map_d=[]
map_t=[]
map_s=[]
map_d=map_d1[:]
map_t=map_t1[:]
map_s=map_s1[:]
map_d1=[]
map_t1=[]
map_s1=[]
sigma_list=[]


#--------------------------------------------------------------display map beta zeroth iteration--------------------------------------

i=0
hp.mollview(np.array(map_d),sub=(2,2,1))#,min=bounds[0][0],max=bounds[0][1])
plt.title('beta dust map iteration %s' %(i))
hp.mollview(np.array(map_t),sub=(2,2,2))#,min=bounds[1][0],max=bounds[1][1])
plt.title('dust temp map iteration %s' %(i))
hp.mollview(np.array(map_s),sub=(2,2,3))#,min=bounds[2][0],max=bounds[2][1])
plt.title('beta sync map iteration %s' %(i))
plt.subplot(2,2,4)
plt.plot(comp)
plt.title('Evolution of the norm over iterations')
# plt.savefig('iteration=%s'%i)
plt.show()
# plt.close()
# sigma_list.append(np.linalg.norm(sigma))  UNCOMMENT !!!!

sigma_d_map=[None]*hp.nside2npix(nside)
sigma_t_map=[None]*hp.nside2npix(nside)
sigma_s_map=[None]*hp.nside2npix(nside)

# sigma_d_map,sigma_t_map,sigma_s_map=patch_map(sigma.diagonal(0),P_d,P_t,P_s)
for p in pixel_list:
    sigma_temp=np.linalg.inv(fisher_list[p])
    sigma_d_map[p]=sigma_temp[0][0]
    sigma_t_map[p]=sigma_temp[1][1]
    sigma_s_map[p]=sigma_temp[2][2]
for i in range(len(sigma_d_map)):
    if sigma_d_map[i]==None:
        sigma_d_map[i]=0
for i in range(len(sigma_t_map)):
    if sigma_t_map[i]==None:
        sigma_t_map[i]=0
for i in range(len(sigma_s_map)):
    if sigma_s_map[i]==None:
        sigma_s_map[i]=0

i=0
# print 'sigma.diagonal(0)',sigma.diagonal(0)
# print 'sigma.diagonal(0) shape',np.shape(sigma.diagonal(0))
# print 'beta zeroth iter shape',np.shape(input_beta_zeroth_iteration)
# print 'sigma d map shape',np.shape(sigma_d_map)
# print 'sigma t map shape',np.shape(sigma_t_map)
# print 'sigma s map shape',np.shape(sigma_s_map)
# print 'beta d map shape', np.shape(map_d)
# print 'beta t map shape', np.shape(map_t)
# print 'beta s map shape', np.shape(map_s)
p=0

binsd=range(pixel_number)
# for j in pixel_list:
# print 'coucou', np.linalg.inv(fisher_list[0])
# print 'coucou1',np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])
p_sigmax_d=np.argmax([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])
p_sigmax_t=np.argmax([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list])
p_sigmax_s=np.argmax([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list])

p_sigmax_d=0
p_sigmax_t=0
p_sigmax_s=0

print 'fisher_list[p_sigmax_t]',fisher_list[p_sigmax_t]
print ''
print 'max([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])',max([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])
print 'np.sqrt(np.linalg.inv(fisher_list[x])[0][0])',np.sqrt( np.linalg.inv(fisher_list[pixel_list[p_sigmax_d]])[0][0])
print 'map_d p_sigmax', map_d[pixel_list[p_sigmax_d]]

print 'max([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list])',max([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list])
print 'np.sqrt(np.linalg.inv(fisher_list[x])[1][1])',np.sqrt( np.linalg.inv(fisher_list[pixel_list[p_sigmax_t]])[1][1])
print 'map_t p_sigmax', map_t[pixel_list[p_sigmax_t]]

print 'max([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list])',max([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list])
print 'np.sqrt(np.linalg.inv(fisher_list[x])[2][2])',np.sqrt( np.linalg.inv(fisher_list[pixel_list[p_sigmax_s]])[2][2])
print 'map_s p_sigmax', map_s[pixel_list[p_sigmax_s]]
print ''
# second max :
# sigmax_d=sorted([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])[-2]#p_  =np.argmax([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])
# sigmax_t=sorted([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list])[-2]#p_  =np.argmax([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list])
# sigmax_s=sorted([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list])[-2]#p_  =np.argmax([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list])
# p_sigmax_d=[np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list].index(sigmax_d)
# p_sigmax_t=[np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list].index(sigmax_t)
# p_sigmax_s=[np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list].index(sigmax_s)

print 'p_sigmax_d',p_sigmax_d
print 'pixel_list shape',np.shape(pixel_list)
print 'fisher s list shape',np.shape(fisher_s_list)
print 'np.sqrt(np.linalg.inv(fisher_list[p_sigmax_d])[0][0]',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_d]]))[0][0]
p_sigmax_d_index=pixel_list.index(pixel_list[p_sigmax_d])
print 'p_sigmax_d=',p_sigmax_d,'    ','p_sigmax_d_index=',p_sigmax_d_index
p_sigmax_s_index=pixel_list.index(pixel_list[p_sigmax_s])
p_sigmax_t_index=pixel_list.index(pixel_list[p_sigmax_t])

print 'new minimisation',scipy.optimize.minimize(fun[pixel_list[p_sigmax_s]],[input_set_of_betas[0],input_set_of_betas[len(P_d)],input_set_of_betas[len(P_d)+len(P_t)]],bounds=bounds).x
print 'old minimisation',[map_d[pixel_list[p_sigmax_s]],map_t[pixel_list[p_sigmax_s]],map_s[pixel_list[p_sigmax_s]]]

print ''
print 'fun[5]',fun[5]([1.59,19.6,-3.1])
print 'new minimisation',scipy.optimize.minimize(fun[5],[input_set_of_betas[0],input_set_of_betas[len(P_d)],input_set_of_betas[len(P_d)+len(P_t)]],bounds=bounds).x
print 'old minimisation',[map_d[5],map_t[5],map_s[5]]


x_axis_list=np.arange(bounds[0][0],bounds[0][1],0.01)
fun_list_plot=[]
fun_list=[]
gauss_list=[]
gauss_hess_list=[]
print 'map_t[pixel_list[p_sigmax_t]]',map_t[pixel_list[p_sigmax_t]]
print 'fisher_list[p_sigmax_t]',fisher_list[pixel_list[p_sigmax_t]]
print 'sigma max t=',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_t]])[1][1])

for x in x_axis_list:
    gauss_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_list[pixel_list[p_sigmax_d]])[0][0]) )  *
                                np.exp(-(x-map_d[pixel_list[p_sigmax_d]])*(x-map_d[pixel_list[p_sigmax_d]])/(2*np.linalg.inv (fisher_list[pixel_list[p_sigmax_d]]))[0][0]) ) )
    gauss_hess_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_hess_list[pixel_list[p_sigmax_d]])[0][0]) )  *
                                np.exp(-(x-map_d[pixel_list[p_sigmax_d]])*(x-map_d[pixel_list[p_sigmax_d]])/(2*np.linalg.inv (fisher_hess_list[pixel_list[p_sigmax_d]]))[0][0]) ) )
    fun_list.append(-(1./2.)*fun[pixel_list[p_sigmax_d]]([x,map_t[pixel_list[p_sigmax_d]],map_s[pixel_list[p_sigmax_d]]]))

max_fun=max(fun_list)
max_gauss=max(gauss_list)
min_fun=min(fun_list)
max_gauss_hess=max(gauss_hess_list)
for i in range(len(x_axis_list)):
    fun_list_plot.append(np.exp(fun_list[i]-max_fun) * max_gauss )
    gauss_hess_list[i]=gauss_hess_list[i] * max_gauss / max_gauss_hess
plt.plot(x_axis_list , fun_list_plot,label='fun')
plt.plot(x_axis_list , gauss_list, label='fisher gaussian')
plt.plot(x_axis_list , gauss_hess_list, label='hessian gaussian')
plt.title('gaussian and fun comparison for pixel sigma d max')
plt.legend()
plt.show()

fun_list_plot=[]
fun_list=[]
gauss_list=[]
gauss_hess_list=[]
x_axis_list=[]
x_axis_list=np.arange(bounds[1][0],40,0.01)
for x in x_axis_list:
    gauss_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_list[pixel_list[p_sigmax_t]])[1][1]) )  *
                                np.exp(-(x-map_t[pixel_list[p_sigmax_t]])*(x-map_t[pixel_list[p_sigmax_t]])/(2*np.linalg.inv (fisher_list[pixel_list[p_sigmax_t]]))[1][1]) ) )
    gauss_hess_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_hess_list[pixel_list[p_sigmax_t]])[1][1]) )  *
                                np.exp(-(x-map_t[pixel_list[p_sigmax_t]])*(x-map_t[pixel_list[p_sigmax_t]])/(2*np.linalg.inv (fisher_hess_list[pixel_list[p_sigmax_t]]))[1][1]) ) )
    fun_list.append(-(1./2.)*fun[pixel_list[p_sigmax_t]]([map_d[pixel_list[p_sigmax_t]],x,map_s[pixel_list[p_sigmax_t]]]))

max_fun=max(fun_list)
max_gauss=max(gauss_list)
max_gauss_hess=max(gauss_hess_list)

for i in range(len(x_axis_list)):
    fun_list_plot.append(np.exp(fun_list[i]-max_fun)*max_gauss)
    gauss_hess_list[i]=gauss_hess_list[i] * max_gauss / max_gauss_hess

plt.plot(x_axis_list , fun_list_plot,label='fun')
plt.plot(x_axis_list , gauss_list, label='fisher gaussian')
plt.plot(x_axis_list , gauss_hess_list, label='hessian gaussian')
plt.title('gaussian and fun comparison for pixel sigma t max')
plt.legend()
plt.show()

fun_list_plot=[]
fun_list=[]
gauss_list=[]
gauss_hess_list=[]
x_axis_list=[]
x_axis_list=np.arange(bounds[2][0],bounds[2][1],0.01)

for x in x_axis_list:
    gauss_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_list[pixel_list[p_sigmax_s]])[2][2]) )  *
                                np.exp(-(x-map_s[pixel_list[p_sigmax_s]])*(x-map_s[pixel_list[p_sigmax_s]])/(2*np.linalg.inv (fisher_list[pixel_list[p_sigmax_s]]))[2][2]) ) )
    gauss_hess_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_hess_list[pixel_list[p_sigmax_s]])[2][2]) )  *
                                np.exp(-(x-map_s[pixel_list[p_sigmax_s]])*(x-map_s[pixel_list[p_sigmax_s]])/(2*np.linalg.inv (fisher_hess_list[pixel_list[p_sigmax_s]]))[2][2]) ) )


    fun_list.append(-(1./2.)*fun[pixel_list[p_sigmax_s]]([map_d[pixel_list[p_sigmax_s]],map_t[pixel_list[p_sigmax_s]],x]))
    last_values[pixel_list[p_sigmax_s]]
max_fun=max(fun_list)
max_gauss=max(gauss_list)
max_gauss_hess=max(gauss_hess_list)
print 'np.argmax(fun_list)',np.argmax(fun_list)
print 'x[np.argmax(fun_list)]',x_axis_list[np.argmax(fun_list)]
print 'map_s[pixel_list[p_sigmax_s]]',map_s[pixel_list[p_sigmax_s]]
print 'min(map_s)',min([x for x in map_s if x is not None])
for i in range(len(x_axis_list)):
    fun_list_plot.append(np.exp(fun_list[i]-max_fun)* max_gauss)
    gauss_hess_list[i]=gauss_hess_list[i] * max_gauss / max_gauss_hess
plt.plot(x_axis_list , fun_list_plot,label='fun')
plt.plot(x_axis_list , gauss_list, label='fisher gaussian')
plt.plot(x_axis_list , gauss_hess_list, label='hessian gaussian')
plt.title('gaussian and fun comparison for pixel sigma s max')
plt.legend()
plt.show()


print '_______________________________________________________________________________'
print 'p_sigmax_d',pixel_list[p_sigmax_d]
print 's pmax',s_list[pixel_list[p_sigmax_d]]
print 'beta d pmax',input_beta_zeroth_iteration[p_sigmax_d_index]
print 'sigma d max',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_d]]))[0][0]
print 'nnt_list d max', nnt_list[pixel_list[p_sigmax_d]]
print 'fisher_list pmax',fisher_list[pixel_list[p_sigmax_d]]
print 'inv fisher pmax',np.linalg.inv(fisher_list[pixel_list[p_sigmax_d]])
print 'fisher s',fisher_s_list[pixel_list[p_sigmax_d]]
print 'fisher n',fisher_n_list[pixel_list[p_sigmax_d]]
print ''
print '_______________________________________________________________________________'
print 'p_sigmax_t',pixel_list[p_sigmax_t]
print 's pmax',s_list[pixel_list[p_sigmax_d]]
print 'beta t pmax',input_beta_zeroth_iteration[len(P_t)+p_sigmax_t_index]
print 'sigma t max',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_t]]))[1][1]
print 'nnt_list t max', nnt_list[pixel_list[p_sigmax_t]]
print 'fisher_list pmax',fisher_list[pixel_list[p_sigmax_t]]
print 'inv fisher pmax',np.linalg.inv(fisher_list[pixel_list[p_sigmax_t]])
print 'fisher s',fisher_s_list[pixel_list[p_sigmax_t]]
print 'fisher n',fisher_n_list[pixel_list[p_sigmax_t]]
print 'fun(p)',fun[pixel_list[p_sigmax_t]]([input_beta_zeroth_iteration[p_sigmax_t_index],input_beta_zeroth_iteration[len(P_d)+p_sigmax_t_index],input_beta_zeroth_iteration[len(P_d)+len(P_t)+p_sigmax_t_index]])
print ''
print '_______________________________________________________________________________'
print 'p_sigmax_s',pixel_list[p_sigmax_s]
print 's pmax',s_list[pixel_list[p_sigmax_d]]
print 'beta s pmax',input_beta_zeroth_iteration[len(P_d)+len(P_t)+p_sigmax_s_index]
print 'sigma s max',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_s]]))[2][2]
print 'nnt_list s max', nnt_list[pixel_list[p_sigmax_s]]
print 'fisher_list pmax',fisher_list[pixel_list[p_sigmax_s]]
print 'inv fisher pmax',np.linalg.inv(fisher_list[pixel_list[p_sigmax_s]])
print 'fisher s',fisher_s_list[pixel_list[p_sigmax_s]]
print 'fisher n',fisher_n_list[pixel_list[p_sigmax_s]]
print '_______________________________________________________________________________'



forbidden_p=[]
for p in pixel_list:
    if np.isnan(np.sqrt(np.linalg.inv(fisher_hess_list[p])[0][0])):
        forbidden_p.append(p)

#------------------------------------------------------------SIGMA HISTO------------------------------------------------------
plt.hist( np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list]) ,pixel_number , histtype='step', label='sigma_d histogram')
plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_hess_list[x])[0][0]) for x in pixel_list if x not in forbidden_p]),pixel_number, histtype='step', label='sigma_d_hess histogram')
plt.legend()
plt.show()


forbidden_p=[]
for p in pixel_list:
    if np.isnan(np.sqrt(np.linalg.inv(fisher_hess_list[p])[1][1])):
        forbidden_p.append(p)
plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list]),pixel_number, histtype='step', label='sigma_t histogram')
plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_hess_list[x])[1][1]) for x in pixel_list if x not in forbidden_p]),pixel_number, histtype='step', label='sigma_t_hess histogram')
plt.legend()
plt.show()

forbidden_p=[]
for p in pixel_list:
    if np.isnan(np.sqrt(np.linalg.inv(fisher_hess_list[p])[2][2])):
        forbidden_p.append(p)
plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list]),pixel_number, histtype='step', label='sigma_s histogram')
plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_hess_list[x])[2][2]) for x in pixel_list if x not in forbidden_p]),pixel_number, histtype='step', label='sigma_s_hess histogram')
plt.legend()
plt.show()

#------------------------------------------------------------BETA HISTO------------------------------------------------------

plt.hist(np.array([input_beta_zeroth_iteration[x] for x in range(len(P_d))]),pixel_number/10)#,binsd)
plt.title('beta_d histogram')
plt.show()

plt.hist(np.array([input_beta_zeroth_iteration[x+len(P_d)] for x in range(len(P_t))]),pixel_number/10)#,binsd)
plt.title('beta_t histogram')
plt.show()

plt.hist(np.array([input_beta_zeroth_iteration[x+len(P_d) + len(P_t)] for x in range(len(P_s))]),pixel_number/10)#,binsd)
plt.title('beta_s histogram')
plt.show()

#------------------------------------------------------------SIGNAL/NOISE HISTO------------------------------------------------------

np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])
plt.hist(input_beta_zeroth_iteration[pixel_list.index(x)]  /  np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list]),pixel_number, histtype='step', label='signal/noise d histogram')#,binsd)

plt.hist(input_beta_zeroth_iteration[len(P_d) + pixel_list.index(x)]  /  np.array([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list]),pixel_number, histtype='step', label='signal/noise t histogram')#,binsd)

plt.hist(np.abs(input_beta_zeroth_iteration[len(P_d)+len(P_t) + pixel_list.index(x)] ) /  np.array([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list]),pixel_number, histtype='step', label='signal/noise s histogram')#,binsd)
plt.legend()
plt.show()

for i in range(pixel_number):
    if np.log(np.array(sigma_t_map))[pixel_list[i]]>0:
        print 'p',pixel_list[i]
        print 'sigma=',np.linalg.inv(fisher_list[pixel_list[i]])
        print 's_list[%s]'%pixel_list[i],s_list[pixel_list[i]]
        print 'beta_d=', input_beta_zeroth_iteration[i]
        print 'beta_t=', input_beta_zeroth_iteration[len(P_d)+i]
        print 'beta_s=', input_beta_zeroth_iteration[len(P_d)+len(P_t)+i]
        p+=1
print 'number of pixel with log(sigma t) above 0:',p
p=0
for i in range(len(sigma_s_map)):
    if np.log(np.array(sigma_s_map))[i]>0:
        print 'np.log(np.array(sigma_s_map))[%s]'%i,np.log(np.array(sigma_s_map))[i]
        print 'number of pixel with log(sigma t) above 0:',p
        p+=1

print 'number of pixel with log(sigma s) above 0:',p
p=0
hp.mollview(np.log(np.array(sigma_d_map)),sub=(2,2,1))
plt.title('sigma^2 beta dust map iteration %s' %(i))
hp.mollview(np.log(np.array(sigma_t_map)),sub=(2,2,2))
plt.title('sigma^2 dust temp map iteration %s' %(i))
hp.mollview(np.log(np.array(sigma_s_map)),sub=(2,2,3))
plt.title('sigma^2 beta sync map iteration %s' %(i))
plt.show()



AZERTY

# print('P_d bla=',P_d)
P_d_new,P_t_new,P_s_new,set_of_beta_slice=patch_making(input_beta_zeroth_iteration,sigma,P_d,P_t,P_s)
# print('P_d_new=',P_d_new)
# print('P_d_new=',P_d_new)
# print('P_t_new=',P_t_new)
# print('P_s_new=',P_s_new)
# print('set_of_beta_slice=',set_of_beta_slice)

# hp.mollview(input_beta_zeroth_iteration[:48])
# plt.title('beta dust map')
# plt.show()
# hp.mollview(input_beta_zeroth_iteration[48:96])
# plt.title('dust temp map')
# plt.show()
# hp.mollview(input_beta_zeroth_iteration[96:144])
# plt.title('beta sync map')
# plt.show()


#---------------------------------------------------------------------------ITERATION--------------------------------------------------------------
sigma_list=[]

# for j in range(len(P_t)):
    # templist5=[x+pixel_number for x in P_t[j]]
# plt.hist([x for x in input_beta_zeroth_iteration[2*pixel_number:3*pixel_number]],pixel_number)
#     # del templist5
# plt.show()
# plt.hist([x for x in input_beta_zeroth_iteration[2*pixel_number:120]],120-2*pixel_number)
# plt.hist([x for x in input_beta_zeroth_iteration[120:3*pixel_number]],3*pixel_number-120)
#     # del templist5
# plt.show()


zerty

i=0
h=1
# while comp[i]>0.00021:
for i in range(10):
    map_d1,map_t1,map_s1=patch_map(set_of_beta_slice,P_d_new,P_t_new,P_s_new,nside)
    map_d=[]
    map_t=[]
    map_s=[]
    map_d=map_d1[:]
    map_t=map_t1[:]
    map_s=map_s1[:]
    map_d1=[]
    map_t1=[]
    map_s1=[]
    index_list=[]
    temp_list5=[]

    # print'sigma'
    # for j in range(len(P_d_new)+len(P_t_new)+len(P_s_new)):
    #     print np.sqrt(sigma[j][j])


    binsd=np.linspace(np.amin(input_beta_zeroth_iteration[:len(P_d)]),np.amax(input_beta_zeroth_iteration[:len(P_d)]),pixel_number)
    for j in range(len(P_d_new)):
        for l in range(len(P_d_new[j])):
            index_list.append(pixel_list.index(P_d_new[j][l]))
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[index_list]],binsd)
        index_list=[]
    # plt.savefig('Bd_histo_iteration=%s'%i)
    plt.title('B_d histogram iteration=%s'%i)
    plt.show()

    binst=np.linspace(np.amin(input_beta_zeroth_iteration[len(P_d):len(P_d)+len(P_t)]),np.amax(input_beta_zeroth_iteration[len(P_d):len(P_d)+len(P_t)]),pixel_number)
    for j in range(len(P_t_new)):
        for l in range(len(P_t_new[j])):
            index_list.append(pixel_list.index(P_t_new[j][l]))
            temp_list5.append(index_list[l]+len(P_d))
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[temp_list5]],binst)
        temp_list5=[]
        index_list=[]
    # plt.savefig('Bt_histo_iteration=%s'%i)
    plt.title('B_t histogram iteration=%s'%i)
    plt.show()

    binss=np.linspace(np.amin(input_beta_zeroth_iteration[len(P_d)+len(P_t):len(P_d)+len(P_t)+len(P_s)]),np.amax(input_beta_zeroth_iteration[len(P_d)+len(P_t):len(P_d)+len(P_t)+len(P_s)]),pixel_number)
    for j in range(len(P_s_new)):
        for l in range(len(P_s_new[j])):
            index_list.append(pixel_list.index(P_s_new[j][l]))
            temp_list5.append(index_list[l]+len(P_d)+len(P_t))
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[temp_list5]],binss)
        temp_list5=[]
        index_list=[]
    # plt.savefig('Bs_histo_iteration=%s'%i)
    plt.title('B_s histogram iteration=%s'%i)
    plt.show()


    hp.mollview(np.array(map_d),sub=(2,2,1),min=bounds[0][0],max=bounds[0][1])
    plt.title('beta dust map iteration %s' %(i))
    hp.mollview(np.array(map_t),sub=(2,2,2),min=bounds[1][0],max=bounds[1][1])
    plt.title('dust temp map iteration %s' %(i))
    hp.mollview(np.array(map_s),sub=(2,2,3),min=bounds[2][0],max=bounds[2][1])
    plt.title('beta sync map iteration %s' %(i))
    plt.subplot(2,2,4)
    plt.plot(comp)
    plt.title('Evolution of the norm over iterations')
    # plt.savefig('iteration=%s'%i)
    plt.show()
    # plt.close()
    sigma_list.append(np.linalg.norm(sigma))


    sigma_d_map,sigma_t_map,sigma_s_map=patch_map(sigma.diagonal(0),P_d_new,P_t_new,P_s_new,nside)

    hp.mollview(np.array(sigma_d_map),sub=(2,2,1))
    plt.title('sigma^2 beta dust map iteration %s' %(i))
    hp.mollview(np.array(sigma_t_map),sub=(2,2,2))
    plt.title('sigma^2 dust temp map iteration %s' %(i))
    hp.mollview(np.array(sigma_s_map),sub=(2,2,3))
    plt.title('sigma^2 beta sync map iteration %s' %(i))
    plt.show()


    # minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,set_of_beta_slice,(P_d_new,P_t_new,P_s_new,nside))
    minimization_result_pixel=[]
    minimization_result=[]
    for p in range(pixel_number):
        # print 'p',p
        minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]]),bounds=bounds).x)
    for j in range(pixel_number):
        minimization_result.append(minimization_result_pixel[j][0])
    for j in range(pixel_number):
        minimization_result.append(minimization_result_pixel[j][1])
    for j in range(pixel_number):
        minimization_result.append(minimization_result_pixel[j][2])
    minimization_result_pixel=[]


    sigma=sigma_matrix_making(minimization_result,P_d_new,P_t_new,P_s_new,last_values)
    # for j in range(len(sigma)):
    #     print(np.sqrt(sigma[j][j]))
    delta_b=delta_beta_matrix_making(P_d_new,P_t_new,P_s_new,input_beta_zeroth_iteration)
    # for j in range(len(delta_b)):
    #     print(np.sqrt(delta_b[j][j]))
    comp.append(matrix_comparison(delta_b,sigma))
    # print('comp%s='%i,comp[i])
    # print('set_of_beta_slice1=',set_of_beta_slice)
    set_of_beta_slice=[]
    P_d_new1,P_t_new1,P_s_new1,set_of_beta_slice1=patch_making(input_beta_zeroth_iteration,sigma,P_d_new,P_t_new,P_s_new)
    # print('set_of_beta_slice1=',set_of_beta_slice1)
    P_d_new=[]
    P_t_new=[]
    P_s_new=[]
    set_of_beta_slice=[]
    P_d_new=P_d_new1[:]
    P_t_new=P_t_new1[:]
    P_s_new=P_s_new1[:]
    set_of_beta_slice=set_of_beta_slice1[:]
    P_d_new1=[]
    P_t_new1=[]
    P_s_new1=[]
    set_of_beta_slice1=[]
    # print('set_of_beta_slice2=',set_of_beta_slice)
    i+=1
# iteration_number=5
# smart=iteration(iteration_number,input_beta_zeroth_iteration,P_d_new,P_t_new,P_s_new,set_of_beta_slice)
# smart=scipy.optimize.minimize(iteration,iteration_number,(input_beta_zeroth_iteration,P_d_new,P_t_new,P_s_new,set_of_beta_slice))

sigma_list.append(np.linalg.norm(sigma))
# print(smart)
# print(smart[1])
map_d,map_t,map_s=patch_map(set_of_beta_slice,P_d_new,P_t_new,P_s_new,nside)
hp.mollview(np.array(map_d),sub=(2,2,1),min=bounds[0][0],max=bounds[0][1])
plt.title('beta dust map iteration %s' %(i))
hp.mollview(np.array(map_t),sub=(2,2,2),min=bounds[1][0],max=bounds[1][1])
plt.title('dust temp map iteration %s' %(i))
hp.mollview(np.array(map_s),sub=(2,2,3),min=bounds[2][0],max=bounds[2][1])
plt.title('beta sync map iteration %s' %(i))
plt.subplot(2,2,4)
plt.plot(comp)
plt.title('Evolution of the norm over iterations')
# plt.savefig('iteration=%s'%i)
plt.show()
# plt.close

plt.plot(sigma_list)
plt.title('Evolution of sigma norm')
plt.show()
