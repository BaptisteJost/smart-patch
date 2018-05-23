# import os
#
# path_to_fgbuster = '/mnt/c/Users/Baptiste/fgbuster'
# os.sys.path.insert(0, os.path.realpath(path_to_fgbuster))


import numpy as np
import healpy as hp
import pylab as py
import scipy
import matplotlib.pyplot as plt
#import fgbuster as fg
from fgbuster.pysm_helpers import get_instrument, get_sky
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.separation_recipies import _get_prewhiten_factors,_A_evaluators
import fgbuster.algebra as fgal
import numdifftools
#P_i = list of list for component i, each list corresponds to the pixel of one patch, i.e. P_d=[patch0, patch1,...,patchn] & patchi=[pixel0,pixel1,...]
#input_set_of_betas is a list of all betas for each component and patch i.e. input_set_of_betas=[beta_d1,beta_d2,...,beta_dn,beta_s1,...,beta_sm,temp1,...,templ]
    #by convention the component are ordered as beta_dust,beta_sync,temperature_dust


#------------------------------------------------------SKY GENERATION----------------------------------------------------
nside = 8
pixel_number=hp.nside2npix(nside)
sky = get_sky(nside, 'c1d1s1')
nu = np.array([40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1])
# hp.mollview(sky.cmb(nu)[1,1,:])
# plt.show()
# hp.mollview(sky.cmb(nu)[1,2,:])
# plt.show()
instrument = get_instrument(nside, 'litebird')
freq_maps = instrument.observe(sky, write_outputs=False)[0] +instrument.observe(sky, write_outputs=False)[1]
freq_maps=freq_maps[:,1:,:]  #on retire la temperature (1:) car on ne prend en compte que Q&U pas I
components = [CMB(), Dust(150.), Synchrotron(20.)]
prewhiten_factors = _get_prewhiten_factors(instrument, freq_maps.shape)         # correspond a N^-1/2
print 'freq_maps shape',np.shape(freq_maps)



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

# input_set_of_betas=[1.59,1.59,1.59,1.59,19.6,19.6,19.6,19.6,-3.1,-3.1,-3.1,-3.1]

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



def data_patch(freq_maps,P_i): #this function returns the data corresponding to the pixels in the input patch. Works for patches as well as for meta patches
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

def data_pixel(freq_maps,P_i): #this function returns the data corresponding to the pixels in the input patch. Works for patches as well as for meta patches
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


def patch_map(input_set_of_betas,P_d,P_t,P_s):
    ind=0
    pixel_number1=hp.nside2npix(nside)
    map_bd=[None]*pixel_number1
    map_bt=[None]*pixel_number1
    map_bs=[None]*pixel_number1
    # print 'input_set_of_betas', input_set_of_betas
    for i in range(len(P_d)):
        for j in range(len(P_d[i])):
            map_bd[P_d[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_t)):
        for j in range(len(P_t[i])):
            map_bt[P_t[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in range(len(P_s[i])):
            map_bs[P_s[i][j]]=input_set_of_betas[ind]
        # print('index=',ind)
        ind+=1
    return map_bd,map_bt,map_bs


def joint_spectral_likelihood(input_set_of_betas,P_d,P_t,P_s,data,p): #computes the joint likelihood for the whole sky taking into account the patches
    #prewhiten_factors must be defined above !
    logL_spec=0
    #first we have to create the maps of beta in the sky to link each patch with its corresponding beta value
    ind=0
    pixel_number1=hp.nside2npix(nside)
    map_bd=[None]*pixel_number1
    map_bt=[None]*pixel_number1
    map_bs=[None]*pixel_number1
    # print('len(P_d)=',len(P_d))
    # print('len(P_d[0])=',len(P_d[0]))
    # print('np.shape(input_set_of_betas)=',np.shape(input_set_of_betas))
    # print(P_d)
    for i in range(len(P_d)):
        # print('map_bd=',np.shape(map_bd))
        # print 'i=',i
        for j in P_d[i]:
            # print 'j=',j
            map_bd[j]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_t)):
        for j in P_t[i]:
            map_bt[j]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in P_s[i]:
            map_bs[j]=input_set_of_betas[ind]
        ind+=1

    # for p in pixel_list:
        # print(fun[p]([map_bd[p],map_bt[p],map_bs[p]]))
    # print 'fun[p]([map_bd[p],map_bt[p],map_bs[p]])',fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    # print 'input_set_of_betas',input_set_of_betas
    logL_spec=fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    del map_bd,map_bt,map_bs
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

def fisher_pixel(map_bd,map_bt,map_bs):
    fisher_list=[None]*hp.nside2npix(nside)
    # fisher_list_N=[None]*hp.nside2npix(nside)
    s_q_00=[None]*hp.nside2npix(nside)
    s_q_10=[None]*hp.nside2npix(nside)
    s_q_20=[None]*hp.nside2npix(nside)
    s_q_01=[None]*hp.nside2npix(nside)
    s_q_11=[None]*hp.nside2npix(nside)
    s_q_21=[None]*hp.nside2npix(nside)
    s_list=[None]*hp.nside2npix(nside)
    nnt_list=[None]*hp.nside2npix(nside)
    fisher_s_list=[None]*hp.nside2npix(nside)
    fisher_n_list=[None]*hp.nside2npix(nside)
    i=0
    for p in pixel_list:
        # fun[p]([map_bd[p],map_bt[p],map_bs[p]])#[minimization_result.x[l*3],minimization_result.x[l*3+1],minimization_result.x[l*3+2]]
        # u_e_v_last, A_dB_last, x_last, pw_d = last_values[p]
        # print 'u_e_v_last[0][1] sigma fisher0 pixel p:%s'%p,u_e_v_last[0][1]

        # s =fgal._Wd_svd(u_e_v_last[0], pw_d[0])
        prewhitened_data = prewhiten_factors * np.transpose(freq_maps)[i]
        s=fgal.Wd(A_ev([map_bd[p],map_bt[p],map_bs[p]]),prewhitened_data, None, return_svd=False)
        i+=1
        # print '_invAtNA_svd(u_e_v_last[0])',fgal._invAtNA_svd(u_e_v_last[0])
        # print 's-n',s-fgal._invAtNA_svd(u_e_v_last[0])
        # print '_invAtNA_svd(u_e_v_last[0]) shape', np.shape(fgal._invAtNA_svd(u_e_v_last[0]))
        # print 's shape',np.shape(s)
        # print 's',s

        # nnt_list[p]=fgal._invAtNA_svd(u_e_v_last[0])
        # fisher_s_list[p]=fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s, A_dB_last[0], comp_of_dB)
        # fisher_n_list[p]=fgal._fisher_logL_dB_dB_svd(u_e_v_last[0],fgal._invAtNA_svd(u_e_v_last[0]),A_dB_last[0], comp_of_dB)
        # if A_dB_ev is None:
        #     fisher = numdifftools.Hessian(fun)(minimization_result.x)
        # else:
        # print 'u_e_v_last[0][1] sigma fisher1 pixel p:%s'%p,u_e_v_last[0][1]

        # fisher_list[p]= fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s, A_dB_last[0], comp_of_dB)# - fgal._fisher_logL_dB_dB_svd(u_e_v_last[0],fgal._invAtNA_svd(u_e_v_last[0]),A_dB_last[0], comp_of_dB)

        # fisher_list_N[p]=
        fisher_list[p]=fgal.fisher_logL_dB_dB(A_ev([map_bd[p],map_bt[p],map_bs[p]]), s, A_dB_ev([map_bd[p],map_bt[p],map_bs[p]]), comp_of_dB, None, return_svd=False)
        s_q_00[p]=s.T[0][0]
        s_q_10[p]=s.T[1][0]
        s_q_20[p]=s.T[2][0]
        s_q_01[p]=s.T[0][1]
        s_q_11[p]=s.T[1][1]
        s_q_21[p]=s.T[2][1]
        s_list[p]=s.T

        # print 'fisher_list[p]',fisher_list[p]
        # print 'fisher_list_N[p]',fisher_list_N[p]
        # if np.log(1/fisher_list[p][0][0])>0:
        #     print 'fisher_list[%s][1][1]'%p,np.log(1/fisher_list[p][1][1])

        if np.isnan(s[0][0]) or np.isnan(s[0][1]) or np.isnan(s[0][2]) or np.isnan(s[1][0]) or np.isnan(s[1][1]) or np.isnan(s[1][2]):
            print ' /!\ NAN in S /!\ '
            print 'at pixel %s'%p
            print 's',s
            print 'u_e_v_last[0][1]',u_e_v_last[0][1]
            print 'pw_d[0]',pw_d[0]
    return fisher_list,s_q_00,s_q_10,s_q_20,s_q_01,s_q_11,s_q_21,s_list,nnt_list,fisher_s_list,fisher_n_list


def patch_making_pbyp(input_beta_zeroth_iteration,fisher_list):

    sorted_beta_d_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k])
    sorted_beta_t_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k+pixel_number])
    sorted_beta_s_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k+pixel_number*2])

    # print 'sorted beta d'
    # for p in range(pixel_number):
        # print 'pixel_list[sorted_beta_d_pixel_list[p]]',pixel_list[sorted_beta_d_pixel_list[p]]
        # print 'sorted_beta_d_pixel_list[p]',sorted_beta_d_pixel_list[p]
        # print 'p',p
        # print input_beta_zeroth_iteration[sorted_beta_d_pixel_list[p]]

    # print 'sorted beta t'
    # for p in range(pixel_number):
    #     print input_beta_zeroth_iteration[sorted_beta_t_pixel_list[p]+pixel_number]
    #
    # print 'sorted beta s'
    # for p in range(pixel_number):
    #     print input_beta_zeroth_iteration[sorted_beta_s_pixel_list[p]+2*pixel_number]

    P_d_new=[]
    P_t_new=[]
    P_s_new=[]

    sigma_patch_temp=0 #arbitrary
    delta_beta_temp=1 #arbitrary
    pixel_patch_temp=[]
    p_list=[]
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
        p+=1
        while sigma_patch_temp <= delta_beta_temp*delta_beta_temp:
            if p==pixel_number:
                print 'pixel limit reached'
                print 'p',p
                break

            # print 'p',p
            # print 'pixel_list[sorted_beta_d_pixel_list[p]]',pixel_list[sorted_beta_d_pixel_list[p]]
            fisher_tot+=fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]

            sigma_patch_temp=(np.linalg.inv(fisher_tot).T)[0][0]
            pixel_patch_temp.append(pixel_list[sorted_beta_d_pixel_list[p]])
            p_list.append(p)
            # print '[input_beta_zeroth_iteration[x] for x in pixel_patch_temp]',[input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list]
            delta_beta_temp=np.std([input_beta_zeroth_iteration[x] for x in pixel_patch_temp])
            # print 'db^2=',delta_beta_temp*delta_beta_temp
            # print 'sigma=',sigma_patch_temp
            p+=1
        patchnum+=1
        print 'NEW PATCH ! patch number=',patchnum

        P_d_new.append(pixel_patch_temp)
        pixel_patch_temp=[]
        p_list=[]
        sigma_patch_temp=0 #arbitrary
        delta_beta_temp=1 #arbitrary
        fisher_tot=0
    print 'OUT OF THE LOOP'
    return P_d_new


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
        minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,set_of_beta_slice,(P_d_new,P_t_new,P_s_new,freq_maps))

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

        #use remove !! le pixel ne sera présent que une seule fois par liste anyway
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


def return_fun(pixel,prewhitened_data):
    return lambda x: -fgal.logL(A_ev(x), prewhitened_data,None)

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
# #fun prend en argument les beta sur le nombre de pixel considéré dans prewhitened_data
# minimization_check=scipy.optimize.minimize(fun,[1.59,19.6,-3.1])
#
#print(minimization_check)
# data=data_patch(freq_maps,meta_P)
# print(joint_spectral_likelihood(input_set_of_betas,P_d,P_t,P_s,freq_maps))


#-----------------------------------------------ZEROTH ITERATION----------------------------------------------------------------------
P_d=[]
P_s=[]
P_t=[]
h=0
for i in range(len(freq_maps[0][1])): P_d.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_s.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_t.append([pixel_list[i]])
# print('P_d first=',P_d)
# P_d=pixel_list[:]
# P_s=pixel_list[:]
# P_t=pixel_list[:]

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
bounds=((1,8),(10,30),(-8,-0.05))


fun=[]
last_values=[]
data=data_patch(freq_maps,P_d)
A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=prewhiten_factors)
print 'param',params
# print 'A_ev(0)',A_ev(0)
# print 'A_ev',A_ev
#A_local=A_ev(beta_pix)
# print 'A_dB_ev',A_dB_ev
A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)
# print 'A_dB_ev',A_dB_ev

# for i in pixel_list:
# print('pixel_number=',pixel_number)
# print('data shape=',np.shape(data))
# print('data[0] transpose shape',np.shape(np.transpose(data[0])))
# print('freq_maps shape',np.shape(np.transpose(freq_maps)[0]))
# print('freq_maps shape',np.shape(freq_maps))
# print('data shape=',np.shape(data[pixel_number]))
# print i


fun=[None]*hp.nside2npix(nside)
last_values=[None]*hp.nside2npix(nside)

for l in range(pixel_number):
     prewhitened_data = prewhiten_factors * np.transpose(freq_maps)[l]
     # print 'np.transpose(freq_maps)[%s]'%l,np.transpose(freq_maps)[l]
     # fun[pixel_list[l]] = lambda x: -fgal.logL(A_ev(x), np.transpose(freq_maps)[l],np.diag(prewhiten_factors**2))#np.transpose(freq_maps)[l], np.diag(prewhiten_factors**2))
     # fun_test.append(lambda x: -fgal.logL(A_ev(x), np.transpose(freq_maps)[l],np.diag(prewhiten_factors**2)))
     fun[pixel_list[l]] = return_fun(l,prewhitened_data)
     # print 'fun',fun
     # print 'fun[pixel_list[l]]',fun[pixel_list[l]]([input_set_of_betas[150],input_set_of_betas[150+pixel_number],input_set_of_betas[150+2*pixel_number]])
     # last_values[pixel_list[l]]=last_valuestemp
     # print 'uev pixel:%s'%l,last_values[0]
     # del funtemp#, jactemp,last_valuestemp
# print 'fun',fun[0]
# print fun

# print '150',fun[150]([1.59,19.6,-3.1])
# print '300',fun[pixel_list[300]]([input_set_of_betas[300],input_set_of_betas[300+pixel_number],input_set_of_betas[300+2*pixel_number]])
# for p in range(hp.nside2npix(nside)):
#     funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
#     fun.append(funtemp)
#     last_values.append(last_valuestemp)
#     print(funtemp([1.59,19.6,-3.1]))

comp=[]
minimization_result_pixel=[]
input_beta_zeroth_iteration=[]
for p in range(pixel_number):
    # print 'fun[150] boucle %s'%p,fun[150]([1.59,19.6,-3.1])
    minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]]),bounds=bounds).x)
    if p == 150:
        # print 'fun[150] if',fun[150]([1.59,19.6,-3.1])
        print 'minimization_result_pixel=',minimization_result_pixel[-1]
        print scipy.optimize.minimize(fun[pixel_list[p]],([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]]),bounds=bounds)
    if p == range(pixel_number)[-1]:
        print 'minimization_result_pixel=',minimization_result_pixel[-1]
        print scipy.optimize.minimize(fun[pixel_list[p]],([input_set_of_betas[p],input_set_of_betas[p+pixel_number],input_set_of_betas[p+2*pixel_number]]),bounds=bounds)
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][0])
    # print 'fun[150] for i 1',fun[150]([1.59,19.6,-3.1])
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][1])
    # print 'fun[150] for i 2',fun[150]([1.59,19.6,-3.1])
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][2])
    # print 'fun[150] for i 3',fun[150]([1.59,19.6,-3.1])
minimization_result_pixel=[]

# print 'fun[150]',fun[150]([1.59,19.6,-3.1])

# input_beta_zeroth_iteration=minimization_result.x

# print 'minimisation results pixel 0',input_beta_zeroth_iteration[0],input_beta_zeroth_iteration[pixel_number],input_beta_zeroth_iteration[pixel_number*2]
# azert
map_bd,map_bt,map_bs=patch_map(input_beta_zeroth_iteration,P_d,P_t,P_s)
fisher_list,s_q_00,s_q_10,s_q_20,s_q_01,s_q_11,s_q_21,s_list,nnt_list,fisher_s_list,fisher_n_list=fisher_pixel(map_bd,map_bt,map_bs)
P_d_test=patch_making_pbyp(input_beta_zeroth_iteration,fisher_list)
print '[np.sqrt(np.linalg.inv(fisher_list[x])[1][1]',sorted([np.sqrt(np.linalg.inv(fisher_list[x])[1][1])for x in pixel_list]) [-1]
print 'P_d_test shape',np.shape(P_d_test)
# azertyuiops


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
#
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
map_d1,map_t1,map_s1=patch_map(input_beta_zeroth_iteration,P_d,P_t,P_s)
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

i=0
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

print 'new minimisation',scipy.optimize.minimize(fun[pixel_list[p_sigmax_s]],[input_set_of_betas[0],input_set_of_betas[pixel_number],input_set_of_betas[2*pixel_number]],bounds=bounds).x
print 'old minimisation',[map_d[pixel_list[p_sigmax_s]],map_t[pixel_list[p_sigmax_s]],map_s[pixel_list[p_sigmax_s]]]

print ''
print 'fun[150]',fun[150]([1.59,19.6,-3.1])
print 'new minimisation',scipy.optimize.minimize(fun[150],[input_set_of_betas[0],input_set_of_betas[pixel_number],input_set_of_betas[2*pixel_number]],bounds=bounds).x
print 'old minimisation',[map_d[150],map_t[150],map_s[150]]


x_axis_list=np.arange(bounds[0][0],15,0.01)
fun_list_plot=[]
fun_list=[]
gauss_list=[]
print 'map_t[pixel_list[p_sigmax_t]]',map_t[pixel_list[p_sigmax_t]]
print 'fisher_list[p_sigmax_t]',fisher_list[pixel_list[p_sigmax_t]]
print 'sigma max t=',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_t]])[1][1])

for x in x_axis_list:
    gauss_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_list[pixel_list[p_sigmax_d]])[0][0]) )  *
                                np.exp(-(x-map_d[pixel_list[p_sigmax_d]])*(x-map_d[pixel_list[p_sigmax_d]])/(2*np.sqrt(np.linalg.inv (fisher_list[pixel_list[p_sigmax_d]]))[0][0]) ) ))
    fun_list.append(-(1./2.)*fun[pixel_list[p_sigmax_d]]([x,map_t[pixel_list[p_sigmax_d]],map_s[pixel_list[p_sigmax_d]]]))

max_fun=max(fun_list)
max_gauss=max(gauss_list)
min_fun=min(fun_list)
for i in range(len(x_axis_list)):
    fun_list_plot.append(np.exp(fun_list[i]-max_fun) * max_gauss )#* np.exp(-min_fun))
plt.plot(x_axis_list , fun_list_plot,label='fun')
plt.plot(x_axis_list , gauss_list, label='gaussian')
plt.title('gaussian and fun comparison for pixel sigma d max')
plt.legend()
plt.show()

fun_list_plot=[]
fun_list=[]
gauss_list=[]
x_axis_list=[]
x_axis_list=np.arange(bounds[1][0],600,0.01)
for x in x_axis_list:
    gauss_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_list[pixel_list[p_sigmax_t]])[1][1]) )  *
                                np.exp(-(x-map_t[pixel_list[p_sigmax_t]])*(x-map_t[pixel_list[p_sigmax_t]])/(2*np.sqrt(np.linalg.inv (fisher_list[pixel_list[p_sigmax_t]]))[1][1]) ) ))
    fun_list.append(-(1./2.)*fun[pixel_list[p_sigmax_t]]([map_d[pixel_list[p_sigmax_t]],x,map_s[pixel_list[p_sigmax_t]]]))
# min_fun=min(fun_list)
max_fun=max(fun_list)
max_gauss=max(gauss_list)
# print 'min_fun=',min_fun,'  ','max_fun=',max_fun
for i in range(len(x_axis_list)):
    fun_list_plot.append(np.exp(fun_list[i]-max_fun)*max_gauss)
    # print 'fun_list_plot[%s]'%i,fun_list_plot[i]
plt.plot(x_axis_list , fun_list_plot,label='fun')
plt.plot(x_axis_list , gauss_list, label='gaussian')
plt.title('gaussian and fun comparison for pixel sigma t max')
plt.legend()
plt.show()

fun_list_plot=[]
fun_list=[]
gauss_list=[]
x_axis_list=[]
x_axis_list=np.arange(-20,0,0.01)

for x in x_axis_list:
    gauss_list.append((    ( 1  /  np.sqrt(2*np.pi*np.linalg.inv (fisher_list[pixel_list[p_sigmax_s]])[2][2]) )  *
                                np.exp(-(x-map_s[pixel_list[p_sigmax_s]])*(x-map_s[pixel_list[p_sigmax_s]])/(2*np.sqrt(np.linalg.inv (fisher_list[pixel_list[p_sigmax_s]]))[2][2]) ) ))
    fun_list.append(-(1./2.)*fun[pixel_list[p_sigmax_s]]([map_d[pixel_list[p_sigmax_s]],map_t[pixel_list[p_sigmax_s]],x]))
    last_values[pixel_list[p_sigmax_s]]
max_fun=max(fun_list)
max_gauss=max(gauss_list)

print 'np.argmax(fun_list)',np.argmax(fun_list)
print 'x[np.argmax(fun_list)]',x_axis_list[np.argmax(fun_list)]
print 'map_s[pixel_list[p_sigmax_s]]',map_s[pixel_list[p_sigmax_s]]
print 'min(map_s)',min([x for x in map_s if x is not None])
for i in range(len(x_axis_list)):
    fun_list_plot.append(np.exp(fun_list[i]-max_fun)* max_gauss)
plt.plot(x_axis_list , fun_list_plot,label='fun')
plt.plot(x_axis_list , gauss_list, label='gaussian')
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
print 'beta t pmax',input_beta_zeroth_iteration[pixel_number+p_sigmax_t_index]
print 'sigma t max',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_t]]))[1][1]
print 'nnt_list t max', nnt_list[pixel_list[p_sigmax_t]]
print 'fisher_list pmax',fisher_list[pixel_list[p_sigmax_t]]
print 'inv fisher pmax',np.linalg.inv(fisher_list[pixel_list[p_sigmax_t]])
print 'fisher s',fisher_s_list[pixel_list[p_sigmax_t]]
print 'fisher n',fisher_n_list[pixel_list[p_sigmax_t]]
print 'fun(p)',fun[pixel_list[p_sigmax_t]]([input_beta_zeroth_iteration[p_sigmax_t_index],input_beta_zeroth_iteration[pixel_number+p_sigmax_t_index],input_beta_zeroth_iteration[2*pixel_number+p_sigmax_t_index]])
print ''
print '_______________________________________________________________________________'
print 'p_sigmax_s',pixel_list[p_sigmax_s]
print 's pmax',s_list[pixel_list[p_sigmax_d]]
print 'beta s pmax',input_beta_zeroth_iteration[pixel_number*2+p_sigmax_s_index]
print 'sigma s max',np.sqrt(np.linalg.inv(fisher_list[pixel_list[p_sigmax_s]]))[2][2]
print 'nnt_list s max', nnt_list[pixel_list[p_sigmax_s]]
print 'fisher_list pmax',fisher_list[pixel_list[p_sigmax_s]]
print 'inv fisher pmax',np.linalg.inv(fisher_list[pixel_list[p_sigmax_s]])
print 'fisher s',fisher_s_list[pixel_list[p_sigmax_s]]
print 'fisher n',fisher_n_list[pixel_list[p_sigmax_s]]
print '_______________________________________________________________________________'

#------------------------------------------------------------SIGMA HISTO------------------------------------------------------
plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list]),pixel_number)#,binsd)
plt.title('sigma_d histogram')
plt.show()

plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list]),pixel_number)#,binsd)
plt.title('sigma_t histogram')
plt.show()

plt.hist(np.array([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list]),pixel_number)#,binsd)
plt.title('sigma_s histogram')
plt.show()

#------------------------------------------------------------BETA HISTO------------------------------------------------------

plt.hist(np.array([input_beta_zeroth_iteration[x] for x in range(pixel_number)]),pixel_number)#,binsd)
plt.title('beta_d histogram')
plt.show()

plt.hist(np.array([input_beta_zeroth_iteration[x+pixel_number] for x in range(pixel_number)]),pixel_number)#,binsd)
plt.title('beta_t histogram')
plt.show()

plt.hist(np.array([input_beta_zeroth_iteration[x+pixel_number*2] for x in range(pixel_number)]),pixel_number)#,binsd)
plt.title('beta_s histogram')
plt.show()

#------------------------------------------------------------SIGNAL/NOISE HISTO------------------------------------------------------
# print 'np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])=',np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])
np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list])
plt.hist(input_beta_zeroth_iteration[pixel_list.index(x)]  /  np.array([np.sqrt(np.linalg.inv(fisher_list[x])[0][0]) for x in pixel_list]),pixel_number)#,binsd)
plt.title('signal/noise d histogram')
plt.show()

plt.hist(input_beta_zeroth_iteration[pixel_number + pixel_list.index(x)]  /  np.array([np.sqrt(np.linalg.inv(fisher_list[x])[1][1]) for x in pixel_list]),pixel_number)#,binsd)
plt.title('signal/noise t histogram')
plt.show()

plt.hist(input_beta_zeroth_iteration[pixel_number*2 + pixel_list.index(x)]  /  np.array([np.sqrt(np.linalg.inv(fisher_list[x])[2][2]) for x in pixel_list]),pixel_number)#,binsd)
plt.title('signal/noise s histogram')
plt.show()

azert

for i in range(pixel_number):
    if np.log(np.array(sigma_t_map))[pixel_list[i]]>0:
        print 'p',pixel_list[i]
        print 'sigma=',np.linalg.inv(fisher_list[pixel_list[i]])
        print 's_list[%s]'%pixel_list[i],s_list[pixel_list[i]]
        print 'beta_d=', input_beta_zeroth_iteration[i]
        print 'beta_t=', input_beta_zeroth_iteration[pixel_number+i]
        print 'beta_s=', input_beta_zeroth_iteration[2*pixel_number+i]
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

# print 'np.log(np.array(sigma_d_map)',np.log(np.array(sigma_d_map))
# print 's_q_d',s_q_d




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
    map_d1,map_t1,map_s1=patch_map(set_of_beta_slice,P_d_new,P_t_new,P_s_new)
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


    binsd=np.linspace(np.amin(input_beta_zeroth_iteration[:pixel_number]),np.amax(input_beta_zeroth_iteration[:pixel_number]),pixel_number)
    for j in range(len(P_d_new)):
        for l in range(len(P_d_new[j])):
            index_list.append(pixel_list.index(P_d_new[j][l]))
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[index_list]],binsd)
        index_list=[]
    # plt.savefig('Bd_histo_iteration=%s'%i)
    plt.title('B_d histogram iteration=%s'%i)
    plt.show()



    binst=np.linspace(np.amin(input_beta_zeroth_iteration[pixel_number:2*pixel_number]),np.amax(input_beta_zeroth_iteration[pixel_number:2*pixel_number]),pixel_number)
    for j in range(len(P_t_new)):
        for l in range(len(P_t_new[j])):
            index_list.append(pixel_list.index(P_t_new[j][l]))
            temp_list5.append(index_list[l]+pixel_number)
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in np.array(input_beta_zeroth_iteration)[temp_list5]],binst)
        temp_list5=[]
        index_list=[]
    # plt.savefig('Bt_histo_iteration=%s'%i)
    plt.title('B_t histogram iteration=%s'%i)
    plt.show()

    binss=np.linspace(np.amin(input_beta_zeroth_iteration[2*pixel_number:3*pixel_number]),np.amax(input_beta_zeroth_iteration[2*pixel_number:3*pixel_number]),pixel_number)
    for j in range(len(P_s_new)):
        for l in range(len(P_s_new[j])):
            index_list.append(pixel_list.index(P_s_new[j][l]))
            temp_list5.append(index_list[l]+2*pixel_number)
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


    sigma_d_map,sigma_t_map,sigma_s_map=patch_map(sigma.diagonal(0),P_d_new,P_t_new,P_s_new)

    hp.mollview(np.array(sigma_d_map),sub=(2,2,1))
    plt.title('sigma^2 beta dust map iteration %s' %(i))
    hp.mollview(np.array(sigma_t_map),sub=(2,2,2))
    plt.title('sigma^2 dust temp map iteration %s' %(i))
    hp.mollview(np.array(sigma_s_map),sub=(2,2,3))
    plt.title('sigma^2 beta sync map iteration %s' %(i))
    plt.show()


    # minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,set_of_beta_slice,(P_d_new,P_t_new,P_s_new,freq_maps))
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
map_d,map_t,map_s=patch_map(set_of_beta_slice,P_d_new,P_t_new,P_s_new)
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
