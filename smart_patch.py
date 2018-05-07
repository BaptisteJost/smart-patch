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
nside = 2
pixel_number=hp.nside2npix(nside)
sky = get_sky(nside, 'c1d1s1')
instrument = get_instrument(nside, 'litebird')
freq_maps = instrument.observe(sky, write_outputs=False)[0]
freq_maps=freq_maps[:,1:,:]  #on retire la temperature (1:) car on ne prend en compte que Q&U pas I
components = [CMB(), Dust(150.), Synchrotron(20.)]
prewhiten_factors = _get_prewhiten_factors(instrument, freq_maps.shape)         # correspond a N^-1/2
# print(freq_maps)


mask=hp.read_map('../map_test/HFI_Mask_GalPlane-apo2_2048_R2.00.fits',field=2)
mask_bin=mask*0
mask_bin=hp.ud_grade(mask_bin,nside)
mask_bin[np.where(hp.ud_grade(mask,nside)!=0)[0]]=1
masked_pixels=[np.where(hp.ud_grade(mask,nside)==0)[0]]
freq_maps_save=freq_maps[:][:][:]
freq_maps= np.empty((len(freq_maps_save), len(freq_maps_save[0]),pixel_number-len(masked_pixels[0])))
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
print('pixel_number=',pixel_number)
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
    for i in range(len(P_i)):
        for f in range(len(freq_maps)):
            for s in range(len(freq_maps[0])):
                for p in range(len(P_i[i])):
                    temp_list[f][s].append(freq_maps[f][s][P_i[0][0]])
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
    print 'P_d=',P_d
    for i in range(len(P_d)):
        for j in range(len(P_d[i])):
            print 'P_d[%s][%s]='%(i,j),P_d[i][j]
            print 'j=',j
            map_bd[P_d[i][j]]=input_set_of_betas[ind]
        ind+=1
    print(pixel_list)
    print map_bd
    for i in range(len(P_t)):
        for j in range(len(P_t[i])):
            map_bt[P_t[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in range(len(P_s[i])):
            map_bs[P_s[i][j]]=input_set_of_betas[ind]
        ind+=1
    return map_bd,map_bt,map_bs


def joint_spectral_likelihood(input_set_of_betas,P_d,P_t,P_s,data): #computes the joint likelihood for the whole sky taking into account the patches
    #prewhiten_factors must be defined above !
    logL_spec=0
    #first we have to create the maps of beta in the sky to link each patch with its corresponding beta value
    ind=0
    pixel_number1=hp.nside2npix(nside)
    map_bd=[0]*pixel_number1
    map_bt=[0]*pixel_number1
    map_bs=[0]*pixel_number1
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
        for j in range(len(P_t[i])):
            map_bt[P_t[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in range(len(P_s[i])):
            map_bs[P_s[i][j]]=input_set_of_betas[ind]
        ind+=1

    for p in pixel_list:
        logL_spec+=fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    del map_bd,map_bt,map_bs
    return logL_spec


def sigma_matrix_making(minimization_result,P_d,P_t,P_s,data): #computes the sigma matrix for each patch and creates the total sigma matrix for the whole sky
    ind=0
    pixel_number1=hp.nside2npix(nside)
    map_bd=[None]*pixel_number1
    map_bt=[None]*pixel_number1
    map_bs=[None]*pixel_number1
    fisher_tot=[[0 for j in range(len(minimization_result.x))]for i in range(len(minimization_result.x))]
    # print(fisher_tot)
    #print(fisher_tot[0][0])
    for i in range(len(P_d)):
        for j in range(len(P_d[i])):
            map_bd[P_d[i][j]]=minimization_result.x[ind]
        ind+=1
    for i in range(len(P_t)):
        for j in range(len(P_t[i])):
            map_bt[P_t[i][j]]=minimization_result.x[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in range(len(P_s[i])):
            map_bs[P_s[i][j]]=minimization_result.x[ind]
        ind+=1

    sigma_m=[]
    for p in pixel_list:
        fun[p]([map_bd[p],map_bt[p],map_bs[p]])#[minimization_result.x[l*3],minimization_result.x[l*3+1],minimization_result.x[l*3+2]]
        # u_e_v_last, A_dB_last, x_last, pw_d = last_values[p]
        # s =fgal._Wd_svd(u_e_v_last[0], pw_d[0])
        # print(s)
        # if A_dB_ev is None:
        #     fisher = numdifftools.Hessian(fun)(minimization_result.x)
        # else:
        #     fisher = fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s,
        #                                     A_dB_last[0], comp_of_dB)

            # print(fisher)
        # print([map_bd[p],map_bt[p],map_bs[p]])
        # print('u_e_v_last[0]=',u_e_v_last[0],'s=', s,'A_dB_last[0]=',A_dB_last[0],'comp_of_dB=', comp_of_dB)
        # print('numdifftools.Hessian(fun)(minimization_result.x)=',numdifftools.Hessian(fun[p])([map_bd[p],map_bt[p],map_bs[p]]))
        #
        # print('fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s, A_dB_last[0], comp_of_dB)=',fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s,A_dB_last[0], comp_of_dB))

        fisher=numdifftools.Hessian(fun[p])([map_bd[p],map_bt[p],map_bs[p]])
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
    sigma_tot=np.linalg.inv(fisher_tot).T
    return(sigma_tot)


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
        temp_list3=[0]*len(temp_list2)
        index_list=[]
        for j in range(len(temp_list2)):
            index_list.append(pixel_list.index(temp_list2[j]))
            temp_list3[j]=index_list[j]+pixel_number
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
        temp_list3=[0]*len(temp_list2)
        index_list=[]
        for j in range(len(temp_list2)):
            index_list.append(pixel_list.index(temp_list2[j]))
            temp_list3[j]=index_list[j]+pixel_number
        min_beta_s_in_list=np.amin(np.take(input_beta_zeroth_iteration,temp_list3))
        max_beta_s_in_list=np.amax(np.take(input_beta_zeroth_iteration,temp_list3))
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
            if input_beta_zeroth_iteration[i]>minimization_result.x[j]-sigma[j][j]/2 and input_beta_zeroth_iteration[i]<minimization_result.x[j]+sigma[j][j]/2:
                beta_dust_slice[i]=minimization_result.x[j]
                temp_list2.append(i)
        if len(temp_list2) >= 1:
            P_d_new.append(temp_list2)
            set_of_beta_slice.append(minimization_result.x[j])
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
for i in range(len(freq_maps[0][1])): P_d.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_s.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_t.append([pixel_list[i]])
print('P_d first=',P_d)
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



fun=[]
last_values=[]
data=data_patch(freq_maps,P_d)
A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=prewhiten_factors)
#A_local=A_ev(beta_pix)
A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)

# for i in pixel_list:
print('pixel_number=',pixel_number)
print('data shape=',np.shape(data[pixel_number-1]))
# print('data shape=',np.shape(data[pixel_number]))
# print i
fun=[None]*hp.nside2npix(nside)
last_values=[None]*hp.nside2npix(nside)
for l in range(pixel_number):
     prewhitened_data = prewhiten_factors * np.transpose(data[l])
     funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
     fun[pixel_list[l]]=funtemp
     last_values[pixel_list[l]]=last_valuestemp
     del funtemp, jactemp,last_valuestemp


# for p in range(hp.nside2npix(nside)):
#     funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
#     fun.append(funtemp)
#     last_values.append(last_valuestemp)
#     print(funtemp([1.59,19.6,-3.1]))
comp=[]
minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,input_set_of_betas,(P_d,P_t,P_s,freq_maps))
# print(minimization_result)
# print('newbeta=',minimization_result.x)
input_beta_zeroth_iteration=minimization_result.x
print(input_beta_zeroth_iteration)
sigma=sigma_matrix_making(minimization_result,P_d,P_t,P_s,freq_maps)
print'sigma_d_iter0='
for j in range(len(P_t)):
    print np.sqrt(sigma[j+len(P_d)][j+len(P_d)])
# print('sigma=')
# for i in range(len(sigma)):
#     print(np.sqrt(sigma[i][i]))
# print('shape sigma=',np.shape(sigma))
# print(np.shape(input_beta_zeroth_iteration))
a=delta_beta_matrix_making(P_d,P_t,P_s,input_beta_zeroth_iteration)
# print('delta=',a)
comp.append(matrix_comparison(a,sigma))
# print('matrix comparison=',comp[0])
# print('input_beta_zeroth_iteration=',input_beta_zeroth_iteration)
print('P_d bla=',P_d)
P_d_new,P_t_new,P_s_new,set_of_beta_slice=patch_making(input_beta_zeroth_iteration,sigma,P_d,P_t,P_s)
print('P_d_new=',P_d_new)
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

i=0
# while comp[i]>0.00021:
for i in range(10):
    map_b1,map_t1,map_s1=patch_map(set_of_beta_slice,P_d_new,P_t_new,P_s_new)
    map_b=[]
    map_t=[]
    map_s=[]
    map_b=map_b1[:]
    map_t=map_t1[:]
    map_s=map_s1[:]
    map_b1=[]
    map_t1=[]
    map_s1=[]
    print'sigma'
    for j in range(len(P_d_new)+len(P_t_new)+len(P_s_new)):
        print np.sqrt(sigma[j][j])

    binsd=np.linspace(np.amin(input_beta_zeroth_iteration[:pixel_number]),np.amax(input_beta_zeroth_iteration[:pixel_number]),pixel_number)
    for j in range(len(P_d_new)):
        templist5=[x for x in range(len(P_d_new[j]))]
        # print 'templist5=',templist5
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in input_beta_zeroth_iteration[templist5]],binsd)
        templist5=[]
    # plt.savefig('Bd_histo_iteration=%s'%i)
    plt.title('B_d histogram iteration=%s'%i)
    plt.show()
    binst=np.linspace(np.amin(input_beta_zeroth_iteration[pixel_number:2*pixel_number]),np.amax(input_beta_zeroth_iteration[pixel_number:2*pixel_number]),pixel_number)
    for j in range(len(P_t_new)):
        templist5=[x+pixel_number for x in range(len(P_t_new[j]))]
        # print 'templist5=',templist5
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in input_beta_zeroth_iteration[templist5]],binst)
        templist5=[]
    # plt.savefig('Bt_histo_iteration=%s'%i)
    plt.title('B_t histogram iteration=%s'%i)
    plt.show()

    binss=np.linspace(np.amin(input_beta_zeroth_iteration[2*pixel_number:3*pixel_number]),np.amax(input_beta_zeroth_iteration[2*pixel_number:3*pixel_number]),pixel_number)
    for j in range(len(P_s_new)):
        templist5=[x+2*pixel_number for x in range(len(P_s_new[j]))]
        # print 'templist5=',templist5
        # print [x for x in input_beta_zeroth_iteration[templist5]]
        plt.hist([x for x in input_beta_zeroth_iteration[templist5]],binss)
        templist5=[]
    # plt.savefig('Bs_histo_iteration=%s'%i)
    plt.title('B_s histogram iteration=%s'%i)
    plt.show()


    hp.mollview(np.array(map_b),sub=(2,2,1))
    plt.title('beta dust map iteration %s' %(i))
    hp.mollview(np.array(map_t),sub=(2,2,2))
    plt.title('dust temp map iteration %s' %(i))
    hp.mollview(np.array(map_s),sub=(2,2,3))
    plt.title('beta sync map iteration %s' %(i))
    plt.subplot(2,2,4)
    plt.plot(comp)
    plt.title('Evolution of the norm over iterations')
    # plt.savefig('iteration=%s'%i)
    plt.show()
    # plt.close()
    sigma_list.append(np.linalg.norm(sigma))

    minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,set_of_beta_slice,(P_d_new,P_t_new,P_s_new,freq_maps))

    sigma=sigma_matrix_making(minimization_result,P_d_new,P_t_new,P_s_new,freq_maps)
    # for j in range(len(sigma)):
    #     print(np.sqrt(sigma[j][j]))
    delta_b=delta_beta_matrix_making(P_d_new,P_t_new,P_s_new,input_beta_zeroth_iteration)
    # for j in range(len(delta_b)):
    #     print(np.sqrt(delta_b[j][j]))
    comp.append(matrix_comparison(delta_b,sigma))
    print('comp%s='%i,comp[i])
    # print('set_of_beta_slice1=',set_of_beta_slice)
    set_of_beta_slice=[]
    P_d_new1,P_t_new1,P_s_new1,set_of_beta_slice1=patch_making(input_beta_zeroth_iteration,sigma,P_d_new,P_t_new,P_s_new)
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
map_b,map_t,map_s=patch_map(set_of_beta_slice,P_d_new,P_t_new,P_s_new)
hp.mollview(np.array(map_b),sub=(2,2,1))
plt.title('beta dust map iteration %s' %(i))
hp.mollview(np.array(map_t),sub=(2,2,2))
plt.title('dust temp map iteration %s' %(i))
hp.mollview(np.array(map_s),sub=(2,2,3))
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
