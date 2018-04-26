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




def data_patch(freq_maps,meta_P): #this function returns the data corresponding to the pixels in the input patch. Works for patches as well as for meta patches
    data=[]
    temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    for i in range(len(meta_P)):
        for f in range(len(freq_maps)):
            for s in range(len(freq_maps[0])):
                for p in range(len(meta_P[i])):
                    temp_list[f][s].append(freq_maps[f][s][meta_P[0][0]])
        data.append(temp_list)# for p in range(len(meta_P[i])))
        temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    del temp_list
    return data




def joint_spectral_likelihood(input_set_of_betas,P_d,P_t,P_s,data): #computes the joint likelihood for the whole sky taking into account the patches
    #prewhiten_factors must be defined above !
    logL_spec=0
    #first we have to create the maps of beta in the sky to link each patch with its corresponding beta value
    ind=0
    pixel_number=hp.nside2npix(nside)
    map_bd=[None]*pixel_number
    map_bt=[None]*pixel_number
    map_bs=[None]*pixel_number

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
        ind+=1

    for p in range(hp.nside2npix(nside)):
        logL_spec+=fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    del map_bd,map_bt,map_bs
    return logL_spec




def sigma_matrix_making(minimization_result,P_d,P_t,P_s,data): #computes the sigma matrix for each patch and creates the total sigma matrix for the whole sky
    ind=0
    pixel_number=hp.nside2npix(nside)
    map_bd=[None]*pixel_number
    map_bt=[None]*pixel_number
    map_bs=[None]*pixel_number
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
    for p in range(pixel_number):
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
    pixel_number=hp.nside2npix(nside)
    beta_d_list=[]
    beta_t_list=[]
    map_bs=[None]*pixel_number

    delta_beta=[]

    for i in range(len(P_d)):
        beta_d_list=[]
        for j in range(len(P_d[i])):
            beta_d_list.append(input_beta_zeroth_iteration[P_d[i][j]])
            # print(beta_d_list)
        delta_beta.append(np.std(beta_d_list))
        # print(np.std(beta_d_list))
    # print(delta_beta)
    for i in range(len(P_t)):
        beta_t_list=[]
        for j in range(len(P_t[i])):
            beta_t_list.append(input_beta_zeroth_iteration[pixel_number+P_t[i][j]])
            # print(beta_t_list)
        delta_beta.append(np.std(beta_t_list))
        # print(np.std(beta_t_list))
    # print(delta_beta)
    for i in range(len(P_s)):
        beta_s_list=[]
        for j in range(len(P_s[i])):
            beta_s_list.append(input_beta_zeroth_iteration[2*pixel_number+P_s[i][j]])
        delta_beta.append(np.std(beta_s_list))
    # print(delta_beta)
    return(np.diag(delta_beta))




def matrix_comparison(delta_beta,sigma_matrix): #compares the delta matrix with the sigma matrix
    delta_beta2=np.dot(delta_beta,delta_beta)
    norm=np.linalg.norm(np.absolute(delta_beta2-sigma_matrix)) #%! is the determinant the right kind of norm to choose ?
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
            if input_beta_zeroth_iteration[i]>inf_boundary_d and input_beta_zeroth_iteration[i]<sup_boundary_d:
                beta_dust_slice[i]=(inf_boundary_d+sup_boundary_d)/2
                temp_list2.append(i)
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
            temp_list2.append(i)
    if len(temp_list2) >=1:
        P_d_new.append(temp_list2)
        min_beta_d_in_list=np.amin(np.take(input_beta_zeroth_iteration,temp_list2))
        max_beta_d_in_list=np.amax(np.take(input_beta_zeroth_iteration,temp_list2))
        set_of_beta_slice.append((min_beta_d_in_list+max_beta_d_in_list)/2)

        #-------------------------------------------------P_t CONSTRUCTION----------------------------------------------------

    temp_list2=[]
    inf_boundary_t=min_beta_t
    sup_boundary_t=min_beta_t+np.sqrt(sigma[len(P_d)][len(P_d)])
    for j in range(len(P_t)-1):
        for i in range(pixel_number):
            if input_beta_zeroth_iteration[i+pixel_number]>inf_boundary_t and input_beta_zeroth_iteration[i+pixel_number]<sup_boundary_t:
                temp_dust_slice[i]=(inf_boundary_t+sup_boundary_t)/2
                temp_list2.append(i)
        if len(temp_list2) >= 1:
            P_t_new.append(temp_list2)
            set_of_beta_slice.append((inf_boundary_t+sup_boundary_t)/2)
        temp_list2=[]
        inf_boundary_t=inf_boundary_t+np.sqrt(sigma[j+len(P_d)][j+len(P_d)])
        sup_boundary_t=sup_boundary_t+np.sqrt(sigma[j+1+len(P_d)][j+1+len(P_d)])

    temp_list2=[]
    for i in range(pixel_number):
        if np.isnan(temp_dust_slice[i]):
            temp_list2.append(i)
    if len(temp_list2) >=1:
        for j in range(len(temp_list2)):
            temp_list2[j]=temp_list2[j]+pixel_number
        min_beta_t_in_list=np.amin(np.take(input_beta_zeroth_iteration,temp_list2))
        max_beta_t_in_list=np.amax(np.take(input_beta_zeroth_iteration,temp_list2))
        set_of_beta_slice.append((min_beta_t_in_list+max_beta_t_in_list)/2)
        P_t_new.append(temp_list2)


        #-------------------------------------------------P_s CONSTRUCTION----------------------------------------------------
    temp_list2=[]
    inf_boundary_s=min_beta_s
    sup_boundary_s=min_beta_s+np.sqrt(sigma[len(P_d)+len(P_t)][len(P_d)+len(P_t)])
    for j in range(len(P_s)-1):
        for i in range(pixel_number):
            if input_beta_zeroth_iteration[i+2*pixel_number]>inf_boundary_s and input_beta_zeroth_iteration[i+2*pixel_number]<sup_boundary_s:
                beta_sync_slice[i]=(inf_boundary_s+sup_boundary_s)/2
                temp_list2.append(i)
        if len(temp_list2) >= 1:
            P_s_new.append(temp_list2)
            set_of_beta_slice.append((inf_boundary_s+sup_boundary_s)/2)
        temp_list2=[]
        inf_boundary_s=inf_boundary_s+np.sqrt(sigma[j+len(P_d)+len(P_t)][j+len(P_d)+len(P_t)])
        sup_boundary_s=sup_boundary_s+np.sqrt(sigma[j+1+len(P_d)+len(P_t)][j+1+len(P_d)+len(P_t)])

    temp_list2=[]
    for i in range(pixel_number):
        if np.isnan(temp_dust_slice[i]):
            temp_list2.append(i)
    if len(temp_list2) >=1:
        for j in range(len(temp_list2)):
            temp_list2[j]=temp_list2[j]+2*pixel_number
        min_beta_s_in_list=np.amin(np.take(input_beta_zeroth_iteration,temp_list2))
        max_beta_s_in_list=np.amax(np.take(input_beta_zeroth_iteration,temp_list2))
        set_of_beta_slice.append((min_beta_s_in_list+max_beta_s_in_list)/2)
        P_t_new.append(temp_list2)
    temp_list2=[]

    return P_d_new,P_t_new,P_s_new,set_of_beta_slice

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
for i in range(len(freq_maps[0][1])): P_d.append([i])
for i in range(len(freq_maps[0][1])): P_s.append([i])
for i in range(len(freq_maps[0][1])): P_t.append([i])

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

for l in range(len(P_d)):
     # for p in range(len(P_d[l])):
             # beta_pix=[map_bd[p],map_bs[p],map_t[p]]
     #print beta_pix
     prewhitened_data = prewhiten_factors * np.transpose(data[l])
     #print('prewhitened_data.shape=',prewhitened_data.shape)
     funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
     fun.append(funtemp)
     last_values.append(last_valuestemp)
     # print(funtemp([1.59,19.6,-3.1]))
     del funtemp, jactemp,last_valuestemp



# for p in range(hp.nside2npix(nside)):
#     funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
#     fun.append(funtemp)
#     last_values.append(last_valuestemp)
#     print(funtemp([1.59,19.6,-3.1]))

minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,input_set_of_betas,(P_d,P_t,P_s,freq_maps))
print(minimization_result)
print('newbeta=',minimization_result.x)
input_beta_zeroth_iteration=minimization_result.x
sigma=sigma_matrix_making(minimization_result,P_d,P_t,P_s,freq_maps)
print('sigma=')
for i in range(len(sigma)):
    print(np.sqrt(sigma[i][i]))
# print('shape sigma=',np.shape(sigma))
print(np.shape(input_beta_zeroth_iteration))
a=delta_beta_matrix_making(P_d,P_t,P_s,input_beta_zeroth_iteration)
# print('delta=',a)
print('comp=',matrix_comparison(a,sigma))
print('input_beta_zeroth_iteration=',input_beta_zeroth_iteration)
P_d_new,P_t_new,P_s_new,set_of_beta_slice=patch_making(input_beta_zeroth_iteration,sigma,P_d,P_t,P_s)
print('P_d_new=',P_d_new)
print('P_t_new=',P_t_new)
print('P_s_new=',P_s_new)
print('set_of_beta_slice=',set_of_beta_slice)


hp.mollview(input_beta_zeroth_iteration[:48])
plt.title('beta dust map')
plt.show()


hp.mollview(input_beta_zeroth_iteration[48:96])
plt.title('dust temp map')
plt.show()

hp.mollview(input_beta_zeroth_iteration[96:144])
plt.title('beta sync map')
plt.show()
