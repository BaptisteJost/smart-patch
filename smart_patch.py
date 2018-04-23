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



# P_d=[range(24),range(24,48)]
# P_t=[range(48)]
# P_s=[range(12),range(12,24),range(24,36),range(36,48)]
# for i in range(len(P_s)):
#     try:
#         P_s[i].index(45)
#         bd_index=i
#         print('bd_index=',bd_index)
#     except:
#         pass
# print bd_index
#
# input_set_of_betas=[1.59,1.59,19.6,-3.1,-3.1,-3.1,-3.1]

#-----------------------------------------The A matrix and the likelihood function are computed once and for all in the program------------
prewhitened_data = prewhiten_factors * np.transpose(freq_maps)
A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=prewhiten_factors)
A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)
fun=[]
last_values=[]
for p in range(hp.nside2npix(nside)):
    funtemp, jactemp, last_valuestemp = fgal._build_bound_inv_logL_and_logL_dB(A_ev, prewhitened_data,np.diag(prewhiten_factors) , A_dB_ev, comp_of_param)
    fun.append(funtemp)
    last_values.append(last_valuestemp)
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
    sigma_tot=[[0 for j in range(len(minimization_result.x))]for i in range(len(minimization_result.x))]
    # print(sigma_tot)
    #print(sigma_tot[0][0])
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
        u_e_v_last, A_dB_last, x_last, pw_d = last_values[p]
        s =fgal._Wd_svd(u_e_v_last[0], pw_d[0])

        if A_dB_ev is None:
            fisher = numdifftools.Hessian(fun)(minimization_result.x)
        else:
            fisher = fgal._fisher_logL_dB_dB_svd(u_e_v_last[0], s,
                                            A_dB_last[0], comp_of_dB)
        Sigma = np.linalg.inv(fisher)
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
        # print('sigma_tot=',sigma_tot)
        # print('Sigma[0][0]=',Sigma[0][0])
        sigma_tot[bd_index][bd_index]+=Sigma[0][0]
        sigma_tot[len(P_d)+bt_index][len(P_d)+bt_index]+=Sigma[1][1]
        sigma_tot[len(P_d)+len(P_t)+bs_index][len(P_d)+len(P_t)+bs_index]+=Sigma[2][2]

        sigma_tot[bd_index][len(P_d)+bt_index]+=Sigma[0][1]
        sigma_tot[len(P_d)+bt_index][bd_index]+=Sigma[1][0]

        sigma_tot[bd_index][len(P_d)+len(P_t)+bs_index]+=Sigma[0][2]
        sigma_tot[len(P_d)+len(P_t)+bs_index][bd_index]+=Sigma[2][0]

        sigma_tot[len(P_d)+bt_index][len(P_d)+len(P_t)+bs_index]+=Sigma[1][2]
        sigma_tot[len(P_d)+len(P_t)+bs_index][len(P_d)+bt_index]+=Sigma[2][1]

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

def patch_making(input_set_of_betas,input_beta_zeroth_iteration,freq_maps,sigma,deltab):

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

minimization_result=scipy.optimize.minimize(joint_spectral_likelihood,input_set_of_betas,(P_d,P_t,P_s,freq_maps))
print(minimization_result)
print('newbeta=',minimization_result.x)
input_beta_zeroth_iteration=minimization_result.x
sigma=sigma_matrix_making(minimization_result,P_d,P_t,P_s,freq_maps)
print('sigma=')
for i in range(len(sigma)):
    print(sigma[i])
print('shape sigma=',np.shape(sigma))
a=delta_beta_matrix_making(P_d,P_t,P_s,input_beta_zeroth_iteration)
print('delta=',a)
print('comp=',matrix_comparison(a,sigma))

hp.mollview(input_beta_zeroth_iteration[:48])
plt.title('beta dust map')
plt.show()


hp.mollview(input_beta_zeroth_iteration[48:96])
plt.title('dust temp map')
plt.show()

hp.mollview(input_beta_zeroth_iteration[96:144])
plt.title('beta sync map')
plt.show()
