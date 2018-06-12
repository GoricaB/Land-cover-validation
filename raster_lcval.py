##raster_lcval=name
##classified=raster
##reference=raster
##output_folder=folder
##Print_example_of_a_priori_data=boolean False
##weights_kappa=file 
##row_membership_probability=file
##column_membership_probability=file
##classess_for_GCSI=file


import processing
import numpy as np
import os


reference = processing.getObject(reference)

ext =reference.extent()
xmin = ext.xMinimum()
xmax = ext.xMaximum()
ymin = ext.yMinimum()
ymax = ext.yMaximum()
coords = "%f,%f,%f,%f" %(xmin, xmax, ymin, ymax)


stats=processing.runalg('grass7:r.stats', [classified,reference],',','0','255',False,False,False,True,False,False,False,False,False,True,False,False,True,coords,None,None)

f = open(stats['rawoutput'],'r')

stats1 = []
for row in f.readlines():
    stats1.append(row[:-1])
    
list_comb = []
ref_cat = []
class_cat = []
for i in stats1:
    clas_v = int(i.split(",")[0])
    ref_v = int(i.split(",")[1])
    v = int(i.split(",")[2])
    list_comb.append( {"ref_v":ref_v,"clas_v":clas_v,"cells":v})
    ref_cat.append(ref_v)
    class_cat.append(clas_v)

ref_ucat = list(set(ref_cat))
class_ucat = list(set(class_cat))

all_ucat = (list(set().union(ref_ucat,class_ucat)))


# Generate confusion matrix with numpy
np_string = ''
val_ok = None
for item_c in all_ucat:
    for item_r in all_ucat:
        for item in list_comb:
            if item["ref_v"]==item_r and item["clas_v"]==item_c:
                val_ok = item["cells"]
            if val_ok==None:
                val_ok = 0
        np_string = np_string+str(val_ok)+" "
        val_ok=None
    np_string = np_string[:-1]+";"
        
a = np.matrix(np_string[:-1])

class_no=np.matrix(all_ucat)  
a1=np.hstack((class_no.T, a))
a2=np.matrix(['Class'])
class_no2=np.vstack((a2, class_no.T))
class_no3=str(class_no2.T)
class_no3=class_no3.replace('[','')
class_no3=class_no3.replace(']','')
class_no3=class_no3.replace('\' \'',',')
class_no3=class_no3.replace(']','')
class_no3=class_no3.replace('\'','')

    
# OUTPUT    
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.savetxt(os.path.join(output_folder,"error_matrix.txt"),a1,fmt='%.2f', header=class_no3,delimiter=',')
"""
f = open(os.path.join(output_folder,"error_matrix.txt"),"w")
f.write("ciao")
f.close()
"""
#Convert from integer matrix to float matrix
a_float = a.astype(float)
a_float[a_float==0]=0.00000001

#Calculate number of elements
n=len(a_float)*len(np.transpose(a_float))

#Calculate sum of elements
sum_px = a_float.sum()

#Normalize matrix
a_norm = a_float/sum_px


#Calculate marginal sums of rows and columns
sum_rows = a_norm.sum(axis=1)
        
sum_cols= a_norm.sum(axis=0)

#Calculate Overall Accuracy - P0
P0 = np.trace(a_norm)

#Calculate Producer Accuracies
PA = a_norm.diagonal()/sum_cols

#Calculate User Accuracies
UA = a_norm.diagonal()/np.transpose(sum_rows)


#Calculate Average User Producer accuracies
AUP = (UA+PA)/2

#Calculate Individual Classification Success Index
ICSI = 1-((1-UA)+(1-PA))
ICSI=(ICSI+1)/2

#Calculate Hellden's mean accuracy
MAH=2/(1/(PA)+1/(UA))

#Calculate Short's mapping accuracy
MAS = MAH / (2-MAH)

#Calculate AUA
AUA = 1.0/PA.size*UA.sum()


#Calculate APA
APA = 1.0/PA.size*PA.sum()


#Calculate DAUP
DAUP = (AUA+APA)/2

#Calculate CAU
CAU = ( P0+AUA)/2

#Calculate CAP
CAP = ( P0+APA)/2

#Calculate Conditional K
Poi = a_norm.diagonal()
Pci = np.multiply(sum_cols,np.transpose(sum_rows))
Pmaxi = sum_rows

k_cond = (Poi-Pci)/(np.transpose(Pmaxi)-Pci)
k_cond_ave = np.mean(k_cond)

#Calculating quantity disagreement and allocation disagreement

#Calculate qd - quantity disagreement for each class
qd = np.abs(sum_cols-np.transpose(sum_rows))


#Calculate ad - allocation disagreement for each class
ad = 2.0*(np.minimum(sum_cols-a_norm.diagonal(),np.transpose(sum_rows)-a_norm.diagonal()))


#Calculate QD- total quantity disagreement
QD = qd.sum()/2


#Calculate AD - total allocation disagreement
AD = ad.sum()/2

#Calculate D- total disagreement D=1-P0
D = AD+QD

#Calculating overall expected agreement
E=np.sum(np.multiply(sum_cols, np.transpose(sum_rows)))

#Calculating overall expected disagreement
R=1-E

#Calculating CSI - Classification succes index
CSI=np.mean(ICSI)

#Calculating GCSI - Group classification succes for ABCD categories
if len(classess_for_GCSI)>0:
    GCSI_classes=(np.matrix(np.loadtxt(classess_for_GCSI))).astype(int)
    GCSI1=[]
    counter=0
    for k in GCSI_classes:
        GCSI1.append(ICSI[0, k-1])
        counter+=1   
    GCSI=np.mean(GCSI1)
else:
    GCSI=CSI
    if Print_example_of_a_priori_data==True:
        GCSI_classes1=np.matrix(range(len(a)))+1
        np.savetxt(os.path.join(output_folder,"example_GCSI_classes_of_interest.csv"), GCSI_classes1, fmt='%.2f')
    


#Calculating standard kappa - K
Pc=E
K=(P0-Pc)/(1-Pc)

#Calculating kappa variance
theta1=P0
theta2=Pc
theta3=np.sum(np.multiply(a_norm.diagonal(), (sum_cols + np.transpose(sum_rows))))

theta4a=[]
for i in range(0, len(a_norm)):
    for j in range(0, len(a_norm)):
        sum_c=np.transpose(sum_cols)
        theta4a.append(np.multiply((a_norm[i,j]),((sum_c[i])+(sum_rows[j]))**2))
theta4b=(sum(theta4a))
theta4=theta4b[0,0]
pk1=theta1*(1-theta1)/((1-theta2)**2)
pk2=2*(1-theta1)*(2*theta1*theta2-theta3)/((1-theta2)**3)
pk3=((1-theta1)**2)*(theta4-4*(theta2**2))/((1-theta2)**4)
varK=(1/sum_px)*(pk1+pk2+pk3)

#Calcuating weighted kappa

#Creating weight matrix

#w_matrixa=np.matrix(np.random.randint(0,2, np.shape(a)))

if len(weights_kappa)==0:
    w_matrixa=np.matrix(np.random.randint(0,2, np.shape(a)))
    if Print_example_of_a_priori_data==True:
        np.savetxt(os.path.join(output_folder,"example_matrix_of_weights.csv"), w_matrixa, fmt='%.2f')
else:
    wmt=np.loadtxt(weights_kappa)
    w_matrixa=np.matrix(wmt)

w_matrix=w_matrixa.astype(float)

#weighted average of the weights in the ith category of the remotely sensed classifcation 
w_i_plusa=[]
for g in range(0, len(a_norm)):
    w_i_plusa.append(np.sum(np.multiply(w_matrix[g,:], sum_cols)))
    w_i_plus=np.transpose(w_i_plusa)

#weighted average of the weights in the jth category of the reference data
w_plusa_i=[]
for k in range(0, len(a_norm)):
    w_plusa_i.append(np.sum(np.multiply(w_matrix[:,k], sum_rows)))
    w_plus_i=np.transpose(w_plusa_i)

#constructing matrix for PO_w calculation
#multipling a_norm by weight matrix gives intermediate matrix 
#which is useful for calculation of overall accuracy of the weighted kappa   
    
wm=np.multiply(a_norm, w_matrix)
P0_w=np.sum(wm)

#constructing matrix for Pc_w calculation
pc_a=[]
for x in xrange(0, len(a_norm)):
    for y in range(0, len(a_norm)):
        pc_a.append(w_matrix[x,y]*(sum_rows*sum_cols)[x,y])
        pc_w=sum(pc_a) 
#calculation of weighted kappa K_w
K_w=(P0_w-pc_w)/(1-pc_w)

#calculation of variance of kappa weighted
thetaw1=P0_w
thetaw2=pc_w
theta_a=[]
for x in xrange(0, len(a_norm)):
    for y in range(0, len(a_norm)):
        elementxy=(a_norm[x,y]*((w_matrix[x,y]*(1-thetaw2)-(w_i_plus[x]+w_plus_i[y])*(1-thetaw1))**2))
        theta_a.append(elementxy)
        thetaw3=sum(theta_a)

var_Kw=((thetaw3-((thetaw1*thetaw2-2*thetaw2+thetaw1)**2))/(sum_px*((1-thetaw2)**4)))

#Calculate GT- Ground Truth Index

# it cannot be used with less than three classes --> IF condition??

# Create a new object a_float
a_float_0 = np.copy(a_float)

#Set diagonal elements to 0
np.fill_diagonal(a_float_0,0)

# Calculate new sum_rows and sum_cols
Ui0 = a_float_0.sum(axis=1)
Uj0 = a_float_0.sum(axis=0)

Ui0_T = Ui0.sum()
Uj0_T = Uj0.sum()

# Inizialize

I_Vj = Uj0/(Uj0_T-Uj0)
I_Vj_sum = I_Vj.sum()


I_UI = Ui0/(I_Vj_sum-I_Vj)
I_UI_sum = I_UI.sum()


# Loop
condit=False
n_loops=0
while condit==False:
    n_loops+=1
    Vj = Uj0/(I_UI_sum-I_UI)
    Vj_sum = Vj.sum()

    UI = Ui0/(Vj_sum-Vj)
    UI_sum = UI.sum()

    if (abs(I_Vj_sum-Vj_sum)<0.001) and (abs(I_UI_sum-UI_sum)<0.001):
        condit=True
    else:
        I_Vj = np.copy(Vj)
        I_UI = np.copy(UI)
        I_Vj_sum = np.copy(Vj_sum)
        I_UI_sum = np.copy(UI_sum)

F = np.multiply(np.matrix(Vj),np.transpose(np.matrix(UI)))
Rj = F.diagonal()/np.transpose(F.sum(axis=1))
GT = (UA-Rj)/(1-Rj)


#Calculation of the margfit

def func_for_while(my_matrix):
    predetermined_value=1
    for column in xrange(0,len(np.transpose(my_matrix))):
        sum_of_this_column=np.sum(my_matrix[:,column])
        if abs(sum_of_this_column-predetermined_value)>0.001:
            return False
    for row in xrange(0,len(my_matrix)):
        sum_of_this_row=np.sum(my_matrix[row,:])
        if abs(sum_of_this_row-predetermined_value)>0.001:
            return False      
    return True

a_marg=a_float+0.001
condit=False
n_loops=0

while condit==False:

    if func_for_while(a_marg)==True:
        condit=True
    else:
        a_marg=a_marg/a_marg.sum(axis=1)
        n_loops+=1
        if func_for_while(a_marg)==True:
            condit=True
        else:
            a_marg=a_marg/a_marg.sum(axis=0)
            n_loops+=1
#every division by row and every division by count are counted separately
P0_marg = np.trace(a_marg)/np.sum(a_marg)


#Computing tau coefficient

#assign prior probabilities to the classes

if len(row_membership_probability)==0:
    pred_value=1.0/len(a_norm)
    p_i_plus_predicted=np.ones(len(a_norm))*pred_value
    p_i_plus_predicted=np.asmatrix(p_i_plus_predicted)
    if Print_example_of_a_priori_data==True:
        np.savetxt(os.path.join(output_folder,"example_row_membership_probability.csv"), p_i_plus_predicted, fmt='%.2f')
else:
    tau_pred=np.loadtxt(row_membership_probability)
    p_i_plus_predicted=np.matrix(tau_pred)

if len(column_membership_probability)==0:
    pred_value=1.0/len(a_norm)
    p_plus_i_predicted=np.ones(len(a_norm))*pred_value
    p_plus_i_predicted=np.asmatrix(p_i_plus_predicted)
    if Print_example_of_a_priori_data==True:
        np.savetxt(os.path.join(output_folder,"example_column_membership_probability.csv"), p_plus_i_predicted, fmt='%.2f')
else:
    tau_pred=np.loadtxt(column_membership_probability)
    p_plus_i_predicted=np.matrix(tau_pred)

pred_value=1.0/len(a_norm)
p_i_plus_predicted=np.ones(len(a_norm))*pred_value
p_i_plus_predicted=np.asmatrix(p_i_plus_predicted)
p_plus_i_predicted=p_i_plus_predicted
#theta_tau_1 and theta_tau_2 are the same as theta_1 and theta_2 for standard kappa
theta_tau_1=theta1
theta_tau_2=np.sum(sum_cols*pred_value)

#computing theta_tau 3
v=[]
for x in range(0, len(a_norm)):
    elementxy=(a_norm[x,x]*(sum_cols[0,x]+p_i_plus_predicted[0,x]))
    v.append(elementxy)

theta_tau_3=sum(v)
#computing theta_tau_4
theta_tau_4a=np.zeros([len(a_norm),len(a_norm)])
for x in range(0, len(a_norm)):
    for y in range(0, len(a_norm)):
        theta_tau_4a[x,y]=a_norm[x,y]*(sum_cols[0,y]+p_plus_i_predicted[0,y])**2
theta_tau_4=np.sum(theta_tau_4a)
# Computing tau coefficient
tau=(theta_tau_1-theta_tau_2)/(1-theta_tau_2)
#computing variance of tau
p1=theta_tau_1*(1-theta_tau_1)/((1-theta_tau_2)**2)
p2=2*(1-theta_tau_1)*(2*theta_tau_1*theta_tau_2-theta_tau_3)/((1-theta_tau_2)**3)
p3=((1-theta_tau_1)**2)*(theta_tau_4-4*(theta_tau_2**2))/((1-theta_tau_2)**4)
var_tau=(1/sum_px)*(p1+p2+p3)


#Computing Change of entropy of ground truth map

#conditional probability - the probability of a pixel belonging to class j on map GT when the pixel is of class i on map M
p_cond_gt_mi=a_norm/sum_rows

#replacing zero values with 1, because log of zero is not possible to compute
#in this way, values that are impossible to compute are set to zero
p_cond_gt_mi=(np.where(p_cond_gt_mi == 0.0, 1.0, p_cond_gt_mi))
p_cond_gt_mi=np.asmatrix(p_cond_gt_mi)

#value of entropy for map GT knowing that for the corresponding localization on map M the class is Mi
H_gt_mi=-sum(np.transpose(np.multiply(p_cond_gt_mi, np.log2(p_cond_gt_mi))))

#then the entropy of gt map is 
H_gt=-np.sum(np.multiply(sum_cols, np.log2(sum_cols)))

#Calculating entropy change
EC_gt=((H_gt-H_gt_mi)/H_gt)


#Computing Change of entropy of classified map

#conditional probability - the probability of a pixel belonging to class i on map M when the pixel is of class j on map GT
p_cond_m_gti=a_norm/sum_cols

#replacing zero values with one, because log of zero is not possible to compute
#in this way, values that are impossible to compute are set to zero
p_cond_m_gti=(np.where(p_cond_m_gti == 0.0, 1.0, p_cond_m_gti))
p_cond_m_gti=np.asmatrix(p_cond_m_gti)

#value of entropy for map M knowing that for the corresponding localization on map GT the class is GTi
H_m_gti=-sum((np.multiply(p_cond_m_gti, np.log2(p_cond_m_gti))))

#then the entropy of M map is 
H_m=-np.sum(np.multiply(sum_rows, np.log2(sum_rows)))

#Calculating entropy change
EC_m=((H_m-H_m_gti)/H_m)


#Computing AMI - average mutual information
AMI=np.sum(np.multiply(a_norm, (np.log2(p_cond_m_gti/sum_rows))))
AMI_percent=AMI/H_gt

#Computing NMI - normalized mutual information
NMI_gt_map=AMI/H_gt
NMI_m_map=AMI/H_m

#the arithmetic mean of the entropies on classified map and on reference data
NMI_alpha=2*AMI/(H_gt+H_m)

#the geometric mean of the entropies on classified map and on reference data
NMI_g=AMI/(np.sqrt(H_gt*H_m))

#the arithmetic mean of the maximum entropies on on classified map and on reference data
NMI_m=2*AMI/(np.max(H_gt_mi)+np.max(H_m_gti))

#Computing Aickin alpha

#setting inital values
pi_m_aickin=sum_rows
pi_gt_aickin=sum_cols
pa_aickin=P0

pe_aickin=np.multiply(pi_m_aickin, np.transpose(pi_gt_aickin))
pe_aickin=sum(pe_aickin)
#computing initial Aickin's alpha
alpha=(pa_aickin-pe_aickin)/(1-pe_aickin)

#Creating matrices needed for iterations
alpha_1=[]
alpha_1.append(alpha)

pi_m_aickin_list=[]
pi_m_aickin_list.append(np.transpose(pi_m_aickin))

condition=False
n_loops=0
#iterations until two consequetive aickin's alpha has difference below 0.001
while condition==False:
    n_loops+=1
    pi_m_aickin1=np.transpose(sum_rows)/((1-alpha_1[n_loops-1])+np.multiply(alpha_1[n_loops-1],(pi_gt_aickin/pe_aickin)))
    pi_m_aickin_list.append(pi_m_aickin1)
    
    pi_gt_aickin=(sum_cols)/((1-alpha_1[n_loops-1])+np.multiply(alpha_1[n_loops-1],(pi_m_aickin_list[n_loops-1]/pe_aickin)))
    
    pe_aickin=np.sum(np.multiply(pi_m_aickin_list[n_loops], (pi_gt_aickin)))
    
    alpha=(pa_aickin-pe_aickin)/(1-pe_aickin)
    alpha_1.append(alpha)

    if abs(alpha_1[n_loops-1]-alpha_1[n_loops])>0.001:
        condition=False
    else:
        condition=True
#Value of the Aickin's alpha after the condition is satisfied
Aickin_alpha=alpha_1[n_loops]


per_class=(np.vstack((PA, UA, AUP, ICSI, MAH, MAS, GT, k_cond, qd, ad))).T



per_class2=np.hstack((class_no.T, per_class))


global_indexes=np.matrix((P0, AUA, APA, CSI, GCSI, P0_marg, K, K_w, tau, AMI_percent, NMI_g, NMI_m, CAU, CAP, D, AD, QD, Aickin_alpha, k_cond_ave))

np.savetxt(os.path.join(output_folder,"global_indexes.csv"), (global_indexes), newline='\n', fmt="%10.2f", delimiter=',', header="'Note:If classes of interest for GCSI are not selected so will be equal to CSI \n\nP0,AUA,APA,CSI,GCSI,P0_marg,K,K_w,tau,AMI,NMI_g,NMI_m,CAU,CAP,D,AD,QD,Aickin_alpha,k_cond_ave", footer="\n\n\nLegend: \n P0,Overall accuracy \n AUA,Average of user\'s accuracies \n APA,Average of producer\'s accuracies \n CSI,Classification success index \n GCSICSI,Group classification success index \n P0_marg,Overall accuracy after margfit \n K,Standard (Cohen\'s) kappa \n K_w,Weighted kappa \n tau,tau index \n AMI,Average mutual information \n NMI_g,The geometric mean of the entropies on classified map and on reference data \n NMI_m,the arithmetic mean of the maximum entropies on on classified map and on reference data \n CAU,Combined user\'s accuracy \n CAP,Combined producer\'s accuracy \n D,Total disagreement \n AD,Total allocation disagreement \n QD,Total quantity disagreement \n Aickin_alpha,Aickin\'s alpha \n k_cond_aver,Average of conditional kappa")

np.savetxt(os.path.join(output_folder,"per_class_indexes.csv"), per_class2, newline='\n', fmt="%10.2f", delimiter=',', header="Class,PA,UA,AUP,ICSI,MAH,MAS,GT,k_cond,qd,ad", footer="\n\n\nLegend: \n PA,Producer's accuracy \n UA,User's accuracy \n AUP,Average user's and producer's accuracy \n ICSI,Individual Classification Success Index \n MAH,Hellden's mean accuracy \n MAS,Short's mapping accuracy \n GT,Ground Truth Index \n k_cond,Conditional kappa \n qd,Quantity disagreement for each class \n ad,Allocation disagreement for each class")