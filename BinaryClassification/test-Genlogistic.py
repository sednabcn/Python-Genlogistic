#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:41:04 2018

@author: sedna
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
from statsmodels.multivariate.pca import PCA as smPCA
from metrics_classifier import MetricClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import matplotlib.pyplot as plt
from tools import Tools
from matplotlib import colors as mcolors
# colors for grahics with matplolib and plotly

testsize=0.4
treshold=0.90
NDataSets=3

Datasets=['Pima','Sonar','Ionosphere']

ac=MetricClassifier()

def show_best_pca_predictors(x,ncomp,treshold):
            try:
                eigenvalues=smPCA(x,ncomp).eigenvals 
            except:
                eigenvalues=smPCA(x,ncomp,method='eig').eigenvals
              
            order=[ii for ii,vals in sorted(enumerate(np.abs(eigenvalues)), \
                                         key=lambda x:x[1],reverse=True)]
            eigenvalues_sorted=[eigenvalues[order[ii]] for ii in range(ncomp)]
            features_sorted=[x.columns[order[ii]] for ii in range(ncomp)]
            
            tot = sum(eigenvalues_sorted)
            best_predictors=[]
            cum_var_exp=0.0
            for ii,i in enumerate(eigenvalues_sorted[:]):
                var_exp=(i/tot)
                cum_var_exp+=var_exp
                if cum_var_exp < treshold:
                    best_predictors.append(features_sorted[ii])
            return best_predictors
        
def plot_matrix_matrix(X,Y,Title,xlabel,ylabel,Labels,\
                           Linestyle,kind,scale,grid,text,boxstyle,mode):
        """Draw matrix vs matrix."""
        """
        kind of graphic
        0-plot
        1-scatter
        2-stem
        scale of plot
        'equal'
        'Log'
        'semilogy'
        'semilogx'
        'Loglog'
        mode of graphic
        0-full
        1-split
        """
        
        if (len(Y.columns)>7 and mode==0):
            colors = [ic for ic in mcolors.CSS4_COLORS.values() if ic !=(1,1,1)]
            keys   = [kc  for kc in mcolors.CSS4_COLORS.keys() \
            if mcolors.CSS4_COLORS[kc]!=(1,1,1)]        
            linewidth=2
        else:
            colors = [ic for ic in mcolors.BASE_COLORS.values() if ic !=(1,1,1)]
            keys   = [kc  for kc in mcolors.BASE_COLORS.keys() \
            if mcolors.BASE_COLORS[kc]!=(1,1,1)]
            linewidth=2
            
        """
        Checking orders of matrices
        """ 
        if (X.shape[0]!=Y.shape[0] or X.shape[1]!=Y.shape[1]):
                    print("These matrices are not equal order")
        
        
        idx=[X.iloc[:,i].argsort() for i in range(X.shape[1])]
        plt.figure()
        ax=plt.subplot(1,1,1)
        for i,linestyle in enumerate(Linestyle):
            if i in range(X.shape[1]):
               label=Labels[i]
               if isinstance(label,float):
                   label=round(label,2)
                   
               if kind==0 :
                   if scale=='equal' or 'Log':
                       ax.plot(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=linewidth,label=label,color=colors[i])
                   elif scale=='semilogy':
                       ax.semilogy(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=linewidth,label=label,color=colors[i])
                   elif scale =='semilogx':
                       ax.semilogx(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=linewidth,label=label,color=colors[i])
                   elif scale =='Loglog':
                       ax.loglog(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=linewidth,label=label,color=colors[i])
                   else:
                       pass
               elif kind==1:
                   ax.scatter(X.iloc[idx[i],i],Y.iloc[idx[i],i],marker=linestyle,label=label,color=colors[i])
               elif kind==2:
                   markerline, stemlines, baseline = ax.stem(X.iloc[idx[i],i],Y.iloc[idx[i],i],'-.',\
                                                             markerfmt=keys[i]+'o',label=label)
                   #ax.setp(baseline, color='r', linewidth=1.5)
               else:
                   pass
            else:
               pass
        
        ax.legend(prop={'size':10})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(Title)
        plt.tight_layout(rect=[0, 0, 1, 1])
        
        if len(Y.columns)>7:
             ax.legend(prop={'size':10},loc=1,mode='expand',numpoints=1,\
                       ncol=10,fancybox=True,fontsize='small')
             plt.grid(grid)
        else:
             plt.grid(grid,color=(1,0,0),linestyle='',linewidth='1')
            
        if len(text)>0:
              plt.text(0.1, 85.0,text,\
                     {'color': 'k', 'fontsize':10, 'ha': 'left', 'va': 'center',\
                      'bbox': dict(boxstyle=str(boxstyle), fc="w", ec="k", pad=0.3)})

        return plt.show()

def frame_from_dict(x,y,xlabel,ylabel,Title,mapping,grid,text,boxstyle,mode):
        """From dict with orient='index'"""
        m,n=y.shape
        X=np.array(np.ones((m,n)))
        
        if X.all()==None:
           X=np.array(X)
           X=pd.DataFrame(X)
           X.loc[0,:]=0
           X=np.cumsum(X)
        else:
           for ii in range(n):
               X[:,ii]=np.copy(y.index)
           X=pd.DataFrame(X)
        
        
        Labels= y.columns.values
        
        if mapping=='Log':
            Title='Log20 '+Title
            Y=np.log(np.abs(y))/np.log(20)
        else:
            Y=y
        Title=Title
        xlabel=xlabel
        ylabel=ylabel
        linestyles = OrderedDict(
                [('solid',               (0, ())),
                 ('mmmmm',               (0, (5,1,1,5,5,1))), 
                 ('nnnnn',               (0, (5,1,5,1,5,1))), 
                 ('densely dotted',      (0, (1, 1))),
                 
                 ('loosely dashed',      (0, (5, 10))),
                 ('dashed',              (0, (5, 5))),
                 ('densely dashed',      (0, (5, 1))),

         ('loosely dashdotted',  (0, (3, 10, 1, 10))),
         ('dashdotted',          (0, (3, 5, 1, 5))),
         ('densely dashdotted',  (0, (3, 1, 1, 1))),

         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
         ('xxxx',                  (0, (3,1,5,1,1,5))),
         ('yyyy',                  (0, (5,1,1,1,5,5))),
         ('zzzzzzzzzzzzzzzzz',     (0, (3,5,1,5,1,5))),
         ('loosely dotted',      (0, (1, 10))),
         ('dotted',              (0, (1, 5)))])
        Linestyle=linestyles.values()
        return plot_matrix_matrix(X,Y,Title,xlabel,ylabel,Labels,\
                                            Linestyle,0,mapping,grid,text,boxstyle,mode)

def float_with_format(x,nplaces):
            return float(''.join(seq for seq in ["{0:.",str(nplaces),"f}"]).\
                         format(round(x,nplaces)))
def float_list_with_format(X,nplaces):
            xx=[]
            for ff in X:
                xx.append(float_with_format(ff,nplaces))
            return xx
    
# full features subset
# Mapping training and testing data to [0,1] X->U

def mapping_zero_one(X):
    """Checking if X matrix is ok!
    """
    eps=1e-15
    f=lambda x: (x-x.min())/(x.max()-x.min() +eps)
    U=pd.DataFrame(X,columns=X.columns)
    for ii in X.columns:
        U[ii]=f(X[ii]).astype(float)
    return U

def Genlogistic(Nx,Nc):
    x=np.linspace(-20,20,Nx)
    C_shape=np.linspace(0.1,1.0,Nc)
    
    cdf=pd.DataFrame(np.empty((Nx,Nc)),index=x,columns=C_shape)
    pdf=pd.DataFrame(np.empty((Nx,Nc)),index=x,columns=C_shape)
    for c in C_shape:
        yy=Tools.cdf_gensigmoid(x,c)
        cdf.loc[:,c]=yy
        zz=Tools.pdf_gensigmoid(x,c)
        pdf.loc[:,c]=zz
    pdf_max=pdf.max()
    pdf=pdf/pdf_max
    frame_from_dict(x,pdf,'x-values','pdf','Genlogistic pdf [c-shape values]',\
                'equal',False,'','square',0)
    frame_from_dict(x,cdf,'x-values','cdf','Genlogistic cdf [c-shape values]',
                'equal',False,'','square',0) 
    return pdf,cdf

Genlogistic(200,7)
AUC_MEAN=[]
ACC_MEAN=[]
FPFN_MEAN=[]
AUC_TOP=[]
ACC_TOP=[]
FPFN_TOP=[]

for data in Datasets:
   
    if data=='Pima':
        """Indias Pima Dataset"""

        pima_tr = pd.read_csv('pima.tr.csv', index_col =0)
        pima_te = pd.read_csv ('pima.te.csv', index_col =0)

        # Training aata
        df=pima_tr

        #df.dropna(how="all", inplace=True) # drops the empty line at file-end
        columns=df.columns[:7]

        X_train=df[columns]
        Y_train=df['type']

        # Testing data
        de=pima_te
        X_test=de[columns]
        Y_test=de['type']
        
        # Mapping 'type' (Yes/No) to (1/0)
        f=lambda x:np.where(x=="No",0,1)
        V_train=f(Y_train).astype(int)
        V_test=f(Y_test).astype(int)  
        
        best_predictors_list=show_best_pca_predictors(X_train,X_train.shape[1],treshold)
         
    elif data=='Sonar':
         sonar_all_data=pd.read_csv('sonar.all-data.csv',index_col=0,\
                                    header=None,names=range(0,60))
         
         y=sonar_all_data.pop(59).to_frame()
         X=sonar_all_data
         X_train, X_test, Y_train, Y_test = train_test_split( \
            X, y, test_size=testsize, random_state=42,stratify=y)
        
         df=X_train
         de=X_test
         # Mapping 'type' (Yes/No) to (1/0)
         f=lambda x:np.where(x=="R",0,1)
         V_train=f(Y_train).astype(int)
         V_test=f(Y_test).astype(int)  
        
         best_predictors_list=show_best_pca_predictors(X_train,X_train.shape[1],treshold)
        
         
    elif data=='Ionosphere':
         ionosphere_all_data=pd.read_csv('ionosphere_data_kaggle.csv',index_col=0)
         y=ionosphere_all_data.pop('label').to_frame()
         X=ionosphere_all_data
         X_transf=X.loc[:,X.columns[1:]]
         X_train, X_test, Y_train, Y_test = train_test_split( \
            X_transf, y, test_size=testsize, random_state=42, \
            stratify=y)
         X_train=pd.DataFrame(X_train,columns=X.columns)
         X_train['feature2']=0
         X_test=pd.DataFrame(X_test,columns=X.columns)
         X_test['feature2']=0
         
         df=X_train
         de=X_test
         # Mapping 'type' (Yes/No) to (1/0)
         f=lambda x:np.where(x=="b",0,1)
         V_train=f(Y_train).astype(int)
         V_test=f(Y_test).astype(int)  
        
         best_predictors_list=show_best_pca_predictors(X_transf,X_transf.shape[1],treshold)
        
    else:
        pass 
         

    U_train=mapping_zero_one(X_train)
    U_test=mapping_zero_one(X_test)

    # Addition a constant to compute intercept to apply \
    #    regressoin models
    X_train_exog=sm.add_constant(X_train,prepend=True)
    X_test_exog=sm.add_constant(X_test,prepend=True)
    U_train_exog=sm.add_constant(U_train,prepend=True)
    U_test_exog=sm.add_constant(U_test,prepend=True)

    
    print('Find out ', len(best_predictors_list), ' predictors in ',data +\
          ' Dataset',best_predictors_list,  \
              'using PCA with a treshold of  %.2f ' %(treshold))

    NN=len(best_predictors_list)
    columns=best_predictors_list
    columns_subsets=[]
         
    for ii in range(NN):
        columns_subsets.append(columns[:ii+1])
    Ncols=len(columns_subsets)   

    
    index_c=np.linspace(0.1,1.0,100)
    
    auc_matrix=np.empty((len(index_c),Ncols))
    acc_matrix=np.empty((len(index_c),Ncols))
    fpfn_matrix=np.empty((len(index_c),Ncols)).astype('int')
 
    index_columns=[]
    for name in range(2,2+Ncols):
        index_columns.append(name-1)

    AUC=pd.DataFrame(auc_matrix,index=index_c,columns=index_columns)
    ACC=pd.DataFrame(acc_matrix,index=index_c,columns=index_columns)
    FPFN=pd.DataFrame(fpfn_matrix,index=index_c,columns=index_columns)
    
    for cols in columns_subsets:
        ii=len(cols)-1
        #print ("Features:",'', cols)
        columns_4=['const']
        for name in cols:
            columns_4.append(name)
        for c in index_c:
          file=open('c.txt','w')
          file.write(str(c))
          file.close()
          Confusion_Matrix=[]
          exog=U_train_exog[columns_4]
          endog=V_train
         
          model=smd.GenLogit(endog,exog).fit(disp=None,maxiter=100)
          
          y_val_=model.predict()
          y_val= list(map((lambda x: np.where(x<0.5,0,1)),y_val_))
          auc = metrics.roc_auc_score(endog,y_val)
         
          auc=float_with_format(auc,2)*100
          msa=ac.metrics(endog,y_val)
          msb=[auc]
          
          msb.extend(float_list_with_format(msa[:9],2))
         
          Confusion_Matrix.append(msb)
                    
          # Testing
          exog_test=U_test_exog[columns_4]
          y_test=V_test
          y_pred_=model.predict(exog_test)
          y_pred= list(map((lambda x: np.where(x<0.5,0,1)),y_pred_))
          auc=metrics.roc_auc_score(y_test,y_pred)
          
          auc=float_with_format(auc,2)*100
          msa=ac.metrics(y_test,y_pred)
          msb=[auc]
          msb.extend(float_list_with_format(msa[:9],2))
    
          AUC.loc[c,index_columns[ii]]=auc
          ACC.loc[c,index_columns[ii]]=float_with_format(msa[0],2)
          fpfn=(msa[6]+msa[8])*100.0/len(y_test)
          FPFN.loc[c,index_columns[ii]]=float_with_format(fpfn,2)
          
          Confusion_Matrix.append(msb)
          
          Confusion_Matrix=pd.DataFrame(Confusion_Matrix, \
                                  index=['Val','Test'+ str('%0.2f'%(c))],\
                                  columns=['AUC','ACC','TNTP','BIAS','PT=1','P=1','TP','FP','P=0','FN'])
    
    print(Confusion_Matrix,'\n','\n')
    
    #Graphs of Results
          
    
    if Ncols<8:
        frame_from_dict(AUC.index.values,AUC,'c-shape-values','AUC','AUC ROC CURVE '+ data +' Dataset',\
                'equal',False,'','square',0)
        frame_from_dict(ACC.index.values,ACC,'c-shape-values','ACC(%)','ACCURACY BINARY CLASSIFICATION '+ data + ' Dataset',\
                          'equal',False,'','square',0)
        frame_from_dict(FPFN.index.values,FPFN,'c-shape-values','FP+FN(%)','FP+FN BINARY CLASSIFICATION '+ data + ' Dataset',\
                          'equal',False,'','square',0)
    else:
        q,r=divmod(Ncols,7)
        for ii in range(q+1):
            if ii<q:
                AUC_split=AUC[AUC.columns[ii*7:(ii+1)*7]]
                
                ACC_split=ACC[ACC.columns[ii*7:(ii+1)*7]]
               
                FPFN_split=FPFN[FPFN.columns[ii*7:(ii+1)*7]]
            else:
                AUC_split=AUC[AUC.columns[q*7:Ncols+1]]
                
                ACC_split=ACC[ACC.columns[q*7:Ncols+1]]
                
                FPFN_split=FPFN[FPFN.columns[q*7:Ncols+1]]

            frame_from_dict(AUC.index.values,AUC_split,'c-shape-values','AUC','AUC ROC CURVE '+ data + ' Dataset',\
                          'equal',False,'','square',1)                
            frame_from_dict(ACC.index.values,ACC_split,'c-shape-values','ACC(%)','ACCURACY BINARY CLASSIFICATION '+ data + ' Dataset',\
                          'equal',False,'','square',1)
            frame_from_dict(FPFN.index.values,FPFN_split,'c-shape-values','FP+FN(%)','FP+FN BINARY CLASSIFICATION '+ data + ' Dataset',\
                          'equal',False,'','square',1)         
                
    
    index_cc=[]
    for ii in index_c:
        index_cc.append(round(ii,2))
    # compute mean values        
    AUC_m=np.transpose(AUC).mean()
    AUC_m=np.array(AUC_m).reshape((1,len(index_c)))
    AUC_MEAN.append(AUC_m)
    AUC_m=pd.DataFrame(AUC_m,index=['AUC MEAN'],columns=index_cc)
    Tools.table(AUC_m,'.2f','fancy_grid','AUC-MEAN ROC CURVE in '+ data +' Dataset', 40)
   
    ACC_m=np.transpose(ACC).mean()
    ACC_m=np.array(ACC_m).reshape((1,len(index_c)))
    ACC_MEAN.append(ACC_m)
    ACC_m=pd.DataFrame(ACC_m,index=['ACC MEAN'],columns=index_cc)
    Tools.table(ACC_m,'.2f','fancy_grid', 'ACC-MEAN in '+ data +' Dataset', 40)
    
    FPFN_m=np.transpose(FPFN).mean()
    FPFN_m=np.array(FPFN_m).reshape((1,len(index_c)))
    FPFN_MEAN.append(FPFN_m)
    FPFN_m=pd.DataFrame(FPFN_m,index=['FPFN MEAN'],columns=index_cc)
    Tools.table(FPFN_m,'.2f','fancy_grid', 'FPFN-MEAN in '+ data +' Dataset', 40)
    
    # Compute top values 

    AUC_t=np.transpose(AUC).max()
    AUC_t=np.array(AUC_t).reshape((1,len(index_c)))
    AUC_TOP.append(AUC_t)
    AUC_t=pd.DataFrame(AUC_t,index=['AUC MEAN'],columns=index_cc)
    Tools.table(AUC_t,'.2f','fancy_grid','AUC-MEAN ROC CURVE in '+ data +' Dataset', 40)
    
   
    ACC_t=np.transpose(ACC).max()
    ACC_t=np.array(ACC_t).reshape((1,len(index_c)))
    ACC_TOP.append(ACC_t)
    ACC_t=pd.DataFrame(ACC_t,index=['ACC MEAN'],columns=index_cc)
    Tools.table(ACC_t,'.2f','fancy_grid', 'ACC-MEAN in '+ data +' Dataset', 40)
    
    FPFN_t=np.transpose(FPFN).min()
    FPFN_t=np.array(FPFN_t).reshape((1,len(index_c)))
    FPFN_TOP.append(FPFN_t)
    FPFN_t=pd.DataFrame(FPFN_t,index=['FPFN MEAN'],columns=index_cc)
    Tools.table(FPFN_t,'.2f','fancy_grid', 'FPFN-MEAN in '+ data +' Dataset', 40)   

# MEAN VALUES
    
AUC_MEAN=Tools.data_list_to_matrix(AUC_MEAN,[len(index_cc),len(Datasets)])

AUC_MEAN=pd.DataFrame(AUC_MEAN,index=index_cc,columns=Datasets)

ACC_MEAN=Tools.data_list_to_matrix(ACC_MEAN,[len(index_cc),len(Datasets)])
 
ACC_MEAN=pd.DataFrame(ACC_MEAN,index=index_cc,columns=Datasets)

FPFN_MEAN=Tools.data_list_to_matrix(FPFN_MEAN,[len(index_cc),len(Datasets)])
   

FPFN_MEAN=pd.DataFrame(FPFN_MEAN,index=index_cc,columns=Datasets)


frame_from_dict(index_cc,AUC_MEAN,'c-shape-values','AUC-MEAN','AUC-MEAN ROC CURVE',\
                'equal',False,'','square',0)

frame_from_dict(index_cc,ACC_MEAN,'c-shape-values','ACC-MEAN(%)','ACCURACY BINARY CLASSIFICATION (AVERAGE VALUES)',\
                          'equal',False,'','square',0)
frame_from_dict(index_cc,FPFN_MEAN,'c-shape-values','FP+FN MEAN(%)','FP+FN BINARY CLASSIFICATION (AVERAGE VALUES) ',\
                          'equal',False,'','square',0)
   
# TOP VALUES
AUC_TOP=Tools.data_list_to_matrix(AUC_TOP,[len(index_cc),len(Datasets)])

AUC_TOP=pd.DataFrame(AUC_TOP,index=index_cc,columns=Datasets)

ACC_TOP=Tools.data_list_to_matrix(ACC_TOP,[len(index_cc),len(Datasets)])
 
ACC_TOP=pd.DataFrame(ACC_TOP,index=index_cc,columns=Datasets)

FPFN_TOP=Tools.data_list_to_matrix(FPFN_TOP,[len(index_cc),len(Datasets)])
   

FPFN_TOP=pd.DataFrame(FPFN_TOP,index=index_cc,columns=Datasets)


frame_from_dict(index_cc,AUC_TOP,'c-shape-values','AUC-MAX','AUC ROC CURVE (TOP VALUES)',\
                'equal',False,'','square',0)

frame_from_dict(index_cc,ACC_TOP,'c-shape-values','ACC-MAX(%)','ACCURACY BINARY CLASSIFICATION (TOP VALUES) ',\
                          'equal',False,'','square',0)
frame_from_dict(index_cc,FPFN_TOP,'c-shape-values','FP+FN MIN(%)','FP+FN BINARY CLASSIFICATION (MIN VALUES) ',\
                          'equal',False,'','square',0)
