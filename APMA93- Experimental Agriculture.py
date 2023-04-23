#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("My name is Femi Royal")


# # Analysis of Data Germination Test (Experimental 1) 

# # General Data for the team ##

# In[2]:


import pandas as pd
df = pd.read_csv('Experimental_agriculture2023_germination_test_data.csv')
print(df)


# In[3]:


df["Normal germination"]


# In[4]:


columns_list = list(df.columns)


# In[5]:


columns_list


# In[6]:


df['prop_normal_germin']= df['Normal germination']/df['Total']
prop_germ_mean = df['prop_normal_germin'].mean()
prop_germ_mean


# In[7]:


df


# In[8]:


df.loc[[1,5]]


# In[9]:


df["Normal germination"] [9]


# In[10]:


raw_data_columns = ["Normal germination", "Abnormal germination", "Dead"]
for col_name in raw_data_columns:
    for value in df[col_name]:
        print (value)
        if (value >= 0) and (value <= 50):
            print("OK")
print("we have a problem")


# ### Add columns based on transformations/mathematical operations on existing columns

# In[11]:


df['prop_normal_germin'] = df['Normal germination']/df['Total']


# In[12]:


df


# ### Compute the mean proportion of normally germinated seeds in each treatment

# In[13]:


df.groupby('Treatment').mean()


# In[14]:


prop_norm_germ_df_pivot = df.pivot(index="Replicate ", columns="Treatment", values="prop_normal_germin")


# In[15]:


prop_norm_germ_df_pivot


# In[16]:


prop_abnormal_df_pivot = df.pivot(index="Replicate ", columns="Treatment", values="Abnormal germination")


# In[17]:


prop_abnormal_df_pivot


# In[18]:


prop_dead_germ_df_pivot = df.pivot(index="Replicate ", columns="Treatment", values="Dead")


# In[19]:


prop_dead_germ_df_pivot


# In[20]:


df['Prop_dead_germin'] = df['Dead']/df['Total']


# In[21]:


df


# In[22]:


df['Prop_abnormal_germin'] = df['Dead']/df['Total']


# In[23]:


df


# In[24]:


prop_germ_mean = df['prop_normal_germin'].mean()


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3), dpi=150)
ax = sns.stripplot(x='Treatment', y='prop_normal_germin', data=df, color='red')
ax = sns.pointplot(x='Treatment', y='prop_normal_germin', data=df, join=False, ci=95)
#plt.scatter([1.0], [prop_germ_mean], s=40.0, marker='s', color='Darkblue')
ax.set_ylabel("Proportion normal germination")
plt.show()


# In[26]:


ax = sns.stripplot(x='Treatment', y='Prop_abnormal_germin', data=df, color='black')
ax = sns.pointplot(x='Treatment', y='Prop_abnormal_germin', data=df, join=False, ci=95)
ax.set_ylabel("Proportion abnormal germination")
plt.figure(figsize=(5,3), dpi=160)
plt.show()


# In[27]:


plt.figure(figsize=(5,3), dpi=150)
ax = sns.stripplot(x='Treatment', y='Prop_dead_germin', data=df, color='green')
ax = sns.pointplot(x='Treatment', y='Prop_dead_germin', data=df, join=False, ci=95)
#plt.scatter([1.0], [prop_germ_mean], s=40.0, marker='s', color='Darkblue')
ax.set_ylabel("Proportion Dead")
plt.show()


# # Illustration: what is a normal distribution? ##

# In[28]:


import numpy as np
import matplotlib.pyplot as pt 

norm_distr_val_ar = np.random.normal(loc=0.0, scale=1.0, size=10)
print(norm_distr_val_ar)


# In[29]:


ax = sns.displot(norm_distr_val_ar)
plt.show()


# ### Compute the residuals, examine their distribution

# In[30]:


from statsmodels.formula.api import ols
model = ols('prop_normal_germin ~ C(Treatment)', data=df).fit()


# In[31]:


#model.resid
axl = sns.displot(model.resid)
plt.show()


# ## Construct the QQ Plot of the distribution of residuals -- compared to the normal distribution

# In[32]:


import statsmodels.api as sm
sm.qqplot(norm_distr_val_ar, line='s')


# In[33]:


sm.qqplot(norm_distr_val_ar, line='s')


# In[34]:


import numpy as np
df['prop_normal_germin_angtransf'] = np.arcsin(np.sqrt(df['prop_normal_germin']))


# In[35]:


df['prop_normal_germin_angtransf'] 


# In[54]:


import numpy as np
df['Prop_abnormal_germin_angtransf'] = np.arcsin(np.sqrt(df['Prop_abnormal_germin_angtransf']))


# In[56]:


df['Prop_abnormal_germin_angtransf'] 


# In[50]:


import numpy as np
df['Prop_dead_germin_angtransf'] = np.arcsin(np.sqrt(df['Prop_dead_germin']))


# In[51]:


df['Prop_dead_germin_angtransf'] 


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3), dpi=150)
ax = sns.stripplot(x='Treatment', y='prop_normal_germin_angtransf', data=df, color='red')
ax = sns.pointplot(x='Treatment', y='prop_normal_germin_angtransf', data=df, join=False, ci=95)
#plt.scatter([1.0], [prop_germ_mean], s=40.0, marker='s', color='Darkblue')
ax.set_ylabel("Proportion normal germination (ang transf)")
plt.show()


# In[37]:


model_angtransf = ols('prop_normal_germin ~ C(Treatment)', data=df).fit()


# In[38]:


anova_table = sm.stats.anova_lm(model_angtransf, typ=1)


# In[39]:


anova_table


# In[40]:


0.05870/0.016396


# In[41]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
MultiComparison(df["prop_normal_germin_angtransf"], df['Treatment'])


# In[42]:


pip install statsmodels


# # Kruskal.Wallis test -non parametric statistical test (analogue of ANOVA) #

# In[43]:


prop_norm_germ_df_pivot


# In[57]:


import scipy.stats as stats
stats.kruskal(prop_norm_germ_df_pivot['A'], prop_norm_germ_df_pivot['B'], prop_norm_germ_df_pivot['C'])


# In[58]:


from scikit_posthocs import posthoc_dunn


# In[59]:


pip install scikit-posthocs


# In[ ]:




