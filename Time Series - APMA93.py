#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
germ_tseries_df = pd.read_csv("exp1_germ_test_2023.csv")
germ_tseries_df


# ## Plot germination time series ##

# In[2]:


import matplotlib.pyplot as plt


# In[3]:


plt.figure(figsize=(5,3), dpi=150)
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["A"], marker="o", label="A")
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["B"], marker="s", label="B")
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["C"], marker="d", label="C")
plt.xlabel("hours")
plt.ylabel("Last germinated seeds measured")
plt.legend()


# In[4]:


treatment_col_names = list(germ_tseries_df.columns[1:])
treatment_col_names


# In[5]:


for treatm in treatment_col_names:
    germ_tseries_df[treatm+"_cum"] = germ_tseries_df[treatm].cumsum()
germ_tseries_df


# In[6]:


plt.figure(figsize=(5,3), dpi=150)
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["A"], marker="o", label="A")
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["B"], marker="s", label="B")
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["C"], marker="d", label="C")
plt.xlabel("hours")
plt.ylabel("Last germinated seeds measured")
plt.legend()

plt.figure(figsize=(5,3), dpi=150)
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["A_cum"], marker="o", label="A")
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["B_cum"], marker="s", label="B")
plt.plot(germ_tseries_df["Hours"], germ_tseries_df["C_cum"], marker="d", label="C")
plt.xlabel("hours")
plt.ylabel("Last germinated seeds measured")
plt.legend()


# In[ ]:





# In[ ]:




