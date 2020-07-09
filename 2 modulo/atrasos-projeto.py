#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system(u'curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv')



import pandas as pd

df = pd.read_csv('flightdata.csv')
print(df)


# In[ ]:




