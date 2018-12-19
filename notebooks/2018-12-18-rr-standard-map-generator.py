#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


plt.rcParams['figure.figsize'] = (15,5)


# ```cpp
# void StandardN_next(StandardN *unit, int inNumSamples)
# {
# 	float *out = ZOUT(0);
# 	float freq = ZIN0(0);
# 	double k = ZIN0(1);
# 	double x0 = ZIN0(2);
# 	double y0 = ZIN0(3);
# 
# 	double xn = unit->xn;
# 	double output = (xn - PI) * RECPI;
# 	double yn = unit->yn;
# 	float counter = unit->counter;
# 
# 	float samplesPerCycle;
# 	if(freq < unit->mRate->mSampleRate)
# 		samplesPerCycle = unit->mRate->mSampleRate / sc_max(freq, 0.001f);
# 	else samplesPerCycle = 1.f;
# 
# 	if((unit->x0 != x0) || (unit->y0 != y0)){
# 		unit->x0 = xn = x0;
# 		unit->y0 = yn = y0;
# 	}
# 
# 	for (int i=0; i<inNumSamples; ++i) {
# 		if(counter >= samplesPerCycle){
# 			counter -= samplesPerCycle;
# 
# 			yn = yn + k * sin(xn);
# 			yn = mod2pi(yn);
# 
# 			xn = xn + yn;
# 			xn = mod2pi(xn);
# 
# 			output = (xn - PI) * RECPI;
# 		}
# 		counter++;
# 		ZXP(out) =  output;
# 	}
# 
# 	unit->xn = xn;
# 	unit->yn = yn;
# 	unit->counter = counter;
# }
# ```

# In[3]:


RECPI = 0.3183098861837907

def mod2pi(x):
    return x % (2 * math.pi)

def nextStandardN(k, x0, y0):
    yn = y0
    xn = x0
    while True:
        yn = yn + k * math.sin(xn)
        yn = mod2pi(yn)

        xn = xn + yn
        xn = mod2pi(xn)

        yield (xn - math.pi) * RECPI


# In[4]:


num_samps = 100000
sn = nextStandardN(1.0, 0.5, 0)
x = np.array([])
for i, xi in zip(range(num_samps), sn):
    x = np.append(x, xi)


# In[5]:


x


# In[6]:


plt.plot(x[:1000])


# In[7]:


plt.plot(np.diff(x[:1000]))


# In[8]:


plt.plot(np.diff(np.diff(x[:1000])))


# In[9]:


plt.plot(np.diff(np.diff(np.diff(x[:1000]))))


# In[10]:


plt.plot(x[:10000])


# In[11]:


plt.plot(np.append(np.array([0]), np.diff(x[:100])))


# In[12]:


x[:10]


# In[13]:


np.append(np.array([0]), np.diff(x[:10]))


# In[14]:


pd.plotting.autocorrelation_plot(x[:100])


# In[15]:


pd.plotting.autocorrelation_plot(np.diff(x[:100]))


# In[ ]:




