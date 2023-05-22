#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# # Load Data

# In[5]:


train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')


# In[6]:


len(train_data) + len(test_data)


# In[7]:


train_data = pd.concat([train_data, test_data])


# # preprocessing Data

# In[8]:


train_data.Cancelled.value_counts()


# In[9]:


test_data.Cancelled.value_counts()


# In[10]:


train_data.Diverted.value_counts()


# In[11]:


test_data.Diverted.value_counts()


# In[12]:


train_data.drop(['Cancelled'],axis=1,inplace=True)
train_data.drop(['Diverted'],axis=1,inplace=True)
# 이 두 컬럼은 모두 0, 필요없으므로 제거
test_data.drop(['Cancelled'],axis=1,inplace=True)
test_data.drop(['Diverted'],axis=1,inplace=True)
# 이 두 컬럼은 모두 0, 필요없으므로 제거


# In[13]:


train_data.info(show_counts=True)


# In[14]:


len(train_data.dropna())


# # fill Nan value

# ## state

# In[16]:


airports = dict()
# airprots : id


# In[17]:


for _,v in tqdm(train_data.loc[:,['Origin_Airport','Origin_Airport_ID']].drop_duplicates().iterrows()):
    oai = v['Origin_Airport_ID']
    oa = v['Origin_Airport']
    if not airports.get(oa):
        airports[oa] = oai
    elif airports[oa] != oai:
        print(oa, oai)


# In[18]:


for i,v in tqdm(train_data.loc[:,['Destination_Airport','Destination_Airport_ID']].drop_duplicates().iterrows()):
    dai = v['Destination_Airport_ID']
    da = v['Destination_Airport']
    if not airports.get(da):
        airports[da] = dai
    elif airports[da] != dai:
        print(da, dai)


# In[20]:


len(airports) # 375 개 airport , ID 와 불일치 없음


# In[21]:


airports_state = dict()
# airprots : state


# In[22]:


airports_state['RIW'] = 'Wyoming'
airports_state['SHR'] = 'Wyoming'
airports_state['DDC'] = 'Kansas'
airports_state['DEC'] = 'Illinois'


# In[18]:


# train_data.loc[train_data['Origin_State'].isna(),'Origin_State'] = ''
# train_data.loc[train_data['Destination_State'].isna(),'Destination_State'] = ''


# In[23]:


for i,v in tqdm(train_data.loc[~train_data['Origin_State'].isna(),['Origin_Airport','Origin_State']].drop_duplicates().iterrows()):
    os = v['Origin_State']
    oa = v['Origin_Airport']
    if not airports_state.get(oa):
        airports_state[oa] = os
    elif airports_state[oa] != os:
        print(oa, os,airports_state[oa])


# In[25]:


for i,v in tqdm(train_data.loc[~train_data['Destination_State'].isna(),['Destination_Airport','Destination_State']].drop_duplicates().iterrows()):
    ds = v['Destination_State']
    da = v['Destination_Airport']
    if not airports_state.get(da):
        airports_state[da] = ds
    elif airports_state[da] != ds:
        print(da, ds,airports_state[da])


# In[26]:


#109015 개 Origin_State null값 있음
#109079 개 Destination_State null 값 있음


# In[29]:


train_data.loc[train_data['Origin_State'].isna() ,'Origin_State'] = train_data.loc[train_data['Origin_State'].isna(),'Origin_Airport'].apply(lambda x : airports_state.get(x))


# In[30]:


train_data.loc[train_data['Destination_State'].isna() ,'Destination_State'] = train_data.loc[train_data['Destination_State'].isna(),'Destination_Airport'].apply(lambda x : airports_state.get(x))


# In[24]:


# State 정보 채워줌


# In[32]:


train_data.info(show_counts=True)


#  > Origin_State, Destination_State는 100% 정확한 정보를 채울수 있었음

# In[33]:


len(train_data.dropna())


# ## airline

# > Airline과 Carrier_ID(DOT)가 일치한다는 가정하에 채우기 시작

# In[34]:


airline = dict()
# airline : id


# In[35]:


len(set(train_data['Airline']))


# In[36]:


for i,v in tqdm(train_data.loc[~train_data['Airline'].isna() * ~train_data['Carrier_ID(DOT)'].isna(),['Airline','Carrier_ID(DOT)']].drop_duplicates().iterrows()):
    al = v['Airline']
    ci = v['Carrier_ID(DOT)']
    if not airline.get(al):
        airline[al] = ci
    elif airline[al] != ci:
        print(ci, al, airline[al])
        break


# In[37]:


(train_data['Carrier_ID(DOT)'].isna() * ~train_data['Airline'].isna() ).sum()
# id 없고 airline 있는거 개수


# In[43]:


train_data.loc[train_data['Carrier_ID(DOT)'].isna() * ~train_data['Airline'].isna() ,'Carrier_ID(DOT)'] = train_data.loc[train_data['Carrier_ID(DOT)'].isna() * ~train_data['Airline'].isna() ,'Airline'].apply(lambda x : airline.get(x))


# In[44]:


(~train_data['Carrier_ID(DOT)'].isna() * train_data['Airline'].isna() ).sum()
# id 있고 airline없는거 개수


# In[45]:


airline_reverse = dict([(v, k) for k,v in airline.items()])
# id : airline


# In[46]:


train_data.loc[~train_data['Carrier_ID(DOT)'].isna() * train_data['Airline'].isna() ,'Airline'] = train_data.loc[~train_data['Carrier_ID(DOT)'].isna() * train_data['Airline'].isna() ,'Carrier_ID(DOT)'].apply(lambda x : airline_reverse.get(x))


# > 해당 출발 공항/ 도착공항 가는게 한개 밖에 없으면 넣음

# In[47]:


train_data['Airline'].isna().sum()


# In[49]:


train_data['ODA'] = train_data.Origin_Airport + '_' + train_data.Destination_Airport


# In[50]:


oda = train_data.loc[train_data.ODA.isin(set(train_data.loc[train_data.Airline.isna(),'ODA']))].groupby(by='ODA').Airline.apply(lambda x : x.value_counts().index[0] if len(x.value_counts()) == 1 else None)


# In[51]:


train_data.loc[train_data.Airline.isna(),['Airline']] = train_data.loc[train_data.Airline.isna(), 'ODA'].map(lambda x : oda[x])


# In[52]:


train_data.loc[train_data['Carrier_ID(DOT)'].isna(),['Carrier_ID(DOT)']] = train_data.loc[train_data['Carrier_ID(DOT)'].isna(), 'ODA'].map(lambda x : airline.get(oda[x]))


# In[53]:


train_data['Airline'].isna().sum()


# In[54]:


train_data.drop('ODA', axis=1 , inplace=True)


# In[55]:


train_data.info(show_counts=True)


# In[56]:


len(train_data.dropna())


# > airline 과 id는 아직 덜 참

# ## Carrier_Code(IATA) 

# Carrier_Code(IATA) -> 지맘대로임

# In[57]:


train_data['Airline'].isna().sum()


# ## Tail_Number

# Tail_Number 같은데 항공사 다를수 있음  
# 이 경우는 항공기를 다른 항공사에서 빌려서 운행한거

# In[58]:


train_data['Airline'].isna().sum()


# In[59]:


len(set(train_data['Tail_Number']))


# > 빌려준적 없는 애는 그거 Tail_Number 토대로 항공사 유추함 
# 

# In[60]:


trail = train_data.loc[train_data.Tail_Number.isin(set(train_data.loc[train_data.Airline.isna(),'Tail_Number']))].groupby(by='Tail_Number').Airline.apply(lambda x : x.value_counts().index[0] if len(x.value_counts()) == 1 else None)


# In[61]:


train_data.loc[train_data.Airline.isna(),['Airline']] = train_data.loc[train_data.Airline.isna(), 'Tail_Number'].map(lambda x : trail[x])
train_data.loc[train_data['Carrier_ID(DOT)'].isna(),['Carrier_ID(DOT)']] = train_data.loc[train_data['Carrier_ID(DOT)'].isna(), 'Tail_Number'].map(lambda x : airline.get(trail[x]))


# In[62]:


train_data['Airline'].isna().sum()


# > 빌려준적 있는 애중에서 그 공항에서 출발한 항공사가 한개면 걔로 넣음

# In[63]:


train_data['ot'] = train_data.loc[:,['Origin_Airport','Tail_Number']].apply(lambda x: '_'.join(x),axis=1)


# In[64]:


train_data['dt'] = train_data.loc[:,['Destination_Airport','Tail_Number']].apply(lambda x: '_'.join(x),axis=1)


# In[65]:


trail = train_data.loc[train_data.ot.isin(set(train_data.loc[train_data.Airline.isna(),'ot']))].groupby(by='ot').Airline.apply(lambda x : x.value_counts().index[0] if len(x.value_counts())==1 else None)


# In[66]:


train_data.loc[train_data.Airline.isna(),['Airline']] = train_data.loc[train_data.Airline.isna(), 'ot'].map(lambda x : trail[x])


# In[67]:


train_data.loc[train_data['Carrier_ID(DOT)'].isna(),['Carrier_ID(DOT)']] = train_data.loc[train_data['Carrier_ID(DOT)'].isna(), 'ot'].map(lambda x : airline.get(trail[x]))


# In[68]:


train_data['Carrier_ID(DOT)'].isna().sum()


# In[69]:


trail = train_data.loc[train_data.dt.isin(set(train_data.loc[train_data.Airline.isna(),'dt']))].groupby(by='dt').Airline.apply(lambda x : x.value_counts().index[0] if len(x.value_counts())==1 else None)


# In[70]:


train_data.loc[train_data.Airline.isna(),['Airline']] = train_data.loc[train_data.Airline.isna(), 'dt'].map(lambda x : trail[x])
train_data.loc[train_data['Carrier_ID(DOT)'].isna(),['Carrier_ID(DOT)']] = train_data.loc[train_data['Carrier_ID(DOT)'].isna(), 'dt'].map(lambda x : airline.get(trail[x]))


# In[71]:


train_data['Carrier_ID(DOT)'].isna().sum()


# In[72]:


train_data.drop('ot',axis=1, inplace=True)
train_data.drop('dt',axis=1, inplace=True)


# > 빌려준적 있는 애중에서 그 state에서 출발한 항공사가 한개면 걔로 넣음

# In[73]:


train_data['os'] = train_data.loc[:,['Origin_State','Tail_Number']].apply(lambda x: '_'.join(x),axis=1)


# In[74]:


train_data['ds'] = train_data.loc[:,['Origin_State','Tail_Number']].apply(lambda x: '_'.join(x),axis=1)


# In[75]:


trail = train_data.loc[train_data.os.isin(set(train_data.loc[train_data.Airline.isna(),'os']))].groupby(by='os').Airline.apply(lambda x : x.value_counts().index[0] if len(x.value_counts())==1 else None)


# In[76]:


train_data.loc[train_data.Airline.isna(),['Airline']] = train_data.loc[train_data.Airline.isna(), 'os'].map(lambda x : trail[x])
train_data.loc[train_data['Carrier_ID(DOT)'].isna(),['Carrier_ID(DOT)']] = train_data.loc[train_data['Carrier_ID(DOT)'].isna(), 'os'].map(lambda x : airline.get(trail[x]))


# In[77]:


train_data['Carrier_ID(DOT)'].isna().sum()


# In[78]:


trail = train_data.loc[train_data.ds.isin(set(train_data.loc[train_data.Airline.isna(),'ds']))].groupby(by='ds').Airline.apply(lambda x : x.value_counts().index[0] if len(x.value_counts())==1 else None)


# In[79]:


train_data.loc[train_data.Airline.isna(),['Airline']] = train_data.loc[train_data.Airline.isna(), 'ds'].map(lambda x : trail[x])
train_data.loc[train_data['Carrier_ID(DOT)'].isna(),['Carrier_ID(DOT)']] = train_data.loc[train_data['Carrier_ID(DOT)'].isna(), 'ds'].map(lambda x : airline.get(trail[x]))


# In[80]:


train_data['Carrier_ID(DOT)'].isna().sum()


# In[81]:


train_data.drop('os',axis=1, inplace=True)
train_data.drop('ds',axis=1, inplace=True)


# > 나머지는 그냥 많이 쓰는 항공회사로 넣음

# In[82]:


trail = train_data.loc[train_data.Tail_Number.isin(set(train_data.loc[train_data.Airline.isna(),'Tail_Number']))].groupby(by='Tail_Number').Airline.apply(lambda x : x.value_counts().index[0])


# In[83]:


train_data.loc[train_data.Airline.isna(),['Airline']] = train_data.loc[train_data.Airline.isna(), 'Tail_Number'].map(lambda x : trail[x])
train_data.loc[train_data['Carrier_ID(DOT)'].isna(),['Carrier_ID(DOT)']] = train_data.loc[train_data['Carrier_ID(DOT)'].isna(), 'Tail_Number'].map(lambda x : airline.get(trail[x]))


# In[84]:


train_data['Airline'].isna().sum()


# In[85]:


train_data.info(show_counts=True)


# In[86]:


len(train_data.dropna())


# ## Estimated_Departure_Time

# 시 분을 대충 평균 때리면 안되는 이유 :  
# 11:50 - 00: 10 의 평균은 00 : 00 이지만 12시가 되어버림  
# 각도로 계산해야함

# https://www.deeyook.com/post/circular-statistics-in-python-an-intuitive-intro

# In[87]:


import math

def time_to_radians(x):
    radians = x / (24 * 60)  * 2.0 * math.pi # 2pi
    return radians

def mean_angle(angles):
    angles = list(filter(np.isfinite,angles))
    if len(angles) == 0:
        return np.nan

    x = list(map(math.sin, angles))
    y = list(map(math.cos, angles))
    x_mean = np.mean(x)
    y_mean = np.mean(y) 
    avg = np.arctan2(x_mean, y_mean)
  
    return avg

def std_angle(angles):
    angles = list(filter(np.isfinite,angles))
    if len(angles) == 0:
        return np.nan
    x = list(map(math.sin, angles))
    y = list(map(math.cos, angles))
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    R = math.sqrt(x_mean**2 + y_mean**2)
    try:
        std = math.sqrt(-2 * math.log(R))
    except :
        return 0
    return std

def radians_to_min(x):
    if np.isnan(x):
        return np.nan
    if x < 0:
        x += math.pi * 2
    m = int(x / (2 * math.pi) * 24 * 60  ) 
    return m

def radians_to_time(x):
    if np.isnan(x):
        return np.nan
    if x < 0:
        x += math.pi * 2
    x = int(x / (2 * math.pi) * 24 * 60  ) 
    h, m = divmod(x, 60)
    return 100 * h + m

def time_to_min(x):
    if np.isnan(x):
        return np.nan
    h, m = divmod(x, 100)
    return 60 * h + m
def min_to_time(x):
    if np.isnan(x):
        return np.nan
    h, m = divmod(x, 60)
    return h * 100 + m


# Estimated_Departure_Time: 전산 시스템을 바탕으로 측정된 비행기의 출발 시간 (현지 시각, HH:MM 형식)   
# Estimated_Arrival_Time: 전산 시스템을 바탕으로 측정된 비행기의 도착 시간 (현지 시각, HH:MM 형식)  

# In[88]:


train_data.Estimated_Arrival_Time.isna().sum() # 도착 x


# In[89]:


train_data.Estimated_Departure_Time.isna().sum() # 출발 x


# In[90]:


(train_data.Estimated_Departure_Time.isna() | train_data.Estimated_Arrival_Time.isna()).sum() # 둘 중 하나 x


# In[92]:


(train_data.Estimated_Departure_Time.isna() * train_data.Estimated_Arrival_Time.isna()).sum() # 둘 다 x


# In[93]:


# 둘 중 하나는 있
((train_data.Estimated_Departure_Time.isna() & ~train_data.Estimated_Arrival_Time.isna()) | (~train_data.Estimated_Departure_Time.isna() & train_data.Estimated_Arrival_Time.isna())).sum()


# 둘 중 하나라도 없는 경우 > 206371  
# 둘 다 없는 경우 > 11688   
# 둘 중 하나는 있는 경우 > 194683  
# 94% 는 둘 중 하나라도 있음

# > 우선 도착 시간 출발 시간 둘 중에 하나만 이라도 있으면 걸린 시간으로 역산하자

# In[94]:


train_data['ODA'] = train_data.Origin_Airport + '_' + train_data.Destination_Airport


# In[95]:


import airporttime # https://pypi.org/project/airporttime/
import datetime


# In[96]:


airports_time = dict()


# In[97]:


for a in airports.keys():
    at= airporttime.AirportTime(iata_code=a).airport.__dict__
    # dst_offset => summer time 
    airports_time[a] = at['gmt_offset']
# 현지시각 -> GMT 0 로 변환
# 맹점 : 서머 타임이 적용되는가?


# In[98]:


train_data['OA_offset'] = train_data.Origin_Airport.map(lambda x : eval(airports_time[x]))
train_data['DA_offset'] = train_data.Destination_Airport.map(lambda x : eval(airports_time[x]))


# In[87]:


for _ in range(2):
    train_data['Estimated_Departure_Time2'] = ((train_data.Estimated_Departure_Time.map(time_to_min) - (train_data.OA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)
    train_data['Estimated_Arrival_Time2'] = ((train_data.Estimated_Arrival_Time.map(time_to_min) - (train_data.DA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)

    train_data['Flying_Time'] = train_data.Estimated_Arrival_Time2.apply(time_to_min) -  train_data.Estimated_Departure_Time2.apply(time_to_min) 
    # under_0 = train_data.loc[train_data.Flying_Time < 0].index
    train_data.loc[train_data.Flying_Time < 0,'Flying_Time' ] += 24 * 60

    oda_std = train_data.groupby(by='ODA').Flying_Time.std()
    train_data['ODA_std'] =  train_data.ODA.apply(lambda x :oda_std[x])
    oda_mean = train_data.groupby(by='ODA').Flying_Time.mean()
    train_data['ODA_mean'] = train_data.ODA.apply(lambda x :oda_mean[x])
    
    train_data.loc[(((train_data.Flying_Time < train_data.ODA_mean -2*train_data.ODA_std))|(train_data.Flying_Time > train_data.ODA_mean + 2*train_data.ODA_std)) & (train_data.ODA_std > 20),'Estimated_Arrival_Time'] = np.NaN


# In[88]:


train_data['Estimated_Departure_Time2'] = ((train_data.Estimated_Departure_Time.map(time_to_min) - (train_data.OA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)
train_data['Estimated_Arrival_Time2'] = ((train_data.Estimated_Arrival_Time.map(time_to_min) - (train_data.DA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)


# In[89]:


train_data.groupby(by='ODA').ODA_std.mean().hist()


# In[90]:


train_data.groupby(by='ODA').ODA_std.mean().hist()


# In[91]:


DoAx = ~train_data.Estimated_Departure_Time2.isna() & train_data.Estimated_Arrival_Time2.isna()


# In[92]:


(~train_data.Estimated_Departure_Time.isna() & train_data.Estimated_Arrival_Time.isna()).sum()


# In[93]:


DoAx.sum()


# In[94]:


train_data.loc[DoAx,'Estimated_Arrival_Time'] = (( train_data.loc[DoAx,'Estimated_Departure_Time2'].map(time_to_min) + train_data.loc[DoAx,'ODA_mean'] + (train_data.loc[DoAx,'DA_offset'] * 60) + 24 * 60) % (24 * 60)).map(min_to_time)


# In[95]:


train_data['Estimated_Departure_Time2'] = ((train_data.Estimated_Departure_Time.map(time_to_min) - (train_data.OA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)
train_data['Estimated_Arrival_Time2'] = ((train_data.Estimated_Arrival_Time.map(time_to_min) - (train_data.DA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)


# In[96]:


train_data[DoAx]


# In[97]:


DxAo = train_data.Estimated_Departure_Time2.isna() & ~train_data.Estimated_Arrival_Time2.isna()


# In[98]:


(train_data.Estimated_Departure_Time.isna() & ~train_data.Estimated_Arrival_Time.isna()).sum()


# In[99]:


DxAo.sum()


# In[100]:


train_data.loc[DxAo,'Estimated_Departure_Time'] = (( train_data.loc[DxAo,'Estimated_Arrival_Time2'].map(time_to_min) - train_data.loc[DxAo,'ODA_mean'] + (train_data.loc[DxAo,'OA_offset'] * 60) + 24 * 60) % (24 * 60)).map(min_to_time)


# In[101]:


train_data['Estimated_Departure_Time2'] = ((train_data.Estimated_Departure_Time.map(time_to_min) - (train_data.OA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)
train_data['Estimated_Arrival_Time2'] = ((train_data.Estimated_Arrival_Time.map(time_to_min) - (train_data.DA_offset * 60) + 24 * 60) % (24 * 60)).map(min_to_time)


# In[102]:


len(train_data.dropna())


# In[103]:


train_data.info(show_counts=True)


# In[104]:


train_data.Estimated_Arrival_Time.isna().sum() # 도착 x


# In[105]:


train_data.Estimated_Departure_Time.isna().sum() # 출발 x


# In[106]:


DxAx = train_data.Estimated_Departure_Time.isna() | train_data.Estimated_Arrival_Time.isna()


# In[107]:


(train_data.Estimated_Departure_Time.isna() | train_data.Estimated_Arrival_Time.isna()).sum() # 둘 중 하나 x


# In[108]:


(train_data.Estimated_Departure_Time.isna() * train_data.Estimated_Arrival_Time.isna()).sum() # 둘 다 x


# In[109]:


# 둘 중 하나는 있
((train_data.Estimated_Departure_Time.isna() & ~train_data.Estimated_Arrival_Time.isna()) | (~train_data.Estimated_Departure_Time.isna() & train_data.Estimated_Arrival_Time.isna())).sum()


# In[110]:


# <<중요>> 할일
# 둘다 없는 11688의 경우 그냥 A-to-B 출발 시간의 평균


# In[111]:


train_data.loc[:,'Estimated_Arrival_Time3'] = train_data.Estimated_Arrival_Time2.apply(time_to_radians)
train_data.loc[:,'Estimated_Departure_Time3'] = train_data.Estimated_Departure_Time2.apply(time_to_radians)
# 분 -> radian


# In[112]:


a3mean = train_data.groupby(by='ODA').Estimated_Arrival_Time3.apply(mean_angle).apply(radians_to_time)
a3std = train_data.groupby(by='ODA').Estimated_Arrival_Time3.apply(std_angle).apply(radians_to_time)
d3mean = train_data.groupby(by='ODA').Estimated_Departure_Time3.apply(mean_angle).apply(radians_to_time)
d3std = train_data.groupby(by='ODA').Estimated_Departure_Time3.apply(std_angle).apply(radians_to_time)


# In[113]:


train_data['a3mean'] = train_data.ODA.apply(lambda x :a3mean[x])
train_data['a3std'] = train_data.ODA.apply(lambda x :a3std[x])
train_data['d3mean'] = train_data.ODA.apply(lambda x :d3mean[x])
train_data['d3std'] = train_data.ODA.apply(lambda x :d3std[x])


# In[114]:


train_data[(train_data.ODA == 'ABE_ATL')]


# In[115]:


train_data.loc[DxAx,'Estimated_Departure_Time'] = (( train_data.loc[DxAx,'d3mean'].map(time_to_min)  + (train_data.loc[DxAx,'OA_offset'] * 60) + 24 * 60) % (24 * 60)).map(min_to_time)
train_data.loc[DxAx,'Estimated_Arrival_Time'] = (( train_data.loc[DxAx,'a3mean'].map(time_to_min) + (train_data.loc[DxAx,'DA_offset'] * 60) + 24 * 60) % (24 * 60)).map(min_to_time)


# In[116]:


train_data.loc[:,train_data.columns[:16]].info(show_counts=True)


# In[117]:


train_data.loc[:,train_data.columns[:16]]


# In[118]:


(train_data.Estimated_Departure_Time.isna() | train_data.Estimated_Arrival_Time.isna()).sum() # 둘 중 하나 x


# In[119]:


(train_data.Estimated_Departure_Time.isna() * train_data.Estimated_Arrival_Time.isna()).sum() # 둘 다 x


# In[120]:


# 둘 중 하나는 있
((train_data.Estimated_Departure_Time.isna() & ~train_data.Estimated_Arrival_Time.isna()) | (~train_data.Estimated_Departure_Time.isna() & train_data.Estimated_Arrival_Time.isna())).sum()


# In[121]:


train_data.loc[:,'Estimated_Departure_Time'].fillna(train_data.loc[:,'Estimated_Departure_Time'].mean(),inplace=True)
train_data.loc[:,'Estimated_Arrival_Time'].fillna(train_data.loc[:,'Estimated_Arrival_Time'].mean(),inplace=True)


# In[122]:


train_data.loc[:,train_data.columns[:16]].info(show_counts=True)


# In[123]:


test_data = train_data.loc[train_data.ID.str.contains('TEST')]
train_data = train_data.loc[train_data.ID.str.contains('TRAIN')]


# In[125]:


train_data = train_data.loc[:,['Month','Estimated_Departure_Time','Estimated_Arrival_Time','Origin_Airport_ID','Origin_State','Destination_Airport_ID','Destination_State','Distance','Carrier_ID(DOT)','Tail_Number','Delay']].dropna()


# In[126]:


test_data = test_data.loc[:,['Month','Estimated_Departure_Time','Estimated_Arrival_Time','Origin_Airport_ID','Origin_State','Destination_Airport_ID','Destination_State','Distance','Carrier_ID(DOT)','Tail_Number']]

