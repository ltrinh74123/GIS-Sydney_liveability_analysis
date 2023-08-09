#!/usr/bin/env python
# coding: utf-8

# Due to some technical issues, we weren't able to reload all the codes before submission, but the charts and maps are still loaded below

# In[ ]:


#Code for connecting Jupyter to PGADMIN
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras
import json
import os
import geojson
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from geoalchemy2 import Geometry, WKTElement
import matplotlib.pyplot as plt
import numpy as np


credentials = "Credentials.json"
def pgconnect(credential_filepath, db_schema="public"):
    with open(credential_filepath) as f:
        db_conn_dict = json.load(f)
        host       = db_conn_dict['host']
        db_user    = db_conn_dict['user']
        db_pw      = db_conn_dict['password']
        default_db = db_conn_dict['user']
        try:
            db = create_engine('postgresql+psycopg2://'+db_user+':'+db_pw+'@'+host+'/'+default_db, echo=False)
            conn = db.connect()
            print('Connected successfully.')
        except Exception as e:
            print("Unable to connect to the database.")
            print(e)
            db, conn = None, None
        return db,conn

    
def query(conn, sqlcmd, args=None, df=True):
    result = pd.DataFrame() if df else None
    try:
        if df:
            result = pd.read_sql_query(sqlcmd, conn, params=args)
        else:
            result = conn.execute(sqlcmd, args).fetchall()
            result = result[0] if len(result) == 1 else result
    except Exception as e:
        print("Error encountered: ", e, sep='\n')
    return result
db, conn = pgconnect(credentials)
conn.execute("set search_path to PUBLIC")


srid = 4283
def create_wkt_element(geom, srid):
    if geom.geom_type == 'Polygon':
        geom = MultiPolygon([geom])
    return WKTElement(geom.wkt, srid)


# # Adding tables to Database

# In[2]:


SA2 = gpd.read_file("SA2_2016_AUST/SA2_2016_AUST.shp")
SA2_ori = SA2.copy()  # creating a copy of the original for later
SA2 = SA2.dropna()
SA2['geom'] = SA2['geometry'].apply(lambda x: create_wkt_element(geom=x,srid=srid))# applying the function
SA2 = SA2.drop(columns="geometry")  # deleting the old copy
SA2.columns = SA2.columns.str.lower()

conn.execute("""
DROP TABLE IF EXISTS sa2 CASCADE;
CREATE TABLE sa2 (
    sa2_main16 INTEGER,
    sa2_5dig16 INTEGER, 
    sa2_name16 TEXT PRIMARY KEY,
    sa3_code16 INTEGER,
    sa3_name16 TEXT,
    sa4_code16 INTEGER,
    sa4_name16 TEXT,
    gcc_code16 TEXT,
    gcc_name16 TEXT,
    ste_code16 INTEGER,
    ste_name16 TEXT,
    areasqkm16 FLOAT,
    geom GEOMETRY(MULTIPOLYGON,4283)
);""")

SA2.to_sql('sa2', conn, if_exists='append', index=False, dtype={'geom': Geometry('MULTIPOLYGON', srid)})


# In[4]:


conn.execute("""
DROP TABLE IF EXISTS neighbourhoods;
CREATE TABLE neighbourhoods (
    area_id INTEGER PRIMARY KEY,
    area_name VARCHAR(100) REFERENCES sa2(sa2_name16),
    land_area NUMERIC,
    population NUMERIC,
    number_of_dwellings NUMERIC,
    median_annual_household_income NUMERIC,
    avg_monthly_rent NUMERIC,
    age0_4 NUMERIC,
    age5_9 NUMERIC,
    age10_14 NUMERIC,
    age15_19 NUMERIC
);""")


NeighbourhoodsData = pd.read_csv("Neighbourhoods.csv")
NeighbourhoodsData = NeighbourhoodsData.iloc[: , 1:]
NeighbourhoodsData = NeighbourhoodsData.drop(columns="number_of_businesses")
NeighbourhoodsData = NeighbourhoodsData.replace(',','', regex=True)
NeighbourhoodsData.rename(columns = {'0_4':'age0_4', '5_9':'age5_9','10_14':'age10_14','15_19':'age15_19'}, inplace = True)
NeighbourhoodsData_Cleaned = NeighbourhoodsData.dropna()
NeighbourhoodsData_Cleaned.to_sql('neighbourhoods', conn, if_exists='append', index=False)


# In[ ]:


conn.execute("""
DROP TABLE IF EXISTS businessstats;
CREATE TABLE businessstats(
   area_id INTEGER PRIMARY KEY,
   area_name VARCHAR(50) REFERENCES sa2(sa2_name16),
   accommodation_and_food_services NUMERIC,
   retail_trade NUMERIC,
   health_care_and_social_assistance NUMERIC
);""")
BusinessStatsData = pd.read_csv('BusinessStats.csv')
BusinessStatsData = BusinessStatsData.drop(columns=["number_of_businesses","agriculture_forestry_and_fishing", "public_administration_and_safety","transport_postal_and_warehousing"])

i = BusinessStatsData[BusinessStatsData['area_name'].str.contains("Unknown")].index.to_list()
BusinessStatsData.drop(i, axis=0, inplace=True)

BusinessStatsData_Cleaned = BusinessStatsData.dropna()
BusinessStatsData_Cleaned.to_sql("businessstats", con=conn, if_exists='append', index=False)


# In[ ]:


breakenterdwelling = gpd.read_file("break_and_enter/BreakEnterDwelling_JanToDec2021.shp")
breakenterdwelling_original = breakenterdwelling.copy()  # creating a copy of the original for later
breakenterdwelling['geom'] = breakenterdwelling['geometry'].apply(lambda x: create_wkt_element(geom=x,srid=srid))  # applying the function
breakenterdwelling = breakenterdwelling.drop(columns="geometry")  # deleting the old copy
breakenterdwelling=breakenterdwelling.dropna()
breakenterdwelling.columns = breakenterdwelling.columns.str.lower()

conn.execute("""
DROP TABLE IF EXISTS breakenterdwelling;
CREATE TABLE breakenterdwelling (
    objectid INTEGER PRIMARY KEY, 
    contour FLOAT, 
    density VARCHAR(80), 
    orig_fid INTEGER,
    shape_leng FLOAT, 
    shape_area FLOAT,
    geom GEOMETRY(MULTIPOLYGON,4283)
);""")

breakenterdwelling.to_sql("breakenterdwelling", conn, if_exists='append', index=False, dtype={'geom': Geometry('MULTIPOLYGON', srid)})


# In[ ]:


## Drinking fountain GEOJSON
drinkingfountains = gpd.read_file("Drinking_fountains_(water_bubblers).geojson")
drinkingfountains = drinkingfountains.copy()
drinkingfountains['geom'] = drinkingfountains['geometry'].apply(lambda x: create_wkt_element(geom=x,srid=srid))  # applying the function
drinkingfountains = drinkingfountains.drop(columns=["geometry","Accessible"])  # deleting the old copy
drinkingfountains=drinkingfountains.dropna()
drinkingfountains.columns = drinkingfountains.columns.str.lower()

conn.execute("""
DROP TABLE IF EXISTS drinkingfountains;
CREATE TABLE drinkingfountains (
    objectid INTEGER PRIMARY KEY, 
    site_name VARCHAR(80), 
    suburb VARCHAR(80), 
    location VARCHAR(80), 
    geom GEOMETRY(POINT,4283)
);""")

drinkingfountains.to_sql("drinkingfountains", conn, if_exists='append', index=False, dtype={'geom': Geometry('MULTIPOLYGON', srid)})


# In[59]:


##STREET BINS SHP
streetbin = gpd.read_file("Street_litter_bins/Street_litter_bins.shp")
streetbin_ori = streetbin.copy()
streetbin['geom'] = streetbin['geometry'].apply(lambda x: WKTElement(x.wkt, srid=srid))
streetbin = streetbin.drop(columns="geometry")
streetbin = streetbin.dropna()
streetbin.columns = streetbin.columns.str.lower()

conn.execute("""
DROP TABLE IF EXISTS streetbin;
CREATE TABLE streetbin (
    objectid INTEGER PRIMARY KEY,
    central_as INTEGER,
    geom GEOMETRY(POINT,4283)
);""")

streetbin.to_sql('streetbin', conn, if_exists='append', index=False, dtype={'geom': Geometry('POINT', srid)})


# In[11]:


# PRIMARY CATCHMENTS SHP
catchments_primary = gpd.read_file("school_catchments/catchments_primary.shp")
catchments_primary_original = catchments_primary.copy()  # creating a copy of the original for later
catchments_primary.columns = catchments_primary.columns.str.lower()
catchments_primary['geom'] = catchments_primary['geometry'].apply(lambda x: create_wkt_element(geom=x,srid=srid))# applying the function
catchments_primary = catchments_primary.drop(columns=["geometry","add_date", "kindergart", "year1", "year2", "year3","year4", "year5", "year6", "year7", "year8", "year9", "year10", "year11", "year12", "priority"])  # deleting the old copy
catchments_primary = catchments_primary.dropna()
# SECONDARY CATCHMENTS SHP
catchments_secondary = gpd.read_file("school_catchments/catchments_secondary.shp")
catchments_secondary_original = catchments_secondary.copy()  # creating a copy of the original for later
catchments_secondary.columns = catchments_secondary.columns.str.lower()
catchments_secondary['geom'] = catchments_secondary['geometry'].apply(lambda x: create_wkt_element(geom=x,srid=srid))# applying the function
catchments_secondary = catchments_secondary.drop(columns=["geometry", "add_date","kindergart", "year1", "year2", "year3","year4", "year5", "year6", "year7", "year8", "year9", "year10", "year11", "year12", "priority"])  # deleting the old copy
catchments_secondary = catchments_secondary.dropna()
# FUTURE CATCHMENTS SHP
catchments_future = gpd.read_file("school_catchments/catchments_future.shp")
catchments_future_original = catchments_future.copy()  # creating a copy of the original for later
catchments_future['geom'] = catchments_future['geometry'].apply(lambda x: create_wkt_element(geom=x,srid=srid))# applying the function
catchments_future.columns = catchments_future.columns.str.lower()
catchments_future = catchments_future.drop(columns=["geometry", "add_date", "kindergart", "year1", "year2", "year3","year4", "year5", "year6", "year7", "year8", "year9", "year10", "year11", "year12"])  # deleting the old copy
catchments_future = catchments_future.dropna()
frames = [catchments_primary, catchments_secondary, catchments_future]
catchments = pd.concat(frames)

conn.execute("""
DROP TABLE IF EXISTS catchments;
CREATE TABLE catchments (
    use_id INTEGER, 
    catch_type TEXT, 
    use_desc VARCHAR(80), 
    geom GEOMETRY(MULTIPOLYGON,4283)
);""")
catchments.to_sql("catchments", conn, if_exists='append', index=False, dtype={'geom': Geometry('MULTIPOLYGON', srid)})


# # Indexes

# In[12]:


conn.execute("""
DROP INDEX IF EXISTS geom_idx;
CREATE INDEX geom_idx ON sa2 USING GIST (geom);

DROP INDEX IF EXISTS gcc_name16;
CREATE INDEX gcc_name16 ON sa2(gcc_name16);

DROP INDEX IF EXISTS sa3_idx;
CREATE INDEX sa3_idx ON sa2(sa3_name16);
""")


# # Calculating Z-scores for greater Sydney

# In[3]:


#SCHOOLS
from scipy.stats import zscore

sql = """SELECT sa2_name16 as "suburb", 
COUNT(use_id) as "number_of_schools", 
SUM(age0_4 + age5_9 + age10_14 + age15_19)/1000 AS "youths_in_area_per_1000",
CAST(COUNT(use_id)/(SUM(age0_4 + age5_9 + age10_14 + age15_19)/1000) AS NUMERIC)  AS "schools_per_1000_youths"
FROM sa2 S
LEFT JOIN catchments C on ST_Contains(S.geom, C.geom)
RIGHT JOIN neighbourhoods N ON (N.area_name = S.sa2_name16)
WHERE S.gcc_name16 = 'Greater Sydney'
GROUP BY sa2_name16
ORDER BY suburb; """

school_df = pd.read_sql(sql, conn)
school_df['z_school'] = zscore(school_df['schools_per_1000_youths'])
school_df


# In[42]:


#CRIME
sql = """
SELECT
sa2_name16 as "suburb", 
ST_AREA(S.geom) AS "total_area",
SUM(shape_area) AS "sum_of_hotspot_areas",
SUM(shape_area)/ST_AREA(S.geom) AS "sum_of_hotspot_areas_divided_by_total_area"
FROM sa2 S
LEFT JOIN breakenterdwelling B on ST_Contains(S.geom, B.geom)
RIGHT JOIN neighbourhoods N ON (N.area_name = S.sa2_name16)
WHERE S.gcc_name16 = 'Greater Sydney'
AND density = 'High Density'
GROUP BY suburb, S.geom
ORDER BY suburb; 
"""
crime_df = pd.read_sql(sql, conn)
crime_df['z_crime'] = zscore(crime_df['sum_of_hotspot_areas_divided_by_total_area'], nan_policy='omit').fillna(0)
crime_df


# In[5]:


#ACCOMODATION 
sql = """SELECT 
sa2_name16 as "suburb",
accommodation_and_food_services/(N.population/1000) AS "accomm_per_1000_people"
FROM businessstats B
RIGHT JOIN neighbourhoods N USING (area_name)
LEFT JOIN sa2 S on (B.area_name = S.sa2_name16)
WHERE S.gcc_name16 = 'Greater Sydney'
ORDER BY sa2_name16; """

accomm_df = pd.read_sql(sql, conn)
accomm_df['z_accomm'] = zscore(accomm_df['accomm_per_1000_people'])
accomm_df


# In[6]:


#RETAIL
sql = """SELECT 
sa2_name16 as "suburb",
retail_trade/(N.population/1000) AS "retail_per_1000_people"
FROM businessstats B
RIGHT JOIN neighbourhoods N USING (area_name)
LEFT JOIN sa2 S on (B.area_name = S.sa2_name16)
WHERE S.gcc_name16 = 'Greater Sydney'
ORDER BY sa2_name16; """

retail_df = pd.read_sql(sql, conn)
retail_df['z_retail'] = zscore(retail_df['retail_per_1000_people'])
retail_df


# In[7]:


#Health
sql = """SELECT 
sa2_name16 as "suburb",
health_care_and_social_assistance/(N.population/1000) AS "health_per_1000_people"
FROM businessstats B
RIGHT JOIN neighbourhoods N USING (area_name)
LEFT JOIN sa2 S on (B.area_name = S.sa2_name16)
WHERE S.gcc_name16 = 'Greater Sydney'
ORDER BY sa2_name16; """

health_df = pd.read_sql(sql, conn)
health_df['z_health'] = zscore(health_df['health_per_1000_people'])
health_df


# In[8]:


# Z score do not include street bins and drinking fountain

z_without_add = pd.merge(school_df,accomm_df, on = 'suburb')
z_without_add = pd.merge(z_without_add,retail_df, on = 'suburb')
z_without_add = pd.merge(z_without_add,crime_df, on = 'suburb')
z_without_add = pd.merge(z_without_add,health_df, on = 'suburb')

z_without_add["zscore"] = z_without_add["z_school"] + z_without_add["z_accomm"] + z_without_add["z_retail"] - z_without_add["z_crime"] + z_without_add["z_health"]
final_z = z_without_add.drop(["number_of_schools", "youths_in_area_per_1000", "schools_per_1000_youths", "accomm_per_1000_people", "retail_per_1000_people", "total_area", "sum_of_hotspot_areas", "sum_of_hotspot_areas_divided_by_total_area", "health_per_1000_people"],axis = 1)
final_z["sigmoid"] = 1 / (1 + np.exp(-final_z["zscore"]))
final_z


# # Inner City Z score

# In[9]:


sql = """ SELECT sa2_name16 as suburb,
CAST(COUNT(objectid)/(N.population/1000) AS NUMERIC)  AS "drinking_fountains_per_1000"
FROM sa2 S
LEFT JOIN drinkingfountains D on ST_Contains(S.geom, D.geom)
RIGHT JOIN neighbourhoods N ON (N.area_name = S.sa2_name16)
WHERE sa3_name16 = 'Sydney Inner City' 
GROUP BY sa2_name16, N.population
ORDER BY sa2_name16;
"""

drinking_df = pd.read_sql(sql, conn)
drinking_df['z_drink'] = zscore(drinking_df['drinking_fountains_per_1000'])
drinking_df


# In[10]:


#COUNT OF BINS PER 1000 PEOPLE
sql = """
SELECT sa2_name16 AS "suburb", 
count(objectid) AS "number_of_bins",
population/1000 AS "pop_per_1000",
CAST(COUNT(objectid)/(population/1000) AS NUMERIC) AS "bins_per_1000_people"
FROM sa2 S
LEFT JOIN streetbin B ON ST_Contains(S.geom, B.geom)
RIGHT JOIN neighbourhoods N ON (N.area_name = S.sa2_name16)
WHERE sa3_name16 = 'Sydney Inner City'
GROUP BY sa2_name16, population
ORDER BY sa2_name16;
"""

bin_df = pd.read_sql(sql, conn)
bin_df['z_bin'] = zscore(bin_df['bins_per_1000_people'])
bin_df


# In[12]:


#Separate z-score calculations for Inner sydney
sql = """SELECT 
sa2_name16 as "suburb",
accommodation_and_food_services/(N.population/1000) AS "accomm_per_1000_people"
FROM businessstats B
RIGHT JOIN neighbourhoods N USING (area_name)
LEFT JOIN sa2 S on (B.area_name = S.sa2_name16)
WHERE sa3_name16 = 'Sydney Inner City'
ORDER BY sa2_name16; """

inner_accomm_df = pd.read_sql(sql, conn)
inner_accomm_df['inner_z_accomm'] = zscore(inner_accomm_df['accomm_per_1000_people'])
inner_accomm_df


# In[13]:


#Separate z-score calculations for Inner sydney
#School
sql = """SELECT sa2_name16 as "suburb", 
COUNT(use_id) as "number_of_schools", 
SUM(age0_4 + age5_9 + age10_14 + age15_19)/1000 AS "youths_in_area_per_1000",
CAST(COUNT(use_id)/(SUM(age0_4 + age5_9 + age10_14 + age15_19)/1000) AS NUMERIC)  AS "schools_per_1000_youths"
FROM sa2 S
LEFT JOIN catchments C on ST_Contains(S.geom, C.geom)
RIGHT JOIN neighbourhoods N ON (N.area_name = S.sa2_name16)
WHERE sa3_name16 = 'Sydney Inner City'
GROUP BY sa2_name16
ORDER BY suburb; """

inner_school_df = pd.read_sql(sql, conn)
inner_school_df['inner_z_school'] = zscore(inner_school_df['schools_per_1000_youths'])
inner_school_df


# The filter in SQL below produced no results, as in the inner city of Syndey, there's no area with 'High Density'

# In[80]:


#Separate z-score calculations for Inner sydney
#CRIME
sql = """
SELECT
sa2_name16 as "suburb",
B.density,
ST_AREA(S.geom) AS "total_area",
SUM(shape_area) AS "sum_of_hotspot_areas",
SUM(shape_area)/ST_AREA(S.geom) AS "sum_of_hotspot_areas_divided_by_total_area"
FROM sa2 S
LEFT JOIN breakenterdwelling B on ST_Contains(S.geom, B.geom)
RIGHT JOIN neighbourhoods N ON (N.area_name = S.sa2_name16)
WHERE sa3_name16 = 'Sydney Inner City'
GROUP BY suburb, S.geom, B.density
ORDER BY suburb; 
"""
inner_crime_df = pd.read_sql(sql, conn)
inner_crime_df[inner_crime_df['density'] != 'High Density']['sum_of_hotspot_areas'] = None
inner_crime_df = inner_crime_df.drop('density', axis =1)
inner_crime_df['inner_z_crime'] = zscore(inner_crime_df['sum_of_hotspot_areas_divided_by_total_area'], nan_policy='omit').fillna(0)
inner_crime_df


# In[15]:


#Separate z-score calculations for Inner sydney
#Health
sql = """SELECT 
sa2_name16 as "suburb",
health_care_and_social_assistance/(N.population/1000) AS "health_per_1000_people"
FROM businessstats B
RIGHT JOIN neighbourhoods N USING (area_name)
LEFT JOIN sa2 S on (B.area_name = S.sa2_name16)
WHERE sa3_name16 = 'Sydney Inner City'
ORDER BY sa2_name16; """

inner_health_df = pd.read_sql(sql, conn)
inner_health_df['inner_z_health'] = zscore(inner_health_df['health_per_1000_people'])
inner_health_df


# In[16]:


#Separate z-score calculations for Inner sydney
#RETAIL
sql = """SELECT 
sa2_name16 as "suburb",
retail_trade/(N.population/1000) AS "retail_per_1000_people"
FROM businessstats B
RIGHT JOIN neighbourhoods N USING (area_name)
LEFT JOIN sa2 S on (B.area_name = S.sa2_name16)
WHERE sa3_name16 = 'Sydney Inner City'
ORDER BY sa2_name16; """

inner_retail_df = pd.read_sql(sql, conn)
inner_retail_df['inner_z_retail'] = zscore(inner_retail_df['retail_per_1000_people'])
inner_retail_df


# In[81]:


#INNER CITHY WITH BINS AND DRINKINGS
z_with_add = pd.merge(inner_school_df,inner_retail_df, on = 'suburb')
z_with_add = pd.merge(z_with_add, inner_accomm_df, on = 'suburb')
z_with_add = pd.merge(z_with_add,inner_crime_df, on = 'suburb')
z_with_add = pd.merge(z_with_add,inner_health_df, on = 'suburb')
z_with_add = pd.merge(z_with_add,bin_df, on = 'suburb')
z_with_add = pd.merge(z_with_add,drinking_df, on = 'suburb')

result = z_with_add.drop(["number_of_schools", "youths_in_area_per_1000", "accomm_per_1000_people", "schools_per_1000_youths", "retail_per_1000_people", "total_area", "sum_of_hotspot_areas", "sum_of_hotspot_areas_divided_by_total_area", "health_per_1000_people", "drinking_fountains_per_1000", "number_of_bins", "pop_per_1000", "bins_per_1000_people"],axis = 1)


result["total_zscore"] = result["inner_z_school"] + result["inner_z_retail"] - result["inner_z_crime"] + result["inner_z_health"] + result["z_bin"] +result["z_drink"]
result["sigmoid"] = 1 / (1 + np.exp(-result["total_zscore"]))
result


# # CORRELATION

# In[82]:


#Correlation between neighbourhood and sigmoid of greater syndey

sql = """
SELECT area_name AS "suburb",
avg_monthly_rent AS "median_rent",
median_annual_household_income AS "median_income"
FROM neighbourhoods N
LEFT JOIN sa2 S ON (N.area_name = S.sa2_name16)
WHERE S.gcc_name16 = 'Greater Sydney'
ORDER BY suburb;
"""

median_df = pd.read_sql(sql, conn)
median_df


# In[83]:


corr = pd.merge(final_z,median_df, on = "suburb")
data = corr.drop(["z_school","z_accomm","z_retail","z_crime","z_health","zscore"], axis =1)
# correlation between sigmoid and median rent
print(data["sigmoid"].corr(data["median_rent"]))
# correlation between sigmoid and median income
print(data["sigmoid"].corr(data["median_income"]))


# In[84]:


from geopandas import GeoDataFrame
SA2 = gpd.read_file("SA2_2016_AUST/SA2_2016_AUST.shp")
SA2 = pd.DataFrame(SA2)
SA2.rename(columns = {'SA2_NAME16':'suburb'}, inplace = True)
z_map = pd.merge(final_z, SA2, on = 'suburb')
z_map = GeoDataFrame(z_map)

#Identifying Greater sydney are from SA2
greater_syd_map = SA2.loc[SA2['GCC_NAME16'] == 'Greater Sydney']
greater_syd_map = GeoDataFrame(greater_syd_map)


# In[85]:


f, ax = plt.subplots(1, figsize=(12, 12))
ax.set_title('Greater Syndey Suburbs and Sigmoid')

greater_syd_map.plot(ax=ax, facecolor='lightgray', linewidth=0.7, edgecolor='white')

z_map.plot(ax=ax, column="sigmoid", cmap="summer", linewidth=0.7, edgecolor='white')
color_bar = plt.cm.ScalarMappable(cmap="summer", norm=plt.Normalize(vmin=0, vmax=1))
cbar = f.colorbar(color_bar)
ax.set_axis_off()


# In[23]:


z_innercity_map = pd.merge(result, SA2, on = 'suburb')
z_innercity_map = GeoDataFrame(z_innercity_map)

f, ax = plt.subplots(1, figsize=(12, 12))

greater_syd_map.plot(ax=ax, facecolor='lightgray', linewidth=0.7, edgecolor='gray')
z_innercity_map.plot(ax=ax, column="sigmoid", cmap="summer", linewidth=0.7, edgecolor='gray')

color_bar = plt.cm.ScalarMappable(cmap="summer", norm=plt.Normalize(vmin=0, vmax=1))
cbar = f.colorbar(color_bar)

ax.set_xlim(151.1, 151.3)
ax.set_ylim(-34, -33.8)
ax.set_axis_off()


# In[86]:


suburb = result["suburb"]
score = result["sigmoid"]
plt.barh(suburb,score)
plt.xlabel("Sigmoid score")
plt.ylabel("Suburb")
plt.title("Livability score for each Suburb")
plt.savefig("suburb and zscore.png")


# In[87]:


stack_result = result.drop(['total_zscore', 'sigmoid'], axis = 1)
plt.rcParams["figure.figsize"] = (15,6)
stack_result.plot(x='suburb',
        kind='bar',
        stacked=False,
        title='Grouped Bar Graph with for suburbs and zscore',
                 rot = 45)
plt.xlabel('Suburbs')
plt.ylabel('Z-scores')
plt.grid(alpha = 0.3)


# In[ ]:


conn.close()
db.dispose()

