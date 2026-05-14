import pandas as pd
import os 
from sqlalchemy import create_engine
import logging
import time
logging.basicConfig(
    filename='logs/data_processing.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a',
    level=logging.DEBUG)
engine=create_engine('sqlite:///inventory.db')
def ingest_db(df,table_name):
    logging.info(f"Ingesting data into table {table_name}")
    df.to_sql(table_name,con=engine,if_exists='replace',index=False,chunksize=5000)
    logging.info(f"Data ingested into table {table_name} successfully")
def load_rawdata():
 start=time.time()
 logging.info("Starting data loading process")
 for file in os.listdir('data'):
    if '.csv' in file:
        df=pd.read_csv(f'data/{file}')
        logging.info(f"Loading data from {file} in Db ")
        print(df.shape)
        ingest_db(df,file.split('.')[0])
 end=time.time()
 logging.info(f"Data loading process completed in {(end-start)/60} minutes")

if __name__=="__main__":
    load_rawdata()
