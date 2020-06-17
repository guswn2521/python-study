# list, dict 를 다양한 파일로 저장
# @date 2019.10.30
# @author k2moon72@gmail.com


try:
    import pandas as pd
    import pickle
    import sqlite3
    from tinydb import TinyDB, Query
    import time
    import csv
    import json
    
except Exception as e:
    str_e = str(e)
    print(e)
    print('pip install ' + str_e.split('\'')[1])

     
save_date = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
file_tmpl_n = '{}.{}'
file_tmpl_t = '{}_{}.{}'
file_encoding = 'utf-8'

# _type : list, dict
def to_csv(data, file_name = 'temp', headers = False, save_time = True): 
    try:
        if save_time:
            file_name = file_tmpl_t.format(file_name, save_date,'csv')
        else:
            file_name = file_tmpl_n.format(file_name,'csv')
        
        data_type = 'list'
        if type(data) is list:
            if type(data[0]) is dict:
                data_type = 'dict'
        elif type(data) is dict:
            data_type = 'dict'
          
        if data_type == 'dict':

            with open(file_name, 'w', encoding=file_encoding, newline='') as f:
                header = data[0].keys()
                f_csv = csv.DictWriter(f, header)
                if headers:
                    f_csv.writeheader()

                f_csv.writerows(data)
        elif data_type == 'list':

            with open(file_name, 'w', encoding=file_encoding, newline='') as f:
                f_csv = csv.writer(f)
                header = data[0]
                if headers:
                    f_csv.writerow(header)
                f_csv.writerows(data)
        else:
            print('data type is not list or dict')
                
    except Exception as e:
        print(e)
        
def to_json(data, file_name = 'temp', save_time = True):
    try:
        data_type = 'list'
        
        if save_time:
            file_name = file_tmpl_t.format(file_name, save_date,'json')
        else:
            file_name = file_tmpl_n.format(file_name,'json')
        
        if type(data) is list:
            if not type(data[0]) is dict:
                 data_type = None
                
        elif type(data) is dict:
            data_type = 'dict'
        
        if data_type is 'dict':

            with open(file_name, 'w', encoding=file_encoding) as f:
                w_data = json.dumps(data, indent = 4, sort_keys = True, ensure_ascii = False)
                f.write(w_data)
        elif data_type == 'list':
            with open(file_name, 'a', encoding=file_encoding) as f:
                for proc_data in data:
                    
                    w_data = json.dumps(proc_data, ensure_ascii = False) + '\n'
                    f.write(w_data)
        else:
            print('data type is not list or dict')

    except Exception as e:
        print(e)        

def to_pickle(data, file_name = 'temp', save_time = True): 
    try:
        if save_time:
            file_name = file_tmpl_t.format(file_name, save_date,'pickle')
        else:
            file_name = file_tmpl_n.format(file_name,'pickle')
    
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(e)

def to_sqlitedb(data, dbname = 'temp', tbname = 'temp' ):
    try:
        
        con = sqlite3.connect("{}.sqlite".format(dbname))

        with con:
            #isinstance(data, list)
            if type(data) is list:
                df = pd.DataFrame(data)
                df.to_sql(tbname, con, if_exists="replace", index=False) # if_exists="append"
            else:
                print('data type is not list')
    except Exception as e:
        print(e)
        
def to_tinydb(data, dbname = 'temp', tbname = 'temp'): 
    try:
        
        db = TinyDB('{}.tinydb'.format(dbname))
        if tbname != 'temp':
            tb = db.table(tbname)

        #isinstance(data, list)
        if type(data) is list:
            for row in data:
                if type(row) is dict:
                    tb.insert(row)
                else:
                    print('data type is not dict')
                    continue
        elif type(data) is dict:
            tb.insert(data)
        else:
            print('data type is not list or dict')
            return

    except Exception as e:
        print(e)
    finally:
        db.close()
            