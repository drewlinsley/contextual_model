import psycopg2
import os
import numpy as np
import psycopg2.extras
from ops.parameter_defaults import PaperDefaults
from ops.sampling import grab_sampler

"""At some point turn this into a class instead of these piecemeal functions"""

defaults = PaperDefaults()

def python_postgresql():
    connect_str = "dbname='context' user='context' host='localhost' " + \
              "password='serrelab'"
    return connect_str

def open_db(use_dict=False):
    conn = psycopg2.connect(python_postgresql())
    if use_dict:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    else:
        cur = conn.cursor()
    return conn,cur

def close_db(cur,conn):
    cur.close()
    conn.close()

def generate_combos():
    ofun = grab_sampler(defaults)
    add_to_database(defaults,ofun)

def prepare_settings(fig_names):
    #Connect to database
    conn,cur = open_db()
    for fn in fig_names:
        cur.execute("INSERT INTO status (fig_names) VALUES (%s)", (fn,))

    #Finalize and close connections
    conn.commit()
    close_db(cur,conn)

def init_db():
    conn,cur = open_db()
    db_schema = open(defaults.db_schema).read().splitlines()
    for s in db_schema:
        t = s.strip()
        if len(t):
            cur.execute(t)
    # Finalize and close connections
    conn.commit()
    close_db(cur,conn)

def add_to_database(parameters,ofun,table_name=defaults.table_name):

    #Connect to database
    conn,cur = open_db()
    for l in parameters.lesions:
        for idx in range(parameters.iterations):
            random_parameters = ofun(parameters)._DEFAULT_PARAMETERS
            cur.execute("INSERT INTO %s (lesions,alpha,beta,mu,nu,gamma,delta) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (table_name,l,random_parameters['alpha'],random_parameters['beta'],
                    random_parameters['mu'],random_parameters['nu'],
                    random_parameters['gamma'],random_parameters['delta']))

    #Finalize and close connections
    conn.commit()
    close_db(cur,conn)

def gather_data(lesion,table_name=defaults.table_name):
    conn,cur = open_db(use_dict=True)
    cur.execute("SELECT (alpha,beta,mu,nu,gamma,delta,_id) FROM %s WHERE lesions='%s'" % (table_name, lesion))
    output = cur.fetchone()
    close_db(cur,conn)
    return output

def get_all_lesion_data(lesion,table_name=defaults.table_name):
    conn,cur = open_db(use_dict=True)
    cur.execute("SELECT * FROM %s WHERE lesions='%s'" % (table_name, lesion))
    output = cur.fetchall()
    close_db(cur,conn)
    return output

def get_row_from_db(fn,table_name=defaults.table_name):
    conn,cur = open_db(use_dict=True)
    cur.execute("SELECT * FROM %s WHERE %s IS NULL LIMIT 1" % (table_name, fn))
    row = dict(cur.fetchone())
    return row 

def get_lesion_rows_from_db(lesion,fn,get_one=False,table_name=defaults.table_name):
    conn,cur = open_db(use_dict=True)
    cur.execute("SELECT * FROM %s WHERE lesions='%s' and %s IS NULL" % (table_name,lesion,fn))
    if get_one:
        data = cur.fetchone()
    else:
        data = cur.fetchall()
    return data

def count_sets(lesion,fn,table_name=defaults.table_name):
    conn,cur = open_db(use_dict=True)
    cur.execute("SELECT count(*) FROM %s WHERE lesions='%s' and %s IS NULL" % (table_name,lesion,fn))
    return cur.fetchall()

def claim_problem():
    conn,cur = open_db(use_dict=True)
    cur.execute("SELECT * from status WHERE working=FALSE LIMIT 1") #Find new problem to work on
    row = cur.fetchone()
    if row is not None:
        cur.execute("UPDATE status SET working=TRUE WHERE _id=%s", (row['_id'],)) #and claim it. could do these in a single query.
        conn.commit()
        close_db(cur,conn)
    return row

def restore_problem(fn):
    conn,cur = open_db()
    cur.execute("UPDATE status SET working=FALSE WHERE fig_names='%s'" % fn) #revert column to untouched in case of error.
    conn.commit()
    close_db(cur,conn)

def update_data(params,figure,idx,score,table_name=defaults.table_name):
    if np.isnan(score):
        score = 0
    conn,cur = open_db()
    cur.execute("UPDATE %s SET %s=%s where _id=%s" % (table_name,figure,score,idx))
    conn.commit()
    close_db(cur,conn)

def create_and_execute_daemons(gpu_names):
    for idx in gpu_names:
        os.system('CUDA_VISIBLE_DEVICES=%s python ops/contextual_model_daemon.py' % idx)

# def max_parameter_values() order postgresql select query by zscore