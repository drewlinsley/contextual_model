#! /usr/bin/env python
import smtplib, time, traceback, sys
from ops.db_utils import claim_problem, get_row_from_db, restore_problem

def stopWatch(value):
    '''From seconds to Days;Hours:Minutes;Seconds'''

    valueD = (((value/365)/24)/60)
    Days = int (valueD)

    valueH = (valueD-Days)*365
    Hours = int(valueH)

    valueM = (valueH - Hours)*24
    Minutes = int(valueM)

    valueS = (valueM - Minutes)*60
    Seconds = int(valueS)

    return '%s:%s:%s:%s' % (Days,Hours,Minutes,Seconds)

def send_notification_email(elapsed):
    gmail_user = 'clicktionary.ai@gmail.com'  
    gmail_password = 'serrelab'
    to = ['drewlinsley@gmail.com']  
    subject = 'Finished contextual hp optim'  
    body = 'Took %s' % (elapsed)

    email_text = """\  
    From: %s  
    To: %s  
    Subject: %s
    %s
    """ % (gmail_user, ", ".join(to), subject, body)
    try:  
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, to, email_text)
        server.close()
        print 'Sent email because process finished'
    except:  
        raise Exception('Could not send email')

def run_script(problem,hps):
    if problem == 'f3a':
        from db_optimization_scripts.db_fig_3a import run
    elif problem == 'f3b':
        from db_optimization_scripts.db_fig_3b import run
    elif problem == 'f4':
        from db_optimization_scripts.db_fig_4 import run
    elif problem == 'f5':
        from db_optimization_scripts.db_fig_5 import run
    elif problem == 'f7':
        from db_optimization_scripts.db_fig_7 import run
    elif problem == 'bw':
        from db_optimization_scripts.db_fig_bw import run
    elif problem == 'tbp':
        from db_optimization_scripts.db_fig_tbp import run
    elif problem == 'tbtcso':
        from db_optimization_scripts.db_fig_tbtcso import run
    else:
        raise Exception('Cannot understand specified problem')
    run(hps)

def run_daemon():
    get_new_problem = True
    start_time = time.time()
    try: 
        while 1:
            if get_new_problem:
                problem = claim_problem()
                get_new_problem = False
            if problem is not None:
                data = get_row_from_db(problem['fig_names'])
                data['current_figure'] = problem['fig_names']
                run_script(problem['fig_names'],data)
                if data is None:
                    get_new_problem = True

            else:#Returning empty dictionaries means we are finished.
                end_time = time.time()
                send_notification_email(stopWatch(end_time - start_time))
                return

    except:
        restore_problem(problem['fig_names'])
        traceback.print_exc(file=sys.stdout)

if __name__ == '__main__':
    run_daemon()