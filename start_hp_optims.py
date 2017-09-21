#! /usr/bin/env python
from ops.db_utils import init_db, generate_combos, create_and_execute_daemons, prepare_settings
from ops.parameter_defaults import PaperDefaults

defaults = PaperDefaults()
print 'Initializing database'
# init_db()
print 'Generating hyperparameter combos'
generate_combos()
print 'Preparing settings table'
prepare_settings(defaults.db_problem_columns)
print 'Running damons on gpus %s' % (defaults.gpu_processes)
#create_and_execute_daemons(defaults.gpu_processes) #ids of gpus. enter a new one for each daemon you want to run.
#create_and_execute_daemons([0]) #ids of gpus. enter a new one for each daemon you want to run.
#create_and_execute_daemons([2]) #ids of gpus. enter a new one for each daemon you want to run.
#create_and_execute_daemons([3]) #ids of gpus. enter a new one for each daemon you want to run.
