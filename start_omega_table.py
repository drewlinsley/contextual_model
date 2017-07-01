#! /usr/bin/env python
from ops.db_utils import generate_omega_combos
from ops.parameter_defaults import PaperDefaults

defaults = PaperDefaults()
print 'Generating omega hyperparameter combos'
generate_omega_combos()
