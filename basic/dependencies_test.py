#! /usr/bin/env python

import os, sys, re, pkg_resources

req = os.path.join(os.path.dirname(__file__), 'requirements.txt')
with open(req,'r') as f:
    dependencies = re.split('[\r\n]+', f.read().strip())
try:
    pkg_resources.require(dependencies)
except Exception as e:
    sys.exit(e)


