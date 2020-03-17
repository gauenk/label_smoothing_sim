import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)

# Add libs to pythonpath
this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir,'..','lib')
sys.path.append(lib_path)

python_module_path = osp.join(this_dir,'..','lib/python_module')
sys.path.append(python_module_path)


