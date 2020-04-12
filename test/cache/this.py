import importlib
import unittest
import sys
import os


__usage__='''
%prog      # Searches CWD
%prog DIR   
'''

# sys.path.append("./test/")
# sys.path.append("./test/cahce/")
# import one_level_cache

def construct_modlist(mod_strings):
    mod_list = []
    for mod_str in mod_strings:
        spec = importlib.util.find_spec(mod_str)
        #spec = importlib.util.spec_from_file_location("one_level_cache","/home/gauenk/Documents/experiments/label_smoothing_sim/test/cache/one_level_cache.py")
        #print(spec)
        foo = importlib.util.module_from_spec(spec)
        #print(test_modules[0] in sys.modules)
        spec.loader.exec_module(foo)
        mod_list.append(foo)
    return mod_list
    
    
if __name__=='__main__':
    if len(sys.argv)>1:
        unit_dir=sys.argv[1]
    else:
        unit_dir='.'

    test_modules=[filename.replace('.py','') for filename in os.listdir(unit_dir)
                  if filename.endswith('.py') and filename.startswith('one_')]
    # help from https://bugs.python.org/issue37521
    modlist = construct_modlist(test_modules)
    suite = unittest.TestSuite()
    for mod in modlist:
        suite.addTest(unittest.TestLoader().loadTestsFromModule(mod))
    unittest.TextTestRunner(verbosity=2).run(suite)

