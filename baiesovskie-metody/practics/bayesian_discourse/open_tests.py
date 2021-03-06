#!/usr/bin/python3.4

# inport your code and choose your variant number
import prototypes as code
var_number = 3

import time
import numpy as np

# functions to test
ff = list()
ff.append(['pa','pb','pc','pd','pc_a','pc_b','pc_d','pc_ab','pc_abd']) # variant 1
ff.append(['pa','pb','pc','pd','pc_a','pc_b','pb_a','pb_d','pb_ad'])   # variant 2
ff.append(['pa','pb','pc','pd','pb_d','pb_ad'])                        # variant 3
ff.append(['generate'])                                                # variant 3

# testing params: variants 1,2 at the beginning, variant 3 later
max_time = 1
models = [1,2]
params = {'amin': 75, 'amax': 90, 'bmin': 500, 'bmax': 600,
              'p1': 0.1, 'p2': 0.01, 'p3': 0.3} 

a=np.arange(80,82)
b=np.arange(500,503)
d=np.arange(400,404)

alen = params['amax'] - params['amin'] + 1
blen = params['bmax'] - params['bmin'] + 1
clen = params['amax'] + params['bmax'] + 1
dlen = (params['amax'] + params['bmax']) * 2 + 1
  
args_long = {'pa':[],'pb':[],'pc':[],'pd':[],'pc_a':[a],'pc_b':[b],
             'pc_d':[d],'pc_ab':[a,b],'pc_abd':[a,b,d],'pb_a':[a],
             'pb_d':[d],'pb_ad':[a,d]}
args_short = {'pa':[],'pb':[],'pc':[],'pd':[],'pc_a':[a[:1]],'pc_b':[b[:1]],
              'pc_d':[d[:1]],'pc_ab':[a[:1],b[:1]],'pc_abd':[a[:1],b[:1],d[:1]], 'pb_a':[a[:1]],
             'pb_d':[d[:1]],'pb_ad':[a[:1],d[:1]]}
size_long = {'pa':(alen,),'pb':(blen,),'pc':(clen,),'pd':(dlen,),'pc_a':(clen,a.size),'pc_b':(clen,b.size),
             'pc_d':(clen,d.size),'pc_ab':(clen,a.size,b.size),'pc_abd':(clen,a.size,b.size,d.size), 'pb_a':(blen,a.size),
             'pb_d':(blen,d.size),'pb_ad':(blen,a.size,d.size)}
size_short = {'pa':(alen,),'pb':(blen,),'pc':(clen,),'pd':(dlen,),'pc_a':(clen,1),'pc_b':(clen,1),
             'pc_d':(clen,1),'pc_ab':(clen,1,1),'pc_abd':(clen,1,1,1), 'pb_a':(blen,1),
             'pb_d':(blen,1),'pb_ad':(blen,1,1)}

if var_number == 3:
    N = 50
    models = [3,4]
    d = np.tile(np.arange(40,44),[7,1])
    args_long.update({'pb_d':[d],'pb_ad':[a,d],'generate':[N,a,b]})
    args_short.update({'pb_d':[d[:1,:]],'pb_ad':[a[:1],d[:1,:]],'generate':[N,a[:1],b[:1]]})
    size_long.update({'pb_d':(blen,d.shape[0]),'pb_ad':(blen,a.size,d.shape[0]),'generate':(N,a.size,b.size)})
    size_short.update({'generate':(N,1,1)})

# testing functions
def test_distribution(f,model,arg_long,arg_short,size_long,size_short):
    # existence
    if not hasattr(code, f):
        raise Exception('Function %s for model %d is not found.' % (f,model))
    # run with vector inputs
    prob, var = getattr(code, f)(*arg_long, params,model)
    if var.shape != (size_long[0],):
        raise Exception('Function %s for model %d returns distribution support of the wrong size: %s instead of %s.' % (f,model,str(var.shape),str((size_long[0],))))
    if prob.shape != size_long:
        raise Exception('Function %s for model %d returns distribution probabilities of the wrong size: %s instead of %s.' % (f,model,str(prob.shape),str(size_long)))
    if np.sum(np.isnan(prob)) != 0 or np.sum(np.isnan(var)) != 0:
        raise Exception('Function %s for model %d returns NaNs.' % (f,model))
    # run with scalar inputs + test time
    t_start = time.clock(); 
    prob, var = getattr(code, f)(*arg_short, params,model)
    t = time.clock() - t_start
    if prob.shape != size_short:
        raise Exception('Function %s for model %d returns distribution probabilities of the wrong size: %s instead of %s.' % (f,model,str(prob.shape),str(size_short)))
    if t > max_time:
        print('Warning: function %s for model %d is too slow: %.4f seconds.' % (f, model, t))
    print('%s, model %d: \t ok; \t time: %.4f seconds' % (f,model,t))
    return

def test_generate(f,model,arg_long,arg_short,size_long,size_short):
    # existence
    if not hasattr(code, f):
        raise Exception('Function %s for model %d is not found.' % (f,model))
    # run with vector inputs
    gen= getattr(code, f)(*arg_long, params,model)
    if gen.shape != size_long:
        raise Exception('Function %s for model %d returns distribution probabilities of the wrong size: %s instead of %s.' % (f,model,str(gen.shape),str(size_long)))
    if np.sum(np.isnan(gen)) != 0:
        raise Exception('Function %s for model %d returns NaNs.' % (f,model))
    # run with scalar inputs + test time
    t_start = time.clock(); 
    gen = getattr(code, f)(*arg_short, params,model)
    t = time.clock() - t_start
    if gen.shape != size_short:
        raise Exception('Function %s for model %d returns distribution probabilities of the wrong size: %s instead of %s.' % (f,model,str(gen.shape),str(size_short)))
    if t > max_time:
        print('Warning: function %s for model %d is too slow: %.4f seconds.' % (f, model, t))
    print('%s, model %d: \t ok; \t time: %.4f seconds' % (f,model,t))
    return

# run tests
for f in ff[var_number-1]:
    test_distribution(f,models[0],args_long[f],args_short[f],size_long[f],size_short[f])
    test_distribution(f,models[1],args_long[f],args_short[f],size_long[f],size_short[f])
    
if var_number == 3:
    for f in ff[-1]:
        test_generate(f,models[0],args_long[f],args_short[f],size_long[f],size_short[f])
        test_generate(f,models[1],args_long[f],args_short[f],size_long[f],size_short[f])