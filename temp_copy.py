import os, glob, shutil
all_cases = r'C:\Users\tians\OneDrive\meshes'
cases = glob.glob('n*', root_dir=r'P:\20230427')

cases.remove('n0042')

segs = ['di','diL','diR','le']

for c in cases:
    shutil.copy(os.path.join(all_cases, c, 'hexmesh_open.inp'), os.path.join(r'P:\20230427', c))
    for s in segs:
        shutil.copy(os.path.join(all_cases, c, 'pre_'+s+'.stl'), os.path.join(r'P:\20230427', c))
        if os.path.exists(os.path.join(all_cases, c, 'pre_'+s+'.tfm')):
            shutil.copy(os.path.join(all_cases, c, 'pre_'+s+'.stl'), os.path.join(r'P:\20230427', c))
    s = 'gen'
    if os.path.exists(os.path.join(all_cases, c, 'pre_'+s+'.stl')):
        shutil.copy(os.path.join(all_cases, c, 'pre_'+s+'.stl'), os.path.join(r'P:\20230427', c))
        if os.path.exists(os.path.join(all_cases, c, 'pre_'+s+'.tfm')):
            shutil.copy(os.path.join(all_cases, c, 'pre_'+s+'.stl'), os.path.join(r'P:\20230427', c))
    s = 'hex_skin.stl'
    if os.path.exists(os.path.join(all_cases, c, s)):
        shutil.copy(os.path.join(all_cases, c, s), os.path.join(r'P:\20230427', c))
        print(c, 'hex skin')

