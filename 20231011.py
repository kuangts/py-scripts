import os
os.chdir(r'P:\intern')
case_name = 'kx01'
os.chdir(case_name)

'''

int CMatrix:: Get_transfrom_element(int m_select,int mRow,int mode, double*  matrix)
{
	vtkMatrix4x4* Transform_M = vtkMatrix4x4::New();
	if (mode == 0 )
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Object_Transform->GetMatrix());
	}
	else if (mode == 1)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Center_Transform->GetMatrix());
	}
	else if (mode == 2)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Object_Transform_NHP_COPY->GetMatrix());
	}
	else if (mode == 3)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Center_Transform_NHP_COPY->GetMatrix());
	}
	else if (mode == 4)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Object_Transform_final_COPY->GetMatrix());
	}
	else if (mode == 5)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Center_Transform_final_COPY->GetMatrix());
	}
	else if (mode == 6)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Object_Transform_termporal_COPY->GetMatrix());
	}
	else if (mode == 7)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Center_Transform_termporal_COPY->GetMatrix());
	}

	else if (mode == 8)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Object_Overcorrection_COPY->GetMatrix());
	}
	else if (mode == 9)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Center_Overcorrection_COPY->GetMatrix());
	}
	else if (mode == 10)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Object_Transform_Original_COPY->GetMatrix());
	}
	else if (mode == 11)
	{
		Transform_M->DeepCopy(ThreeD_objects[m_select].Center_Transform_Original_COPY->GetMatrix());
	}

    '''


import subprocess
import numpy as np
cass = os.path.realpath(rf'..\..\intern_cass\{case_name}.cass')
subprocess.call([r'C:\Program Files\WinRAR\UnRAR.exe', 'x', '-o+', cass, 'nhp.bin', 'Three_Data_Info.bin'])
with open('nhp.bin') as f:
    nhp = np.array(f.read().strip(',').split(','), dtype=float).reshape(4,4)
    print('nhp:', nhp, sep='\n')

with open('Three_Data_Info.bin') as f:
    info_str = f.read().strip(';').split(';')
    num_obj = int(info_str[0])
    info_str = info_str[1:]
    info = []
    for i in range(len(info_str)):
        info_i = info_str[i].split(',')
        t = info_i[41:201] + info_i[203:]
        t = np.array(t, dtype=float).reshape(-1,4,4)
        info.append(dict(name=info_i[0], t=t))

from tools.polydata import *
from tools.ui import Window

print(*[f'{i}: {x["name"]}' for i,x in enumerate(info)], sep='\n')
n = int(input('which object: '))
os.makedirs('stl-extracted', exist_ok=True)
subprocess.call([r'C:\Program Files\WinRAR\UnRAR.exe', 'x', '-o+', cass, f'{n}.stl', 'stl-extracted\\'])
extracted = polydata_from_stl(rf'stl-extracted\{n}.stl')

# extracted = transform_polydata(extracted, info[n]['t'][0])
extracted = transform_polydata(extracted, info[n]['t'][4])
extracted = transform_polydata(extracted, info[n]['t'][2])
pre = polydata_from_stl(rf'.\stl-pre\{info[n]["name"]}.stl')
post = polydata_from_stl(rf'.\stl-post\{info[n]["name"]}.stl')
w = Window()
# w.add_polydata(pre)
w.add_polydata(post)
w.add_polydata(extracted)
w.initialize()
w.start()


# os.makedirs('stl-extracted', exist_ok=True)
# n = [x['name'] for x in info].index('Mandible')
# subprocess.call([r'C:\Program Files\WinRAR\UnRAR.exe', 'x', '-o+', cass, f'{n}.stl', 'stl-extracted\\'])
# extracted = polydata_from_stl(rf'stl-extracted\{n}.stl')
# extracted_t = transform_polydata(extracted, nhp)
# pre = polydata_from_stl(f'{info[n]["name"]}.stl')
# w = Window()
# w.add_polydata(pre)
# # w.add_polydata(extracted)
# w.add_polydata(extracted_t)
# w.initialize()
# w.start()