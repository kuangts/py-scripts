import numpy as np
import re
test="""<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="3.0">
<Mesh>
<Nodes name="MergedObject1">
<node id="1"> 2.0774550e+02, 1.0365740e+02, 1.6971720e+02</node>
<node id="2"> 2.0663660e+02, 1.0408680e+02, 1.6980840e+02</node>
<node id="3"> 2.0695580e+02, 1.0208330e+02, 1.6963740e+02</node>
<node id="4"> 2.0581860e+02, 1.0251230e+02, 1.6973180e+02</node>
<node id="5"> 2.0573110e+02, 9.9725740e+01, 1.6949440e+02</node>
<node id="6"> 2.0458960e+02, 1.0017670e+02, 1.6959000e+02</node>
<node id="7"> 2.0437770e+02, 9.7242870e+01, 1.6934870e+02</node>
<node id="8"> 2.0327330e+02, 9.7721830e+01, 1.6944430e+02</node>
<node id="15"> 1.9882700e+02, 8.7910370e+01, 1.6866880e+02</node>
<node id="16"> 1.9780970e+02, 8.8415320e+01, 1.6871470e+02</node>
<node id="17"> 1.9734480e+02, 8.5595230e+01, 1.6847260e+02</node>
<node id="95428"> 1.5373241e+02, 5.0735256e+01, 4.0610146e+01</node>
<node id="95429"> 1.5468788e+02, 5.1838631e+01, 4.0730804e+01</node>
<node id="95430"> 1.5401247e+02, 5.1399185e+01, 4.1982368e+01</node>
<node id="95431"> 1.4911003e+02, 4.7851952e+01, 2.1499763e+01</node>
<node id="95432"> 1.5029842e+02, 4.8708282e+01, 2.1588572e+01</node>
<node id="95433"> 1.5084950e+02, 4.7481819e+01, 2.1867037e+01</node>
</Nodes>
<Elements type="hex8" name="Part1">
<elem id="1"> 2, 1, 3, 4, 536, 535, 537, 538</elem>
<elem id="2"> 4, 3, 5, 6, 538, 537, 539, 540</elem>
<elem id="3"> 6, 5, 7, 8, 540, 539, 541, 542</elem>
<elem id="4"> 8, 7, 9, 10, 542, 541, 543, 544</elem>
<elem id="5"> 10, 9, 11, 12, 544, 543, 545, 546</elem>
<elem id="6"> 12, 11, 13, 14, 546, 545, 547, 548</elem>
<elem id="7"> 14, 13, 15, 16, 548, 547, 549, 550</elem>
<elem id="38278"> 46366, 46455, 46456, 46367, 46900, 46989, 46990, 46901</elem>
<elem id="38279"> 46367, 46456, 46457, 46368, 46901, 46990, 46991, 46902</elem>
<elem id="38280"> 46368, 46457, 46458, 46369, 46902, 46991, 46992, 46903</elem>
</Elements>
<Elements type="tri3" name="Part2">
<elem id="2"> 47167, 47168, 47169</elem>
<elem id="3"> 47170, 47171, 47172</elem>
<elem id="5437"> 95428, 95429, 95430</elem>
<elem id="5438"> 95431, 95432, 95433</elem>
</Elements>
<NodeSet name="FixedDisplacement1">
<n id="38"/>
<n id="40"/>
<n id="42"/>
<n id="44"/>
<n id="46"/>
<n id="48"/>
<n id="50"/>
<n id="24602"/>
<n id="24604"/>
<n id="24694"/>
<n id="24696"/>
<n id="24700"/>
<n id="25232"/>
</NodeSet>
<NodeSet name="PrescribedDisplacement1">
<n id="41702"/>
<n id="42772"/>
<n id="42774"/>
<n id="43308"/>
<n id="43310"/>
<n id="43374"/>
<n id="43376"/>
<n id="43378"/>
<n id="43836"/>
<n id="43840"/>
<n id="43842"/>
<n id="43844"/>
</NodeSet>
<NodeSet name="Nodeset01">
<n id="1"/>
</NodeSet>
<NodeSet name="Nodeset01">
<n id="1"/>
</NodeSet>
<Surface name="SlidingInnerSurface1">
<quad4 id="1">14484,14482,15016,15018</quad4>
<quad4 id="2">14486,14484,15018,15020</quad4>
<quad4 id="3">14488,14486,15020,15022</quad4>
<quad4 id="4">14490,14488,15022,15024</quad4>
<quad4 id="5">14492,14490,15024,15026</quad4>
<quad4 id="6">14494,14492,15026,15028</quad4>
<quad4 id="1490">30046,30044,30578,30580</quad4>
<quad4 id="1491">30048,30046,30580,30582</quad4>
<quad4 id="1492">30050,30048,30582,30584</quad4>
<quad4 id="1493">30052,30050,30584,30586</quad4>
<quad4 id="1494">30054,30052,30586,30588</quad4>
</Surface>
<Surface name="Sliding_le_Surface">
<tri3 id="1">47167,47168,47169</tri3>
<tri3 id="2">47170,47171,47172</tri3>
<tri3 id="3">47173,47174,47175</tri3>
<tri3 id="4">47176,47177,47178</tri3>
<tri3 id="5">47179,47180,47181</tri3>
<tri3 id="6">47182,47183,47184</tri3>
<tri3 id="2623">55033,55034,55035</tri3>
<tri3 id="2624">55036,55037,55038</tri3>
<tri3 id="2625">55039,55040,55041</tri3>
<tri3 id="2626">55042,55043,55044</tri3>
<tri3 id="2627">55045,55046,55047</tri3>
<tri3 id="2628">55048,55049,55050</tri3>
</Surface>
<Surface name="SlidingInnerSurface2">
<quad4 id="1">14042,14040,14574,14576</quad4>
<quad4 id="2">14044,14042,14576,14578</quad4>
<quad4 id="3">14046,14044,14578,14580</quad4>
<quad4 id="4">14574,14572,15106,15108</quad4>
<quad4 id="5">14576,14574,15108,15110</quad4>
<quad4 id="6">14578,14576,15110,15112</quad4>
<quad4 id="111">47009,47007,47008,47010</quad4>
<quad4 id="112">47007,47005,47006,47008</quad4>
<quad4 id="113">47005,47003,47004,47006</quad4>
<quad4 id="114">47003,47001,47002,47004</quad4>
<quad4 id="115">47001,46999,47000,47002</quad4>
</Surface>
<SurfacePair name="SlidingSurfacePair1">
<primary>SlidingInnerSurface1</primary>
<secondary>Sliding_le_Surface</secondary>
</SurfacePair>
<SurfacePair name="SlidingSurfacePair2">
<primary>SlidingInnerSurface2</primary>
<secondary>Sliding_diR_Surface</secondary>
</SurfacePair>
<SurfacePair name="SlidingSurfacePair3">
<primary>SlidingInnerSurface3</primary>
<secondary>Sliding_diL_Surface</secondary>
</SurfacePair>
<SurfacePair name="SlidingSurfacePair4">
<primary>SlidingInnerSurface4</primary>
<secondary>Sliding_di_Surface</secondary>
</SurfacePair>
<SurfacePair name="SlidingSurfacePair5">
<primary>SlidingUpperLipSurface</primary>
<secondary>SlidingLowerLipSurface</secondary>
</SurfacePair>
</Mesh>
<MeshDomains>
<SolidDomain name="Part1" mat="Material1"/>
<SolidDomain name="Part2" mat="Material2"/>
<SolidDomain name="Part3" mat="Material3"/>
<SolidDomain name="Part4" mat="Material4"/>
<SolidDomain name="Part5" mat="Material5"/>
</MeshDomains>
<MeshData>
<NodeData name="values1" node_set="PrescribedDisplacement1">
<node lid="1"> -4.726013</node>
<node lid="2"> -4.383588</node>
<node lid="3"> -4.690824</node>
<node lid="4"> -4.781093</node>
<node lid="5"> -4.951978</node>
<node lid="16"> -4.617146</node>
<node lid="17"> -4.523492</node>
<node lid="18"> -4.919374</node>
<node lid="19"> -5.093369</node>
<node lid="20"> -5.175161</node>
<node lid="21"> -5.254805</node>
</NodeData>
<NodeData name="values2" node_set="PrescribedDisplacement1">
<node lid="1"> 5.166113</node>
<node lid="2"> 7.008126</node>
<node lid="14"> 5.761136</node>
<node lid="15"> 7.313587</node>
<node lid="16"> 7.325249</node>
<node lid="17"> 7.330724</node>
<node lid="18"> 5.562288</node>
<node lid="19"> 5.720830</node>
<node lid="20"> 5.802657</node>
<node lid="21"> 5.882540</node>
</NodeData>
<NodeData name="values3" node_set="PrescribedDisplacement1">
<node lid="1"> 1.487020</node>
<node lid="2"> 6.317307</node>
<node lid="3"> 1.750159</node>
<node lid="4"> 1.641219</node>
<node lid="19"> 1.475712</node>
<node lid="20"> 1.388037</node>
<node lid="21"> 1.303084</node>
</NodeData>
</MeshData>
</febio_spec>"""






first = re.search(r"<Nodes name[\S ]*\s+(.*)<\/Nodes>", test, flags=re.MULTILINE|re.DOTALL)
print(r"<Nodes name[\S ]*\s+(.*)<\/Nodes>")
temp = first.groups()[0]
second = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)+", temp, flags=re.MULTILINE|re.DOTALL)
nodes=np.array(list(map(float, second)))
nodeid=re.findall(r"(?<=\")\d+(?=\")", temp,flags=re.MULTILINE|re.DOTALL)
nodes=np.reshape(nodes,(-1,3))
nodedict=dict(zip(nodeid,nodes))
print(nodedict)
print(nodedict["1"][2])

first = re.search(r"<Elements[\S ]*\n(.*)<Elements(?!<\/Elements> type)", test,flags=re.MULTILINE|re.DOTALL)
temp=first.groups()[0]
second = re.findall(r"\d+", temp,flags=re.MULTILINE|re.DOTALL)
ele=np.array(list(map(int, second)))
ele=np.reshape(ele,(-1,9))
eleid=ele[:, 0]
ele=np.delete(ele,0,1)


eledict=dict(zip(eleid,ele))
print(eledict)
print(eledict[1][2])