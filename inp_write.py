	def write(self, file):
		with open(file,'w') as f:
			n_format = f'{{:0{int(np.log10(self.N.shape[0]))+1:d}d}},{{:.8f}},{{:.8f}},{{:.8f}}'
			e_format = f'{{:0{int(np.log10(self.E.shape[0]))+1:d}d}},'+','.join([f'{{:0{int(np.log10(self.N.shape[0]))+1:d}d}}']*self.E.shape[1])


			# if file.endswith('.inp'):
			# 	s = '*NODE\n{}\n*END NODE\n*ELEMENT\n{}\n*END ELEMENT\n'
			# 	node = np.hstack((range(node.shape[0]),self.N))
			# 		r'\*.*NODE[\S ]*\s+(.*)\*.*END NODE.*\*.*ELEMENT[\S ]*\s+(.*)\*.*END ELEMENT', 
			# 		f.read(), 
			# 		re.MULTILINE | re.DOTALL)
			# 	try:
			# 		nodes, elems = match.group(1), match.group(2)
			# 	except Exception as e:
			# 		print(e)
			# 		raise ValueError('the file cannot be read for nodes and elements')
			# nodes4 = [node.split(',') for node in nodes.strip().split('\n')]
			# nodes = np.asarray(nodes4, dtype=float)[:,1:]
			# elems9 = [elem.split(',') for elem in elems.strip().split('\n')]
			# elems = np.asarray(elems9, dtype=int)[:,1:]-1

			# elif file.endswith('.feb'):
			# 	Nodes, Elements = re.search(r"<Nodes[\S ]*>\s*(<.*>)\s*</Nodes>.*<Elements[\S ]*\"hex8\"[\S ]*>\s*(<.*>)\s*</Elements>", f.read(), flags=re.MULTILINE|re.DOTALL).groups()
			# 	nodes = re.findall(r'<node id[= ]*"(\d+)">(' + ','.join([r" *[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)+ *"]*3) + r')</node>', Nodes, flags=re.MULTILINE|re.DOTALL)
			# 	elems = re.findall(r'<elem id[= ]*"(\d+)">(' + ','.join([r' *\d+ *']*8) + r')</elem>', Elements, flags=re.MULTILINE|re.DOTALL)

			# nodeid = np.asarray([int(nd[0]) for nd in nodes])-1
			# nodes_with_id = np.asarray([nd[1].split(',') for nd in nodes], dtype=float)
			# elems = np.asarray([el[1].split(',') for el in elems], dtype=int)-1
			# nodes = np.empty((nodeid.max()+1,3))
			# nodes[:] = np.nan
			# nodes[nodeid,:] = nodes_with_id[:]



	def rewrite(self, file, **kwargs):
		if not len(kwargs):
			self.write(file)
		with open(file,'r+') as f:
			original = f.read()
			s = '.*'.join([rf"<{k}[\S ]*>\s*(<.*>)\s*</{k}>" for k in kwargs])
			matches = re.search(s, original, flags=re.MULTILINE|re.DOTALL).groups()



