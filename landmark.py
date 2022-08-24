import os, sys, re, argparse, json
import plotly
from copy import deepcopy
from math import isnan
from collections.abc import Sequence

_nan = float('nan')
_anynan = lambda x: any(map(isnan,x))
with open('labels.json','r') as f:
	Labels = json.loads(f.read())


class Landmark(dict):
	'''This is a class for handling landmark files of various formats
	No __init__ -- use this exactly like a dictionary
	Be mindful of the data type put in - list and numpy.ndarray both works with the class
	Examples:
			lmk = Landmark.read('path/to/file') # specify header if header is present
			lmk.write('path/to/file') # write only text files
			lmk = Landmark.merge(lmk1, lmk2)
			lmk_select = lmk.select(['Gb','N','Rh'])
			lmk_select, lmk_remaining = lmk.select({'Cranium', 'ZMC Right'}, return_remaining=True)
			lmk_select = Landmark.merge(**lmk_select)
			lmk_sorted = lmk.sorted(Labels['Soft Tissue'])
			lmk_sorted['Gb'] = [1,2,3]

	'''
	@classmethod
	def read(cls, file, **parseargs):
		# read text file, mainly txt and csv
		with open(file, 'r') as f:
			lmk_str = f.read()
			return cls.parse(lmk_str, **parseargs)

	@classmethod
	def from_excel(cls, file):
		import pandas
		# read excel sheet with very specific format
		V = pandas.read_excel(file, header=0, engine='openpyxl').values
		return cls(zip(V[1:,0], V[1:,1:4]+V[0,1:4]))

	def write(self, file, **kwargs):
		write_str = self.string(**kwargs)
		if write_str:
			with open(file,'w') as f:
				f.write(write_str)

	def remove_nan(self):
		for l in list(self.keys()):
			if _anynan(self[l]):
				self.pop(l)
		return self

	@classmethod
	def merge(cls, *args, **kwargs):
		lmk = cls()
		for l in args:
			lmk.update(l)
		for l in kwargs.values():
			lmk.update(l)
		return lmk

	@property
	def coordinates(self):
		return list(self.values())

	@property
	def labels(self):
		return list(self.keys())

	@classmethod
	def parse(cls, lmk_str, header=None, separator='[,: ]+', line_ending='[;\n]+', nan_str={'N/A','n/a','NA','na','nan','NaN'}):		
		'''
		`header` stores the number of lines before read data begins. it must be set if not `None` - automatic determination is not supported yet.
		checks any combination of ';' and '\n' for line breaks
		checks any combination of ',', ':', and space for delimiter within each line
		detects if label is present
			with labels, each line starting with alphabetic characters
			without label, numeric only, use 0-based index in string as label
		'''
		lmk = cls()

		# 'header' stores number of lines in the begining of file before data, might be different from pandas, perform n splits where n equals 'header'
		if header:
			lmk_str = lmk_str.split('\n', header) 
			lmk.header = lmk_str[0:header]
			lmk_str = lmk_str[-1]

			# regexp checks for patterns ';\n' or '\n' or ';' (in order given by a 'Sequence' of str), then splits at all occurrences 

		assert line_ending and separator, 'must provide pattern for parsing'

		lmk_str = lmk_str.strip().strip(line_ending) # removes trailing newlines and semicolon to prevent creation of '' after split
		lines = re.split(line_ending, lmk_str)
		for i,line in enumerate(lines):
			if re.fullmatch('^[\s]*[a-zA-Z]+.*', line) is not None:
				# label is present - read four columns
				label, *coord = re.split(separator, line.strip())
			else:
				coord = re.split(separator, line.strip())
				if len(coord) == 4:
					label, *coord = coord
				elif len(coord) == 3:
					label = str(i)

			assert len(coord)==3, 'split went wrong'

			coord = [_nan if x.strip() in nan_str else float(x) for x in coord] # if nan, assign nan to coordinate 

			if not any(coord): # all zero scenario
				coord = [_nan]*3 
			lmk[label.strip()] = coord
		return lmk

	def string(self, nan_str='0.0', formatted=False, keep_label=True, keep_coordinate=True, header=None, separator=',', line_ending='\n'):
		'''
		`header` stores the header content of the coming file, in a `list` or `str`. a `list` of three `str`s would correspond to three header lines. the default is `None`, no header. if set to `''`, it will try to write header using previous header if possible.
		an nan value is written to file as `nan_str`. if `nan_str` is set to '0.0', then nan coordinates are written as '0.0, 0.0, 0.0', which is the default
		if nan_str is '', all nan values are removed
		'''

		# line_ending='\n' -> csv
		# line_ending=';'  -> cass-readable
		# line_ending=None -> do not join lines - returns list for further processing

		lmk = []

		if header is None:
			header = ''
		else:
			header = header if header else self.header
			header = '\n'.join(header if isinstance(header,Sequence) and not isinstance(header,str) else [header]) + '\n'
		
		remove_nan = len(nan_str)==0

		if formatted:
			if not self.len(remove_nan=remove_nan):
				lmk = ''
			else:
				header += '   LABEL  |        X        Y        Z\n' + '-'*40 + '\n'
				lmk = '\n'.join([f'{l:10}| {x[0]:8.3f} {x[1]:8.3f} {x[2]:8.3f}' for l,x in self.items()])+'\n\n'
				if self.len(remove_nan=False)==len(self):
					lmk += f'Total: {len(self)}\n'
				else:
					lmk += f'Present/Total: {self.len(remove_nan=False)}/{len(self)}\n'
		else:
			for label, coord in self.items():
				if not remove_nan or not _anynan(coord):
					l = []
					if keep_label:
						l += [label]
					if keep_coordinate:
						if _anynan(coord):
							l += [nan_str]*3 
						else:
							l += [*map(str,coord)]
					lmk += [l]

			if separator is None:
				return lmk
			lmk = [separator.join(l).strip(separator) for l in lmk]

			if line_ending is None:
				return lmk
			lmk = line_ending.join(lmk).strip(line_ending)

			if lmk:
				lmk += line_ending
		
		return header + lmk

	def select(self, select_labels_or_groups, return_remaining=False):

		# use list for labels and set for groups
		# for a list of labels, returns one Landmark instance with selected labels
		# for a set of groups, returns dictionary of Landmark instances
		if isinstance(select_labels_or_groups, str):
			raise ValueError('use dict["key"] notation' )
		elif isinstance(select_labels_or_groups, Sequence):
			lmk_selected, lmk_remaining = self.__class__(), self.copy()
			lmk_selected.update({ l:lmk_remaining.pop(l) for l in self if l in select_labels_or_groups })
			return (lmk_selected, lmk_remaining) if return_remaining else lmk_selected

		elif isinstance(select_labels_or_groups, set):
			lmk_selected, lmk_remaining = {}, self.copy()
			for s in select_labels_or_groups:
				lmk_selected[s] = self.__class__({ l:lmk_remaining.pop(l) for l in self if l in Labels[s] })
			return (lmk_selected, lmk_remaining) if return_remaining else lmk_selected

		else:
			raise ValueError('input is not supported')

	def sorted(self, ordered_labels):
		# select those labels, and order them, filling in nan if necessary
		self_new_copy = self.copy()
		v = [self_new_copy.setdefault(o, [_nan,_nan,_nan]) for o in ordered_labels]
		return self.__class__(zip(ordered_labels, v))

	def copy(self):
		return deepcopy(self)

	def len(self, remove_nan=False):
		l = len(self)
		if remove_nan:
			l -= sum([ _anynan(v) for v in self.values() ])
		return l

	def __repr__(self):
		return self.string(formatted=True)

	@property
	def header(self):
		# n items on the list _header represent n lines of header
		if not hasattr(self, '_header'):
			setattr(self, '_header', ())
		return self._header 

	@header.setter
	def header(self, value):
		if isinstance(value, str):
			value = (value,)
		setattr(self, '_header', value)

	def move_to_mask(self, mask, threshold=None):

		import pkg_resources
		try:
			pkg_resources.require(['SimpleITK','scipy','numpy'])
		except Exception as e:
			print(e)
			return None
		import SimpleITK as sitk
		import numpy as np
		from scipy.ndimage import binary_dilation, binary_erosion

		lmk_not_moved, lmk = self.select({'Detached'}, return_remaining=True)
		ind2coord = lambda index: np.array([ mask.TransformIndexToPhysicalPoint(ind.tolist()) for ind in index ])
		closest = lambda l, bd: bd[np.argmin(np.sum((bd - l)**2, axis=1)),:]

		if isinstance(mask, sitk.SimpleITK.Image):
			arr = sitk.GetArrayFromImage(mask)
		else:
			print('wrong input argument')
			return None 
		arr = arr>0
		
		# pass 1 - dilation
		arr1_bd = np.logical_xor(arr, binary_dilation(arr))
		arr1_bd_ind = np.array(np.nonzero(arr1_bd)).T[:,::-1]
		coords_bd1 = ind2coord(arr1_bd_ind)

		# pass 2 - erosion
		arr2_bd = np.logical_xor(arr, binary_erosion(arr))
		arr2_bd_ind = np.array(np.nonzero(arr2_bd)).T[:,::-1]
		coords_bd2 = ind2coord(arr2_bd_ind)

		# average two passes
		coords_new = np.array([ closest(l, coords_bd1)/2 + closest(l, coords_bd2)/2 for l in self.coordinates ])

		# check dist with thres
		if threshold!=None:
			d = np.sum((coords_new-self.coordinates)**2, axis=1)**.5
			ind = d>threshold
			coords_new[ind] = self.coordinates[ind]

		lmk = Landmark(zip(lmk.labels, coords_new))
		lmk.update(lmk_not_moved['Detached'])

		return lmk


	def plot(self, figure=None, **kwargs):
		# if 'intensity' not in kwargs:
		# 	kwargs['intensity'] = self.V[:,0]
		# 	kwargs['colorscale'] = [(0, "red"), (1, "blue")]
		xyz = dict(zip(['x','y','z'], zip(*self.coordinates)))
		trc = plotly.graph_objects.Scatter3d(**xyz, text=self.labels, **kwargs)
		fig = plotly.graph_objects.Figure() if figure is None else figure
		fig.add_trace(trc)
		if figure is None:
			fig.show()




if __name__=='__main__':
	parser = argparse.ArgumentParser(allow_abbrev=True)
	parser.add_argument('input', type=str, nargs='?')
	parser.add_argument('-o', '--output', type=str, nargs='?')
	parser.add_argument('--num-header', type=int, default=None)
	parser.add_argument('--read-nan', nargs='+', default=('N/A','n/a','NA','na','nan','NaN')) 
	parser.add_argument('--read-delimiter', type=str, default='[,: ]+') 
	parser.add_argument('--read-line-break', type=str, default='[;\n]+')
	parser.add_argument('--select', nargs='+', type=str)
	parser.add_argument('--order', nargs='+', type=str)
	parser.add_argument('--formatted', action='store_true', default=False)
	parser.add_argument('--header', nargs='*', type=str)
	parser.add_argument('-nan','--nan-string', type=str, default='')
	parser.add_argument('-d','--delimiter', type=str, default=',')
	parser.add_argument('-b','--line-break', type=str, default='\n')
	parser.add_argument('--no-label', action='store_true', default=False) # print only coordinates
	parser.add_argument('--label-only', action='store_true', default=False) # print only labels

	args = parser.parse_args()

	if args.input is not None:
		if args.input.endswith('.xlsx'):
			lines = Landmark.from_excel(args.input).string()
		else:
			with open(args.input, 'r') as f:
				lines = f.read()
	else:
		lines = ''.join([line for line in sys.stdin])

	lmk = Landmark.parse(
				lines,
				header=args.num_header,
				nan_str=args.read_nan,
				separator=args.read_delimiter,
				line_ending=args.read_line_break)
	lmk_str = lmk.string(
				header=args.header,
				formatted=args.formatted,
				nan_str=args.nan_string,
				keep_label=not args.no_label,
				keep_coordinate=not args.label_only,
				separator=args.delimiter,
				line_ending=args.line_break)

	if args.output is not None:
		with open(args.output, 'wt', newline='') as f:
			f.write(lmk_str)
	else:			
		sys.stdout.write(lmk_str)

