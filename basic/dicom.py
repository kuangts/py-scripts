from collections.abc import Sequence
from xml.etree import ElementTree as ET
import os, re, codecs, csv, shutil, copy, time
import SimpleITK as sitk
import numpy as np


__ALL__ = []



class Tag(list):
	
	@property
	def tag(self, conj='|'):
		return conj.join(self[0:2])

	@property
	def name(self):
		return self[2]

class Dictionary:
# part06.html and dicom.css are downloaded from https://dicom.nema.org/medical/dicom/current/output/html/ with no modification

	tags_for_record = [
		"Study ID", # example FACEPRED
		"Clinical Trial Subject ID", # example (this is the anonymized name of the patient, throughout each study) FACEPRED-001
		"Clinical Trial Series ID", # example (64 chars max) FACEPRED-001-pre
		"Clinical Trial Series Description", # example (64 chars max) Facial Prediction Case 001 Pre-operative CT Scan
		"Patient's Name",
		"Patient's Sex",
		"Patient's Age",
		"Patient's Birth Date",
		"Instance Creation Date",
		"Study Date",
		"Series Date",
	]


	def __init__(self):
		with codecs.open(rf'{__file__}\..\nema-dicom-part06\part06.html', mode='r', encoding='utf-8') as f:
			t = f.read()
			body = re.findall(r'<body>.*</body>', t, re.MULTILINE | re.DOTALL)[0]
			tree = ET.fromstring(body)

		table = []
		for tr in tree.find(".//*[@id='table_6-1']/..//*[@class='table-contents']/table/tbody"):
			entry = []
			row = tr.findall('./td/p')
			for d in [0,1,3]:
				try:
					entry += [row[d].find('span').text]
				except:
					try:
						entry += [row[d].find('a').tail]
					except:
						entry += [row[d].text]
			table += [ Tag([*entry.pop(0).strip('()').split(','), *map(str,entry)]) ] # only use Tag.__init__() here at the creation
		setattr(self, '_table', table)


	def search(self, *list_of_necessary_words):
		result = []
		for t in self._table:
			if all( [ any( [ w in i for i in t ] ) for w in list_of_necessary_words ] ):
				result += [t]
		return result


	def convert(self, item):
		
		if isinstance(item, str):
			item = item.strip('()')
			if '|' in item:
				item_split = item.split('|')
			elif ',' in item:
				item_split = item.split(',')
			else:
				item_split = []
			if len(item_split)==2 and len(item_split[0])==4 and len(item_split[1])==4 :
				item = item_split

		if isinstance(item, str):
			for i in range(len(self._table)):
				if item==self._table[i][2]:
					v = (self._table[i][0], self._table[i][1])
					return v
		elif isinstance(item, Sequence):
			assert len(item)==2 and isinstance(item[0], str) and isinstance(item[1], str)
			for i in range(len(self._table)):
				if item[0]==self._table[i][0] and item[1]==self._table[i][1]:
					return self._table[i][2]
		return None


	def __getitem__(self, item):

		if isinstance(item, Tag):
			return item

		translation_table = {
			'Study ID': ('study',), 
			"Patient's Name": ('name','patient'), 
			"Patient's Age": ('age',), 
			"Patient's Sex": ('sex','gender'), 
			"Patient's Birth Date": ('birthdate','birthday','dob','dateofbirth'), 
			'Clinical Trial Subject ID': ('subject','subjectid','id'), 
			'Clinical Trial Series ID': ('series','seriesid'), 
			'Clinical Trial Series Description': ('description','seriesdescription'), 
		} # use this dictionary case-insensitively and alphanumerically


		if isinstance(item, str):
			if '|' in item:
				item_split = item.split('|')
			elif ',' in item:
				item_split = item.split(',')
			else:
				item_split = []
			if len(item_split)==2 and len(item_split[0])==4 and len(item_split[1])==4 :
				item = item_split
			else:
				for k,v in translation_table.items():
					if ''.join(filter(str.isalnum, item)).lower() in v:
						item = k
						break

		if isinstance(item, str):
			for i in range(len(self._table)):
				if item==self._table[i][2]:
					return self._table[i]

		if isinstance(item, Sequence) and len(item)==2:
			for i in range(len(self._table)):
				if item[0]==self._table[i][0] and item[1]==self._table[i][1]:
					return self._table[i]

		return None


	def vr(self, *item):
		if len(item)==1:
			item = item[0]
			assert isinstance(item, str)
			for i in range(len(self._table)):
				if item==self._table[i][2]:
					return self._table[i][3]
		elif len(item)==2:
			assert isinstance(item[0], str) and isinstance(item[1], str)
			for i in range(len(self._table)):
				if item[0]==self._table[i][0] and item[1]==self._table[i][1]:
					return self._table[i][3]
		return None


	def __contains__(self, item):
		return self[item] is not None


	# def all_tags(self, return_str_or_tuple='str', conj='|'):
	# 	if return_str_or_tuple == 'str':
	# 		return [ conj.join(t[0:2]) for t in self._table]
	# 	else:
	# 		return [ (t[0],t[1]) for t in self._table]

	# def all_names(self):
	# 	return [ t[2] for t in self._table]

class Info(dict):

	def __init__(self, *args, check_keys=False, **kwargs):
		super().__init__(*args, **kwargs)
		if check_keys:
			for k in list(self.keys()):
				x = lookup[k]
				if x is None:
					print(f'{k} is not a key')
					continue
				else:
					self[x] = self.pop(k)

	def tags(self):
		return list(self.keys())

	def names(self):
		return [ lookup[k].tag for k in self.keys() ]

	def __contains__(self, item):
		x = lookup[item]
		return x is not None and super().__contains__(x.tag)

	def __getitem__(self, item):
		if isinstance(item, Tag):
			return super().__getitem__(item.tag)
		else:
			return super().__getitem__(lookup[item].tag)
		
	def __setitem__(self, item, val):
		if isinstance(item, Tag):
			super().__setitem__(item.tag, val)
		else:
			super().__setitem__(lookup[item].tag, val)

	def filter(self, tags_to_keep):
		# return a generic dictionary, with keys from tags_to_keep and present
		self_filt = {}
		for k in tags_to_keep:
			x = lookup[k]
			if x is None:
				print(f'{k} is not a key')
				continue
			elif x not in self:
				print(f'{k} not found')
				continue
			self_filt[k] = self[x]
		return self_filt



def read(dicom_series_file_or_dir, return_image=True, return_info=False, **dicomargs):
	if not os.path.exists(dicom_series_file_or_dir):
		return None
	elif os.path.isfile(dicom_series_file_or_dir):
		try:
			file_reader = sitk.ImageFileReader()
			file_reader.SetFileName(dicom_series_file_or_dir)
			file_reader.ReadImageInformation()
			dicom_series_file_or_dir = os.path.dirname(dicom_series_file_or_dir) # change this later to finding files within the same series
		except:
			print(dicom_series_file_or_dir, "has no image information")
			return None
	
	series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_series_file_or_dir)
	if not series_IDs:
		print("ERROR: directory \"" + dicom_series_file_or_dir + "\" does not contain a DICOM series.")
		return None
	if len(series_IDs)>1:
		print('WARNING: more than one series found')

	result = ()

	if return_image:
		dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_series_file_or_dir, series_IDs[0])
		reader = sitk.ImageSeriesReader()
		reader.SetImageIO("GDCMImageIO")
		reader.SetFileNames(dicom_names)
		reader.MetaDataDictionaryArrayUpdateOn()
		img = reader.Execute()
		result += (img,)

	if return_info:
		file_reader = sitk.ImageFileReader()
		file_reader.SetFileName(dicom_names[0])
		file_reader.LoadPrivateTagsOn()
		file_reader.ReadImageInformation()
		info = { k:file_reader.GetMetaData(k) for k in file_reader.GetMetaDataKeys() }
		result += (Info(info),)
	
	if len(result)==1:
		result = result[0]
	return result
    



def write(img, dcm_dir, info, file_name_format='{:04}.dcm'):
	dcm_dir = os.path.expanduser(dcm_dir)
	dcm_dir = os.path.normpath(os.path.realpath(dcm_dir))
	if os.path.exists(dcm_dir) and not os.path.isdir(dcm_dir):
		print(f'cannot write to {dcm_dir}, not a directory')
	else:
		os.makedirs(dcm_dir, exist_ok=True)
	if os.listdir(dcm_dir):
		print(f'cannot write to {dcm_dir}, directory not empty')
		return

	# downloaded from simpleitk example 'Dicom Series From Array'
	# at https://simpleitk.readthedocs.io/en/master/link_DicomSeriesFromArray_docs.html
	# modified to always write int16, original huntsfield value

	# Write the 3D image as a series
	# IMPORTANT: There are many DICOM tags that need to be updated when you modify
	#            an original image. This is a delicate operation and requires
	#            knowledge of the DICOM standard. This example only modifies some.
	#            For a more complete list of tags that need to be modified see:
	#                  http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
	#            If it is critical for your work to generate valid DICOM files,
	#            It is recommended to use David Clunie's Dicom3tools to validate
	#            the files:
	#                  http://www.dclunie.com/dicom3tools.html

	writer = sitk.ImageFileWriter()
	writer.SetImageIO("GDCMImageIO")
	# Use the study/series/frame of reference information given in the meta-data
	# dictionary and not the automatically generated information from the file IO
	writer.KeepOriginalImageUIDOn()

	modification_time = time.strftime("%H%M%S")
	modification_date = time.strftime("%Y%m%d")

	# Copy some of the tags and add the relevant tags indicating the change.
	# For the series instance UID (0020|000e), each of the components is a number,
	# cannot start with zero, and separated by a '.' We create a unique series ID
	# using the date and time. Tags of interest:
	direction = img.GetDirection()
	# Tags shared by the series.
	series_tag_values = {

		"0008|0060": "CT", # Setting the type to CT so that the slice location is preserved and the thickness is carried over.
		"0008|0008": "DERIVED\\SECONDARY",  # Image Type
		"0020|000e": "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time,  # Series Instance UID
		"0020|0037": '\\'.join(map(str, (direction[0], direction[3], direction[6],
										direction[1], direction[4], direction[7]))), # Image Orientation # (Patient)
	}
	series_tag_values.update(info)
	# Write slices to output directory
	for i in range(img.GetDepth()):
		image_slice = img[:, :, i]
		#   Instance Creation Date
		series_tag_values["0008|0012"] = time.strftime("%Y%m%d")
		#   Instance Creation Time
		series_tag_values["0008|0013"] = time.strftime("%H%M%S")
		# (0020, 0032) image position patient determines the 3D spacing between
		# slices.
		#   Image Position (Patient)
		series_tag_values["0020|0032"] = '\\'.join(map(str, img.TransformIndexToPhysicalPoint((0, 0, i))))
		#   Instance Number
		series_tag_values["0020|0013"] = str(i)
		#   set
		for k,v in series_tag_values.items():
			image_slice.SetMetaData(k,v)
		# Write to the output directory and add the extension dcm, to force
		# writing in DICOM format.
		writer.SetFileName(os.path.join(dcm_dir, file_name_format.format(i+1)))
		writer.Execute(image_slice)


def anonymize(dcm_dir_in, dcm_dir_out, reset_origin=True, **anon_info):
	# kwargs can include study_id, subject_id, series_id, description, ...
	# program will try to understand those terms and assign to the correct tag
	# those tags will override what is in dicom series in the returning dict
	img, full_info = read(dcm_dir_in, return_info=True)
	anon_info = Info(anon_info, check_keys=True)
	full_info.update(anon_info)
	if reset_origin:
		img.SetOrigin((0.0,0.0,0.0))
	write(img, dcm_dir_out, anon_info)
	return full_info

def write_info(to_registry, info, fieldnames=[]):
	# fieldnames is ignored when there's existing data
	if not isinstance(info, Info):
		info = Info(info, check_keys=True)
		
	data = []
	if not len(fieldnames):
		fieldnames = lookup.tags_for_record
	to_registry = os.path.realpath(os.path.normpath(os.path.expanduser(to_registry)))
	temp_copy = ''

	if os.path.isfile(to_registry):
		temp_copy = os.path.join(os.path.dirname(to_registry), '_'+ os.path.basename(to_registry))
		shutil.copy(to_registry, temp_copy)
		# read fieldnames, check duplicate
		with open(to_registry, 'r', newline='') as f:
			reader = csv.DictReader(f)
			data = [ row for row in reader ]
			if len(data):
				fieldnames = list(data[0].keys())

	# check duplicate is not implemented yet

	data += [info.filter(fieldnames=fieldnames)]

	try: 
		with open(to_registry, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for d in data:
				writer.writerow(d)
	except:
		print('csv write failed')
		if temp_copy:
			shutil.copy(temp_copy, to_registry)
	finally:
		if temp_copy:
			os.remove(temp_copy)


def close_enough(d1, d2, *kwargs):
	img1 = read(d1)
	img2 = read(d2)
	equality = (np.allclose(img1.GetSize(),img2.GetSize()))\
	and (np.allclose(img1.GetOrigin(),img2.GetOrigin()))\
	and (np.allclose(img1.GetSpacing(),img2.GetSpacing()))\
	and (np.allclose(img1.GetDirection(),img2.GetDirection()))\
	and (np.allclose(sitk.GetArrayFromImage(img1),sitk.GetArrayFromImage(img2)))
	return equality
	# kwargs
	# file equality
	# value equality
	# value equality up to scale and shift
	
	



lookup = Dictionary()
