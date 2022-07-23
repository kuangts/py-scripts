from basic import dicom

print(dicom.lookup['name'],
    dicom.lookup['subject'].tag,
    dicom.lookup['series_id'].name,
    *dicom.lookup.search('Patient', 'Name'),
    dicom.lookup.convert('0010|0010'),
    dicom.lookup.convert("Patient's Name"),
    'subject' in dicom.lookup,
    sep='\n')


img, info = dicom.read(r'C:\Users\tmhtxk25\Desktop\FL-dicom\FL001', return_info=True)
print(info,
    'name' in info,
    info['DoB'],
    sep='\n')

dicom.anonymize(r'C:\Users\tmhtxk25\Desktop\FL-dicom\FL001', r'C:\Users\tmhtxk25\Desktop\test', reset_origin=True)

same_or_not = dicom.close_enough(r'C:\Users\tmhtxk25\Desktop\FL-dicom\FL001', r'C:\Users\tmhtxk25\Desktop\test')
print(same_or_not)