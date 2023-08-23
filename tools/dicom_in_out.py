import os, csv, time, shutil, argparse
import dicom

def main(args):

    os.makedirs(args.output, exist_ok=True)

    img = dicom.read(args.input)

    origin = img.GetOrigin()

    if args.reset_origin:
        img.SetOrigin((0.0, 0.0, 0.0))
        print('origin reset')

    info = img.info

    # extract the following info items from original series
    info_keys = {
    "0020|0010(SH)": "Study ID",
    "0012|0040(LO)": "Clinical Trial Subject ID",
    "0012|0071(LO)": "Clinical Trial Series ID",
    "0012|0072(LO)": "Clinical Trial Series Description",
    "0010|0010(PN)": "Patient's Name",
    "0010|0040(CS)": "Patient's Sex",
    "0010|1010(AS)": "Patient's Age",
    "0010|0030(DA)": "Patient's Birth Date",
    "0008|0012(DA)": "Instance Creation Date",
    "0008|0020(DA)": "Study Date",
    "0008|0021(DA)": "Series Date",
    "0010|4000(LT)": "Patient Comments",
    }

    info_keys = {k[:9]:v for k,v in info_keys.items()}
    info_keep = {k:v for k,v in info.items() if k in info_keys}

    with open(args.output+'_info.csv', 'w', newline='') as f:
        writer = csv.writer(f).writerows(info.items())
                             
    # slope and intercept
    if '0028|1052' in info and '0028|1053' in info:
        info_keep['0028|1052'] = info['0028|1052']
        info_keep['0028|1053'] = info['0028|1053']

    if args.study:
        info_keep["0020|0010"] = args.study

    if args.subject:
        info_keep["0012|0040"] = args.subject

    if args.series:
        info_keep["0012|0071"] = args.series

    # writing info and copying dicom files must succeed or fail together
    success = True

    info_log = {}
    info_log['Entry Date'] = time.strftime("%Y-%m-%d")
    info_log['Entry Time'] = time.strftime("%H:%M:%S")
    info_log['Origin'] = origin
    info_log['Reset Origin'] = args.reset_origin
    info_log['De-identified'] = args.anonymize

    # log the current job
    
    info_log.update({info_keys[k]:info_keep[k] if k in info_keep else '' for k in info_keys.keys() })

    _log_path = os.path.join(os.path.dirname(args.log_path), '~' + os.path.basename(args.log_path))
    try:
        with open(_log_path, 'w', newline='') as _f:
            writer = csv.DictWriter(_f, fieldnames=list(info_log.keys()))
            writer.writeheader()
            if os.path.exists(args.log_path):
                with open(args.log_path, 'r') as f:
                    reader = csv.DictReader(f)
                    read_list = list(reader)
                    writer.writerows(read_list)
            writer.writerow(info_log)
            
    except Exception as e:
        success = False
        print(f'cannot write to {args.log_path}: {e}')
    
    # 
    if args.anonymize:
        info_keep["0010|0010"] = args.subject if args.subject else 'deid'
        print('anonymized')

    # write dicom files
    try:
        img.info = info_keep
        dicom.write(img, args.output)

    except:
        success = False
        print('cannot write dicom')

    if success:
        shutil.move(_log_path, args.log_path)

    else:
        if os.path.exists(_log_path):
            os.remove(_log_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--anonymize', default=False, action='store_true')
    parser.add_argument('--reset-origin', default=False, action='store_true')
    parser.add_argument('--log-path', type=str, default=os.path.join(os.path.dirname(__file__),'dicom_log.csv'))
    parser.add_argument('--study', type=str, default='', help='example: "AUTOSEG" or "FACEPRED"')
    parser.add_argument('--subject', type=str, default='', help='example: "001" or "N01"')
    parser.add_argument('--series', type=str, default='', help='example: "PRE" or "POST"')
    parser.add_argument('--method', type=str, default='', help='example: "lq" or "xy"')

    parser.add_argument('--comments', type=str, default='', help='example: "missing tooth L1-R" or "hemifacial hypoplasia"')

    args = parser.parse_args()

    main(args)
