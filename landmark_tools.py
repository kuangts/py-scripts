import csv


def landmark_from_csv(file:str):
    with open(file, 'r') as f:
        axyz = list(csv.reader(f))
    return {b[0]:[float(a) for a in b[1:]] for b in axyz}



def write_landmark(lmk, file):
    with open(file, 'w', newline='') as f:
        csv.writer(f).writerows([[k,*v] for k,v in lmk.items()])


