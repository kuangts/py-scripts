import csv
from CASS import CASS
import numpy as np


f = CASS(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\n09^plan.CASS')
lmk = f.landmarks(index=range(1,300))
lmk = {i:np.around(v,1) for i,v in lmk.items()}
with open(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\Desktop\n09.csv','r') as f:
    t = list(csv.reader(f))

t = np.array(t)
xyz = t[:,1:].astype(float) + [80,0,0]
label = t[:,0]

mapping = {}

for i,v in lmk.items():
    ind = (xyz==v).all(axis=1).nonzero()[0]
    if len(ind)==1:
        mapping[i] = label[ind[0]]





mapping1 = {2: 'S', 1: 'Ba', 10: 'Gb', 11: 'N', 13: 'Fz-R', 14: 'Fz-L', 19: 'OrM-R', 20: 'OrM-L', 17: 'OrS-R', 18: 'OrS-L', 4: 'Ft-R', 5: 'Ft-L', 15: 'Or-R', 16: 'Or-L', 3: 'FMP', 177: 'IC', 163: 'SMF-R', 164: 'SMF-L', 8: 'Po-R', 9: 'Po-L', 6: 'M-R', 7: 'M-L', 28: 'ANS', 27: 'A', 12: 'Rh', 21: 'ION-R', 22: 'ION-L', 169: 'SOF-R', 170: 'SOF-L', 25: 'J-R', 23: 'Zy-R', 24: 'Zy-L', 26: 'J-L', 29: 'PNS', 171: 'GPF-R', 172: 'GPF-L', 30: 'U0', 33: 'U1E-R', 48: 'U1E-L', 34: 'U2E-R', 35: 'U3T-R', 50: 'U3T-L', 37: 'U4BC-R', 52: 'U4BC-L', 36: 'U4LC-R', 51: 'U4LC-L', 39: 'U5BC-R', 54: 'U5BC-L', 38: 'U5LC-R', 53: 'U5LC-L', 41: 'U6MBC-R', 56: 'U6MBC-L', 40: 'U6MLC-R', 55: 'U6MLC-L', 42: 'U6DBC-R', 57: 'U6DBC-L', 43: 'U6DLC-R', 58: 'U6DLC-L', 45: 'U7MBC-R', 60: 'U7MBC-L', 44: 'U7MLC-R', 59: 'U7MLC-L', 46: 'U7DBC-R', 61: 'U7DBC-L', 47: 'U7DLC-R', 62: 'U7DLC-L', 31: 'U1R-R', 32: 'U1R-L', 161: 'U0R', 98: 'B', 99: 'Pg', 100: 'Gn', 101: 'Me', 118: 'MF-R', 119: 'MF-L', 116: 'Co-R', 117: 'Co-L', 114: 'COR-R', 115: 'COR-L', 106: 'Cr-R', 107: 'Cr-L', 104: 'SIG-R', 105: 'SIG-L', 102: 'RMA-R', 103: 'RMA-L', 108: 'RP-R', 109: 'RP-L', 112: 'Go-R', 113: 'Go-L', 167: 'Gos-R', 168: 'Gos-L', 165: 'Goi-R', 166: 'Goi-L', 110: 'Ag-R', 111: 'Ag-L', 63: 'L0', 66: 'L1E-R', 82: 'L1E-L', 67: 'L2E-R', 83: 'L2E-L', 68: 'L3T-R', 84: 'L3T-L', 70: 'L4BC-R', 86: 'L4BC-L', 69: 'L4LC-R', 85: 'L4LC-L', 72: 'L5BC-R', 88: 'L5BC-L', 71: 'L5LC-R', 87: 'L5LC-L', 74: 'L6MBC-R', 90: 'L6MBC-L', 73: 'L6MLC-R', 89: 'L6MLC-L', 75: 'L6DBC-R', 91: 'L6DBC-L', 77: 'L6DLC-R', 93: 'L6DLC-L', 79: 'L7MBC-R', 95: 'L7MBC-L', 78: 'L7MLC-R', 94: 'L7MLC-L', 80: 'L7DBC-R', 96: 'L7DBC-L', 81: 'L7DLC-R', 97: 'L7DLC-L', 76: 'L6DC-R', 92: 'L6DC-L', 120: "Gb'", 121: "N'", 158: 'C', 127: "A'", 131: 'En-R', 132: 'En-L', 129: 'Ex-R', 130: 'Ex-L', 135: 'AL-R', 136: 'AL-L', 137: 'Ac-R', 138: 'Ac-L', 125: 'Mf-R', 126: 'Mf-L', 134: 'Prn', 141: 'CM', 133: 'Sn', 128: "OR'-R", 160: "OR'-L", 139: 'Nt-R', 140: 'Nt-L', 142: 'Nb-R', 143: 'Nb-L', 123: "Zy'-R", 124: "Zy'-L", 144: 'Ss', 146: 'Cph-R', 147: 'Cph-L', 150: 'Ch-R', 151: 'Ch-L', 148: 'Stm-U', 159: 'Stm-L', 152: "B'", 156: "Pog'", 157: "Gn'", 153: "Go'-R", 154: "Go'-L", 145: 'Ls', 149: 'Li', 122: "Me'", 155: 'Sl', 49: 'U2E-L', 189: 'GFC-R', 188: 'GFC-L', 185: 'L6CF-R', 184: 'L6CF-L', 192: 'SOr-R', 193: 'SOr-L', 194: 'L34Embr-L', 195: 'L34Embr-R', 196: 'U6CF-R', 197: 'U6CF-L'}
mapping2 = {166: 'Goi-L', 167: 'Gos-R', 168: 'Gos-L', 8: 'Po-R', 9: 'Po-L', 13: 'Fz-R', 14: 'Fz-L', 15: 'Or-R', 16: 'Or-L', 26: 'J-L', 171: 'GPF-R', 172: 'GPF-L', 11: 'N', 12: 'Rh', 28: 'ANS', 27: 'A', 29: 'PNS', 177: 'IC', 41: 'U6MBC-R', 56: 'U6MBC-L', 35: 'U3T-R', 50: 'U3T-L', 30: 'U0', 134: 'Prn', 121: "N'", 131: 'En-R', 132: 'En-L', 40: 'U6MLC-R', 39: 'U5BC-R', 37: 'U4BC-R', 34: 'U2E-R', 33: 'U1E-R', 48: 'U1E-L', 49: 'U2E-L', 52: 'U4BC-L', 55: 'U6MLC-L', 79: 'L7MBC-R', 74: 'L6MBC-R', 72: 'L5BC-R', 70: 'L4BC-R', 68: 'L3T-R', 67: 'L2E-R', 66: 'L1E-R', 63: 'L0', 82: 'L1E-L', 83: 'L2E-L', 84: 'L3T-L', 86: 'L4BC-L', 88: 'L5BC-L', 90: 'L6MBC-L', 95: 'L7MBC-L', 10: 'Gb', 115: 'COR-L', 116: 'Co-R', 117: 'Co-L', 106: 'Cr-R', 107: 'Cr-L', 112: 'Go-R', 113: 'Go-L', 110: 'Ag-R', 111: 'Ag-L', 99: 'Pg', 101: 'Me', 100: 'Gn', 98: 'B', 162: 'L0R', 189: 'GFC-R', 188: 'GFC-L', 169: 'SOF-R', 170: 'SOF-L', 2: 'S', 163: 'SMF-R', 164: 'SMF-L', 161: 'U0R', 1: 'Ba', 3: 'FMP', 104: 'SIG-R', 105: 'SIG-L', 120: "Gb'", 185: 'L6CF-R', 184: 'L6CF-L', 4: 'Ft-R', 5: 'Ft-L', 17: 'OrS-R', 18: 'OrS-L', 192: 'SOr-R', 193: 'SOr-L', 19: 'OrM-R', 20: 'OrM-L', 21: 'ION-R', 22: 'ION-L', 6: 'M-R', 7: 'M-L', 23: 'Zy-R', 24: 'Zy-L', 31: 'U1R-R', 32: 'U1R-L', 36: 'U4LC-R', 51: 'U4LC-L', 38: 'U5LC-R', 53: 'U5LC-L', 42: 'U6DBC-R', 57: 'U6DBC-L', 43: 'U6DLC-R', 58: 'U6DLC-L', 60: 'U7MBC-L', 45: 'U7MBC-R', 44: 'U7MLC-R', 59: 'U7MLC-L', 46: 'U7DBC-R', 61: 'U7DBC-L', 47: 'U7DLC-R', 62: 'U7DLC-L', 64: 'L1R-R', 65: 'L1R-L', 118: 'MF-R', 119: 'MF-L', 102: 'RMA-R', 108: 'RP-R', 103: 'RMA-L', 109: 'RP-L', 69: 'L4LC-R', 85: 'L4LC-L', 71: 'L5LC-R', 87: 'L5LC-L', 73: 'L6MLC-R', 89: 'L6MLC-L', 75: 'L6DBC-R', 91: 'L6DBC-L', 77: 'L6DLC-R', 93: 'L6DLC-L', 76: 'L6DC-R', 92: 'L6DC-L', 78: 'L7MLC-R', 94: 'L7MLC-L', 80: 'L7DBC-R', 96: 'L7DBC-L', 81: 'L7DLC-R', 97: 'L7DLC-L', 129: 'Ex-R', 130: 'Ex-L', 125: 'Mf-R', 126: 'Mf-L', 128: "OR'-R", 160: "OR'-L", 135: 'AL-R', 136: 'AL-L', 137: 'Ac-R', 138: 'Ac-L', 124: "Zy'-L", 141: 'CM', 139: 'Nt-R', 140: 'Nt-L', 142: 'Nb-R', 143: 'Nb-L', 133: 'Sn', 127: "A'", 144: 'Ss', 145: 'Ls', 146: 'Cph-R', 147: 'Cph-L', 148: 'Stm-U', 159: 'Stm-L', 150: 'Ch-R', 151: 'Ch-L', 149: 'Li', 155: 'Sl', 152: "B'", 156: "Pog'", 157: "Gn'", 122: "Me'", 158: 'C', 153: "Go'-R", 154: "Go'-L", 54: 'U5BC-L', 194: 'L34Embr-L', 195: 'L34Embr-R', 196: 'U6CF-R', 197: 'U6CF-L'}
mapping = {}

for i in mapping1:
    if i in mapping2:
        if mapping1[i] == mapping2[i]:
            mapping[i] = mapping1[i]
        else:
            print(mapping1[i] , mapping2[i])

mapping