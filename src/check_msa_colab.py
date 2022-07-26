import argparse
import sys
import os
import pdb
import re

##############FUNCTIONS##############

def read_a3m(a3mfile):
    '''Read an a3m file
    '''
    parsed_a3m = {}
    #Regex pattern
    pattern =  '[A-Z]+'
    nhits = 0
    with open(a3mfile, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line[0]=='>':
                nhits+=1
            else:
                if line[0]=='#':
                    continue
                else:
                    parsed_a3m[nhits] = ''.join(re.findall('[A-Z,-]+', line))

    return parsed_a3m


def process_a3m(a3mfile, sequence, outname):
    '''Process a3m file - remove insertions and get only matches and gaps
    Write these to a new file
    '''

    parsed_a3m = read_a3m(a3mfile)
    if len([*parsed_a3m.values()][0])!=len(sequence):
        print('The sequence length does not match with the MSA.')
        sys.exit()
    with open(outname, 'w') as file:
        for key in parsed_a3m:
            file.write('>'+str(key)+'\n')
            file.write(parsed_a3m[key]+'\n')
