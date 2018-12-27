#!/usr/bin/env python3
import os

def main():
    for fname in sorted(os.listdir('./')):
        if len(fname)==22 and fname[0:3]=='HI.' and \
                fname[3:11].isdigit() and \
                fname[12:17].isdigit() and fname[-5:]=='.fits':

            # read header and data
            file1 = open(fname, 'rb')
            head_lst = []
            c = 0
            while(True):
                row = file1.read(80)
                if b'\x92' in row:
                    row = row.replace(b'\x92',b"'")
                row = row.decode('ascii')
                c += 1
                head_lst.append(row)
                if row[0:3]=='END':
                    break
            while(c%36!=0):
                file1.read(80)
                c += 1
            data = file1.read()
            file1.close()

            file2 = open(fname,'wb')
            c = 0
            for row in head_lst:
                file2.write(row.encode())
                c += 1
            while(c%36!=0):
                blank_str = ' '*80
                file2.write(blank_str.encode())
                c += 1
            file2.write(data)
            file2.close()

            print('fix', fname)

if __name__=='__main__':
    main()
