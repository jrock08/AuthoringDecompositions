import urllib2
from tarfile import TarFile
import os

def read_and_write_url(url, file_name):
    response = urllib2.urlopen(url)
    with open(file_name, 'wb') as f:
        block_sz = 8192
        while True:
            buff = response.read(block_sz)
            if not buff:
                break
            f.write(buff)


def main():
    output_location = '/data/jrock/'
    print 'output location is %s'%(output_location)

    print 'downloading train data'
    read_and_write_url(
        'http://webhost.engr.illinois.edu/~jjrock2/data/AuthoringArxiv16/train_data.tar',
        output_location + 'train_data.tar')
    TarFile.open(output_location + 'train_data.tar').extractall(output_location)
    print 'Train directory created at %strain_data'%(output_location)
    os.remove(output_location + 'train_data.tar')
    
    print 'downloading test data'
    read_and_write_url(
        'http://webhost.engr.illinois.edu/~jjrock2/data/AuthoringArxiv16/test_data.tar',
        output_location + 'test_data.tar')
    TarFile.open(output_location + 'test_data.tar').extractall(output_location)
    print 'Test directory created at %stest_data'%(output_location)
    os.remove(output_location + 'test_data.tar')

if __name__ == '__main__':
    main()
