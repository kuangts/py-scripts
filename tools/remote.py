import os, sys, argparse, stat
from pathlib import Path, PurePath, PurePosixPath
import paramiko


def connect_(server, user, passwd):

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    client.connect(server, username=user, password=passwd)
    return client


def execute_(ssh:paramiko.SSHClient, command, *args):

    exec_line = command + ' ' + " ".join(f"'{x}'" for x in args)
    print(f'execute {exec_line}')
    stdin, stdout, stderr = ssh.exec_command(exec_line)

    for line in stdout:
        print(line)
    for line in stderr:
        print(line)
        
    return None


def isfile_(sftp:paramiko.SFTPClient, remote_path:str):
    try:
        return stat.S_ISREG(sftp.stat(remote_path).st_mode)
    except:
        return False

def isdir_(sftp:paramiko.SFTPClient, remote_path:str):
    try:
        return stat.S_ISDIR(sftp.stat(remote_path).st_mode)
    except:
        return False



def remove_(sftp:paramiko.SFTPClient, remote_path:str):
    
    # assuming posix remote
    # must be either file -> file or directory -> directory
    print(f'remove {remote_path}')

    # if path is file
    if isfile_(sftp, remote_path):
        sftp.remove(remote_path)

    # if path is dir
    elif isdir_(sftp, remote_path):

        for fileattr in sftp.listdir_attr(remote_path): 

            f = str(PurePosixPath(remote_path)/fileattr.filename)

            if stat.S_ISREG(fileattr.st_mode):
                sftp.remove(f)
            elif stat.S_ISDIR(fileattr.st_mode):
                remove_(sftp, f)
            else:
                print('unusual file detected')

        sftp.rmdir(remote_path)
        print(f'{remote_path} deleted')    

    else:
        raise FileNotFoundError(f'cannot remove file {remote_path}')
    
    return None


def put_(sftp:paramiko.SFTPClient, local_path:str, remote_path:str, override=True):

    # assuming posix remote
    # must be either file -> file or directory -> directory
    print(f'put {local_path} to {remote_path}')

    # if path is file
    if os.path.isfile(local_path):

        if isdir_(sftp, remote_path):
            remote_path = str(PurePosixPath(remote_path)/Path(local_path).name)

        if isfile_(remote_path):
            sftp.remove(remote_path)

        sftp.put(local_path, remote_path)

    # if path is dir
    elif os.path.isdir(local_path):

        if not isdir_(sftp, remote_path):
            try:
                sftp.mkdir(remote_path)
            except:
                raise ValueError(f'cannot copy dir from {local_path} to {remote_path}')

        for file in os.listdir(local_path): 

            f = os.path.join(local_path, file)
            f_ = str(PurePosixPath(remote_path)/file)

            if os.path.isfile(f):
                sftp.put(f, f_)
            elif os.path.isdir(f):
                put_(sftp, f, f_)
            else:
                ValueError(f'cannot recognize {f}')


        # for a,b,c in os.walk(local_path):
        #     os.chdir(a)
        #     sub_path = PurePosixPath(Path(a).relative_to(local_path))
        #     for d in b: sftp.mkdir(str(sub_path/d))
        #     for f in c: sftp.put(f, str(sub_path/f))

    else:
        raise FileNotFoundError(f'cannot find file {local_path}')

    return None


def get_(sftp:paramiko.SFTPClient, remote_path:str, local_path:str, override=True):

    # assuming posix remote
    # must be either file -> file or directory -> directory

    print(f'get from {remote_path} to {local_path}')
    
    # if path is file
    if isfile_(sftp, remote_path):

        if os.path.isdir(local_path):
            # if not os.access(local_path, os.W_OK):
            #     raise ValueError(f'{local_path} does not have write permission')
            local_path = str(Path(local_path)/PurePosixPath(remote_path).name)

        sftp.get(remote_path, local_path)

    # if path is dir
    elif isdir_(sftp, remote_path):

        if not os.path.isdir(local_path):
            try:
                os.makedirs(local_path)
            except:
                raise ValueError(f'cannot copy dir from {remote_path} to {local_path}')

        for fileattr in sftp.listdir_attr(remote_path):
            
            f = str(PurePosixPath(remote_path)/fileattr.filename)
            f_ = str(Path(local_path)/fileattr.filename)

            if stat.S_ISREG(fileattr.st_mode):
                sftp.get(f, f_)
            elif stat.S_ISDIR(fileattr.st_mode):
                os.makedirs(f_)
                get_(sftp, f, f_)
            else:
                print(f'cannot recognize {f}')

        print(f'{remote_path} copied')    

    return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('args', type=str, nargs='*')
    parser.add_argument('--server', type=str, default='lambda-xia.tmh.tmhs')
    parser.add_argument('--username', type=str, default='gu')
    parser.add_argument('--password', type=str, default='gu')
    parser.add_argument('--remote-file-location', type=str, default='/data/gu/temp')

    return parser.parse_args()

