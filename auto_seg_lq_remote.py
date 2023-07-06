from remote_tools import *

if __name__=='__main__':

    args = get_args()
    command = 'auto-seg-lq'
    from_dir = args.args[0]
    to_dir = args.args[1]
    remote_dir = str(PurePosixPath(args.remote_file_location)/Path(from_dir).name)
    os.makedirs(to_dir, exist_ok=True)

    assert os.path.isdir(from_dir) and os.access(from_dir, os.R_OK), 'file path is not valid'

    with connect_(args.server, args.username, args.password) as ssh:
        with ssh.open_sftp() as sftp:
            
            # clear existing files
            if isdir_(sftp, remote_dir) or isfile_(sftp, remote_dir):
                remove_(sftp, remote_dir)
            else:
                sftp.mkdir(remote_dir)

            put_(sftp, from_dir, remote_dir)
            execute_(ssh, '/usr/local/share/bin/auto-seg-lq', remote_dir, args.remote_file_location)

            for x in sftp.listdir(args.remote_file_location):
                if PurePosixPath(args.remote_file_location)/x != PurePosixPath(remote_dir):
                    get_(sftp, str(PurePosixPath(args.remote_file_location)/x), str(Path(to_dir)/x))
