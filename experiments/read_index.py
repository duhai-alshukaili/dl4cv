import os
import struct
import argparse

def read_git_index(index_path):
    with open(index_path, 'rb') as f:
        header = f.read(12)
        signature, version, num_entries = struct.unpack('!4sLL', header)

        if signature != b'DIRC':
            raise ValueError('Invalid Git index file')

        print(f"Index version: {version}")
        print(f"Number of entries: {num_entries}")

        entries = []
        for _ in range(num_entries):
            entry = {}
            entry_header = f.read(62)
            (
                ctime_s, ctime_n,
                mtime_s, mtime_n,
                dev, ino, mode,
                uid, gid, size,
                sha1_1, sha1_2, sha1_3, sha1_4, sha1_5,
                flags
            ) = struct.unpack('!LLLLLLLLLLLLLLLH', entry_header)

            sha1 = b''.join(struct.pack('!L', x) for x in [sha1_1, sha1_2, sha1_3, sha1_4, sha1_5])[:20]
            entry['sha1'] = sha1.hex()

            name_len = flags & 0x0FFF

            entry['name-len'] = name_len

            path = b''
            while True:
                byte = f.read(1)
                if byte == b'\x00':
                    break
                path += byte
            entry['path'] = path.decode('utf-8')
            entries.append(entry)

            # Padding to 8-byte alignment
            entry_length = 62 + len(path) + 1
            padding = (8 - (entry_length % 8)) or 8
            f.read(padding)

        return entries

def display_index_info(git_repo_path):
    index_file = os.path.join(git_repo_path, '.git', 'index')
    if not os.path.exists(index_file):
        raise FileNotFoundError("No index file found in the provided Git repository.")

    entries = read_git_index(index_file)
    for i, entry in enumerate(entries, 1):
        print(f"{i}. Path: {entry['path']}, SHA1: {entry['sha1']} ({entry['name-len']})")

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, 
    help="path to the github local repo.")
args = vars(ap.parse_args())


# Example usage:
# Replace with the path to your local Git project
display_index_info(args['path'])
