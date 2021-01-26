import datetime
import os
import subprocess
import argparse

M_S = '%Y-%m-%d_%H:%M:%S'


def collect(path):
    dir_name = path + "/data_" + datetime.datetime.now().strftime(M_S)
    os.mkdir(dir_name, 0o777)
    os.chmod(dir_name, 0o777)
    with open(str(dir_name) + "/pci.txt", "wb") as file:
        file.write(subprocess.check_output(["lspci", "-x"]))
    with open(str(dir_name) + "/acpi.txt", "wb") as file:
        file.write(subprocess.check_output(["acpidump"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the directory where to save data should be specified", type=str)
    args = parser.parse_args()
    collect(args.path)
