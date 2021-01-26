import pandas as pd
import numpy as np
from os import listdir
from os.path import join, isdir
import re
import argparse
from pathlib import Path

PATTERN_PCI = "\d\d:\s(.+)$"
PATTERN_ACPI = "\u0020{4}\w{4}:\u0020(.{47})"


def reformat(mode, path, settings_path):
    pci_values, acpi_values, labels = read_data(mode, path)
    settings = receive_settings(mode, settings_path)

    result_pcis, max_len_pci = get_normalised_values(pci_values, int(settings[0]))
    result_acpis, max_len_acpi = get_normalised_values(acpi_values, int(settings[1]))
    result_array = collect_result(mode, result_pcis, result_acpis, labels)

    save_setting(mode, max_len_pci, max_len_acpi)
    save_data(mode, result_array)


def save_data(mode, result_array):
    if mode == "train":
        filename = 'additional_data/data_train.csv'
    else:
        filename = 'additional_data/data_test.csv'
    df = pd.DataFrame(result_array)
    df.to_csv(filename, index=False, header=False)
    print("Result csv file:  " + str(Path().absolute()) + "/" + filename)


def collect_result(mode, result_pcis, result_acpis, labels):
    result_array = []
    for i in range(len(result_pcis)):
        if mode == "train":
            result_array.append([labels[i]] + result_pcis[i] + result_acpis[i])
        else:
            result_array.append(result_pcis[i] + result_acpis[i])
    return result_array


def save_setting(mode, max_len_pci, max_len_acpi):
    if mode == "train":
        with open("additional_data/model.conf", "w") as outfile:
            outfile.write("\n".join([str(max_len_pci), str(max_len_acpi)]))
            print("Result settings file:  " + str(Path().absolute()) + "/additional_data/model.conf")


def find_required_parts_pci(lines):
    data = []
    for line in lines:
        if re.match(PATTERN_PCI, line):
            line = line[4:-1]
            [data.append(int(x, 16)) for x in line.split(' ')]
    return data


def find_required_parts_acpi(lines):
    data = []
    for line in lines:
        if re.match(PATTERN_ACPI, line):
            line = line[10:57]
            for x in line.split(' '):
                if len(x) != 0 or not x.strip() or x != "\n":
                    try:
                        data.append(int(x, 16))
                    except:
                        pass
    return data


def get_normalised_values(input_values, model_max_len=-1):
    result_values = []
    if model_max_len != -1:
        max_len = model_max_len
    else:
        max_len = len(max(input_values, key=len))

    for input_value in input_values:
        result_to_append = [len(input_value)] + input_value
        if max_len > len(input_value):
            result_to_append += list(np.zeros(max_len - len(input_value), dtype=np.int16))
        elif max_len < len(input_value):
            result_to_append = result_to_append[0:max_len + 1]
        result_values.append(result_to_append)
    return result_values, max_len


def receive_settings(mode, settings_path):
    if mode == "test" and settings_path is not None:
        settings = open(settings_path, "r").read().split('\n')
    else:
        settings = ["-1", "-1"]
    return settings


def read_data(mode, path):
    directories = [f for f in listdir(path) if isdir(join(path, f))]
    pci_values = []
    acpi_values = []
    labels = []

    for dir in directories:
        with open(join(path, dir) + "/pci.txt", "r") as file:
            content = file.readlines()
            pci_values.append(find_required_parts_pci(content))

        with open(join(path, dir) + "/acpi.txt", "r") as acpi_file:
            content = acpi_file.readlines()
            acpi_values.append(find_required_parts_acpi(content))

        if mode == "train":
            with open(join(path, dir) + "/label.txt", "r") as label:
                content = label.readline()
                if int(content) not in [0, 1]:
                    print("Wrong label format in file")
                    raise Exception
                labels.append(int(content))

    return pci_values, acpi_values, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="mode should be train or test", type=str)
    parser.add_argument("path", help="path to the directory should be specified", type=str)
    parser.add_argument("settings_path", help="path to the directory should be specified", nargs='?', default=None,
                        type=str)
    args = parser.parse_args()
    if args.mode not in ["test", "train"]:
        print("Wrong mode argument value. Required value: train or test")
    else:
        reformat(args.mode, args.path, args.settings_path)
