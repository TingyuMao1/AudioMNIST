import os


def filter_data(from_path="../tidigits_flac/data", to_path="../data"):
    """
    Filter original data, only use 11 isolated digits.
    Read raw data from @from_path, and write filtered data to @to_path
    """
    types = ("train", "test", "valid")
    if not os.path.exists(to_path):
        os.mkdir(to_path)
        for t in types:
            os.mkdir(os.path.join(to_path, t))

    for folder in os.listdir(from_path):
        for t in types[:2]:
            idx = [1] * 11
            path = os.path.join(from_path, folder, t)
            for sub_folder in os.listdir(path):
                sub_path = os.path.join(path, sub_folder)
                for people in os.listdir(sub_path):
                    dest = os.path.join(sub_path, people)
                    for f in os.listdir(dest):
                        name, ext = os.path.splitext(f)
                        if len(name) == 2:
                            label = name[0]
                            if label == "z":
                                label = 0
                            elif label == "o":
                                label = 10
                            else:
                                label = int(label)

                            if t == "test":
                                type_ = "valid" if name[1] == "a" else "test"
                            else:
                                type_ = t

                            # copy to @to_path
                            cmd = "cp {} {}".format(os.path.join(dest, f), os.path.join(to_path, type_))
                            os.system(cmd)
                            new_name = "{}_{}{}".format(name[0], idx[label], ext)
                            cmd = "mv {} {}".format(os.path.join(to_path, type_, f), 
                                    os.path.join(to_path, type_, new_name))
                            os.system(cmd)
                            idx[label] += 1


filter_data()
