import os
import yaml
import pickle
import io
import csv

# the default project root
os.environ['PROJECT_ROOT'] = os.path.abspath(os.path.join(__file__, '../../../'))


def safe_makedir(directory):
    """
    Safely create a directory
    :param directory: a string indicating the path
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def before_save(file_or_dir):
    """
    make sure that the dedicated path exists (create if not exist)
    :param file_or_dir:
    :return:
    """
    dir_name = os.path.dirname(os.path.abspath(file_or_dir))
    safe_makedir(dir_name)


def get_path(path, file_name=None, absolute=False):
    """
    A helper function that get the real/abs path of a file on disk, with the project dir as the base dir.
    Note: there is no checking on the illegality of the args!
    :param path: a relative path to base_dir, optional file_name to use
    :param file_name: an optional file name under the path
    :param absolute: return the absolute path
    :return: return the path relative to the project root dir, default to return relative path to the called place.
    """
    _p = os.path.join(os.environ['PROJECT_ROOT'], path)
    if file_name:
        _p = os.path.join(_p, file_name)
    if absolute:
        return os.path.abspath(_p)
    return os.path.relpath(_p)


def yaml2dict(file_name, mode='r'):
    with open(file_name, mode) as f:
        d = yaml.safe_load(f)
    return d


def lists2csv(lists, file_path, header=None, delimiter=',', encoding=None):
    with io.StringIO() as s_io:
        writer = csv.writer(s_io, delimiter=delimiter)
        if header is not None:
            writer.writerow(header)
        for ls in lists:
            writer.writerow([str(i) for i in ls])
        write2file(s_io, file_path, 'w', encoding=encoding)


def write2file(s_io, file_path, mode, encoding=None):
    """
    This is a wrapper function for writing files to disks,
    it will automatically check for dir existence and create dir or file if needed
    :param s_io: a io.StringIO instance or a str
    :param file_path: the path of the file to write to
    :param mode: the writing mode to use
    :return: None
    """
    before_save(file_path)
    with open(file_path, mode, encoding=encoding) as f:
        if isinstance(s_io, io.StringIO):
            f.write(s_io.getvalue())
        else:
            f.write(s_io)


def obj2pkl(obj, file_name):
    before_save(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def pkl2obj(file_name):
    with open(file_name, 'rb') as f:
        results = pickle.loads(f.read())
        return results


def maybe_calculate(file_name, cal_fn, *args, **kwargs):
    """
    Check whether a cached .pkl file exists.
    If exists, directly load the file and return,
    Else, call the `cal_fn`, dump the results to .pkl file specified by `filename`, and return the results.
    :param filename: the name of the target cached file
    :param cal_fn: a function that maybe called with `*args` and `**kwargs` if no cached file is found.
    :return: the pickle dumped object, if cache file exists, else return the return value of cal_fn
    """
    if os.path.isfile(file_name):
        print("Reading from tmp file", file_name)
        results = pkl2obj(file_name)
    else:
        results = cal_fn(*args, **kwargs)
        obj2pkl(results, file_name)
    return results


if __name__ == "__main__":
    print("The default project root is:", os.environ['PROJECT_ROOT'])