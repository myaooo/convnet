import os
import yaml

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


if __name__ == "__main__":
    print("The default project root is:", os.environ['PROJECT_ROOT'])