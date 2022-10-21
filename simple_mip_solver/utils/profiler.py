import cProfile
import pstats
from functools import wraps
import linecache
import os
import tracemalloc


def profile_run_time(run_time_profile_file=None, sort_by='cumulative',
                     lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        run_time_profile_file: str, pathlike, or None. Default is None.
            Path of the output file. If only name of the file
            is given, it's saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = run_time_profile_file or kwargs.get('run_time_profile_file') \
                           or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


def profile_memory(memory_profile_file=None, key_type='lineno', limit=3, unit='KB'):
    """decorator that profiles memory usage throughout the decorated function

    :param memory_profile_file: str, pathlike, or None. Default is None.
    Path of the output file. If only name of the file is given, it's saved in
    the current directory.
    :param key_type: what attribute to sort on
    :param limit: how many lines to show
    :param unit: what unit to show data usage in. expects KB, MB, GB, or TB
    :return:
    """

    def inner(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            power_dict = {'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4}
            assert unit in power_dict, 'please provide appropriate memory unit'
            power = power_dict[unit]
            
            _output_file = memory_profile_file or kwargs.get('memory_profile_file') \
                           or func.__name__ + '.prof'
            tracemalloc.start()

            retval = func(*args, **kwargs)

            snapshot = tracemalloc.take_snapshot()
            snapshot = snapshot.filter_traces((
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            ))
            top_stats = snapshot.statistics(key_type)
            with open(_output_file, 'w') as f:
                f.write(f"Top {limit} lines \n")
                for index, stat in enumerate(top_stats[:limit], 1):
                    frame = stat.traceback[0]
                    # replace "/path/to/module/file.py" with "module/file.py"
                    filename = os.sep.join(frame.filename.split(os.sep)[-2:])
                    f.write(f"#{index}: {filename}:{frame.lineno}: "
                            f"{'{:.2f}'.format(stat.size / (1024**power))} {unit} \n")
                    line = linecache.getline(frame.filename, frame.lineno).strip()
                    if line:
                        f.write(f'    {line} \n')

                other = top_stats[limit:]
                if other:
                    size = sum(stat.size for stat in other)
                    f.write(f"{len(other)} other: {'{:.2f}'.format(size / (1024**power))} {unit} \n")
                total = sum(stat.size for stat in top_stats)
                f.write(f"Total allocated size: {'{:.2f}'.format(total / (1024**power))} {unit} \n")

            return retval

        return wrapper

    return inner