import sys


def format_float(afloat, no_one=True, tol=1e-8, out_str=True):
    """
    This function is used to make pretty formulas by formatting the amounts.
    Instead of Li1.0 Fe1.0 P1.0 O4.0, you get LiFePO4.

    Args:
        afloat (float): a float
        no_one (bool): if true, floats of 1 are ignored.
        tol (float): Tolerance to round to nearest int. i.e. 2.0000000001 -> 2
        out_str (bool): if true, output type is str
    Returns:
        A string, int or float representation of the float for formulas.
    """
    if no_one and afloat == 1:
        return ""
    elif abs(afloat - int(afloat)) < tol:
        return str(int(afloat)) if out_str else int(afloat)
    else:
        return str(round(afloat, 8)) if out_str else round(afloat, 8)


def float_filename(afloat, float_len=5):
    """
    This function is used to format the filename with float.
    :param afloat: (float) a float
    :param float_len: (int) maximum number of the float length
    :return: A string representation of the float
    """
    if afloat < 0:
        name = "n%.{}s".format(float_len) % str(afloat).replace(".", "_")[1:]
    else:
        name = "p%.{}s".format(float_len) % str(afloat).replace(".", "_")
    return name


def str_delimited(results, header=None, delimiter="\t"):
    """
        Given a tuple of tuples, generate a delimited string form.
    >>> results = [["a","b","c"],["d","e","f"],[1,2,3]]
    >>> print(str_delimited(results, delimiter=","))
    a,b,c
    d,e,f
    1,2,3

    :param results: 2d sequence of arbitrary types.
    :param header: optional header
    :param delimiter: delimiter between each str
    :return: Aligned string output in a table-like format.
    """
    return_str = ""
    if header is not None:
        return_str += delimiter.join(header) + "\n"
    return return_str + "\n".join([delimiter.join(map(str, result))
                                   for result in results])


def find_re_pattern(pattern, content):
    outs = {}
    for k, v in pattern.items():
        out = v.findall(content)
        if len(out) > 1:
            outs[k] = out
        else:
            try:
                outs[k] = out[0]
            except IndexError:
                sys.stderr.write("No '%s' is found.\n" % k)
                outs[k] = ""
    return outs
