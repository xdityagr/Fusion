"""
utils.py : Utility file for fusion, to add *colors* or format the debugging info (Redundant for public release, no debugging avaibale in it.)
Developed by Aditya Gaur
Github : @xdityagr 
Email  : adityagaur.home@gmail.com 
"""
from colorama import Fore
def accent(string, custom_color=None, bold=False):
    colors = {
        'RED': '\033[31m',
        'ORANGE': '\033[38;5;208m',
        'YELLOW': '\033[33m',
        'GREEN': '\033[32m',
        'BLUE': '\033[34m'
    }
    
    color_code = colors.get(custom_color.upper() if custom_color else None, custom_color if custom_color else None)
    if color_code is None: color_code = colors.get('ORANGE')
    
    style_code = '\033[1m' if bold else ''
    reset_code = '\033[0m'

    return f"{style_code}{color_code}{string}{reset_code}"+Fore.RESET

def visualize_ast(node, indent=0, is_last=True, prefix=""):
    """Used to visualize an AST node and its children in a tree-like structure."""
def get_ast_docstring(parser_result): 
    """get docstring repr of visualisation"""
def print_debug_info(db_info): pass