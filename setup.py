#!/usr/bin/env python
from __future__ import print_function

import glob
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Thanks to http://patorjk.com/software/taag/
logo = r"""
 ______   __  __     ______     __    __                                             
/\  == \ /\ \_\ \   /\  __ \   /\ "-./  \                                            
\ \  _-/ \ \____ \  \ \  __ \  \ \ \-./\ \                                           
 \ \_\    \/\_____\  \ \_\ \_\  \ \_\ \ \_\                                          
  \/_/     \/_____/   \/_/\/_/   \/_/  \/_/                                          
                                                                                     
 ______     __   __     ______     __         __  __     ______     __     ______    
/\  __ \   /\ "-.\ \   /\  __ \   /\ \       /\ \_\ \   /\  ___\   /\ \   /\  ___\   
\ \  __ \  \ \ \-.  \  \ \  __ \  \ \ \____  \ \____ \  \ \___  \  \ \ \  \ \___  \  
 \ \_\ \_\  \ \_\\"\_\  \ \_\ \_\  \ \_____\  \/\_____\  \/\_____\  \ \_\  \/\_____\ 
  \/_/\/_/   \/_/ \/_/   \/_/\/_/   \/_____/   \/_____/   \/_____/   \/_/   \/_____/ 
                                                                                     
"""

INFO = {
    'version': '0.1.0',
}


def main():
    print(logo)

    packages = [
        'pyam_analysis',
    ]
    pack_dir = {
        'pyam_analysis': 'pyam_analysis',
    }
    entry_points = {
        'console_scripts': [
            # list CLIs here
        ],
    }
    setup_kwargs = {
        "name": "pyam_analysis",
        "version": INFO['version'],
        "description": 'Analyze Integrated Assessment Model Results'
        'Trajectories',
        "author": 'Matthew Gidden & Daniel Huppmann',
        "author_email": 'matthew.gidden@gmail.com & huppmann@iiasa.ac.at',
        "url": 'http://github.com/iiasa/pyam_analysis',
        "packages": packages,
        "package_dir": pack_dir,
        "entry_points": entry_points,
        "zip_safe": False,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
