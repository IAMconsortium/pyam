#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import shutil

from setuptools import setup, Command, find_packages
from setuptools.command.install import install

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


class Cmd(install):
    """Custom clean command to tidy up the project root."""

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)
        dirs = [
            'pyam_analysis.egg-info',
            'build',
        ]
        for d in dirs:
            print('removing {}'.format(d))
            shutil.rmtree(d)


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
    package_data = {
        'pyam_analysis': ['region_mappings/*'],
    }
    cmdclass = {
        'install': Cmd,
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
        "cmdclass": cmdclass,
        "package_data": package_data,
    }
    rtn = setup(**setup_kwargs)


if __name__ == "__main__":
    main()
