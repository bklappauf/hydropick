#
# Copyright (c) 2014, Texas Water Development Board
# All rights reserved.
#
# This code is open-source. See LICENSE file for details.
#
"""
Define algorithm classes following the model and place the class name in the
ALGORITHM_LIST constant at the top.  Each algorithm must have a unique name
string or it will not be recognized.

"""

from __future__ import absolute_import
import logging

import numpy as np

from traits.api import provides, Str, HasTraits, Float

from .i_algorithm import IAlgorithm

logger = logging.getLogger(__name__)


ALGORITHM_LIST = [
    'ZeroAlgorithm',
    'OnesAlgorithm',
    'XDepthAlgorithm'
]


@provides(IAlgorithm)
class ZeroAlgorithm(HasTraits):
    """ A default algorithm for testing or hand drawing a new line

    """
    #: a user-friendly name for the algorithm
    name = Str('zeros algorithm')

    # list of names of traits defined in this class that the user needs
    # to set when the algorithm is applied
    arglist = []

    # instructions for user (description of algorithm and required args def)
    instructions = Str('Demo algorithm that creates a depth line at 0')

    # args (none for this algorithm)

    def process_line(self, survey_line, *args, **kw):
        """ returns all zeros to provide a blank line to edit.
        Size matches horizontal pixel number of intensity arrays
        """
        trace_array = survey_line.trace_num
        zeros_array = np.zeros_like(trace_array)
        return trace_array, zeros_array

@provides(IAlgorithm)
class OnesAlgorithm(HasTraits):
    """ A default algorithm for testing or hand drawing a new line

    """
    #: a user-friendly name for the algorithm
    name = Str('ones algorithm')

    # list of names of traits defined in this class that the user needs
    # to set when the algorithm is applied
    arglist = []

    # instructions for user (description of algorithm and required args def)
    instructions = Str('Demo algorithm that creates a depth line at 1')

    # args (none for this algorithm)

    def process_line(self, survey_line, *args, **kw):
        """ returns all zeros to provide a blank line to edit.
        Size matches horizontal pixel number of intensity arrays
        """
        trace_array = survey_line.trace_num
        depth_array = np.ones_like(trace_array)
        return trace_array, depth_array

@provides(IAlgorithm)
class XDepthAlgorithm(HasTraits):
    """ A default algorithm for testing or hand drawing a new line

    """

    #: a user-friendly name for the algorithm
    name = Str('x depth algorithm')

    # list of names of traits defined in this class that the user needs
    # to set when the algorithm is applied
    arglist = ['depth']

    # instructions for user (description of algorithm and required args def)
    instructions = Str('Demo algorithm that creates a depth line at' +
                       ' a depth set by user (defalut = 3.0)\n' +
                       'depth = float')

    # args (none for this algorithm)
    depth = Float(3.0)

    def process_line(self, survey_line):
        """ returns all zeros to provide a blank line to edit.
        Size matches horizontal pixel number of intensity arrays
        """
        depth = self.depth
        trace_array = survey_line.trace_num
        depth_array = depth * np.ones_like(trace_array)
        return trace_array, depth_array


def get_algorithm_dict():
    name_list = ALGORITHM_LIST
    classes = [globals()[cls_name] for cls_name in name_list]
    names = [cls().name for cls in classes]
    logger.debug('found these algorithms: {}'.format(names))
    return dict(zip(names, classes))
