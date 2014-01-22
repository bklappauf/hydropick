#
# Copyright (c) 2014, Texas Water Development Board
# All rights reserved.
#
# This code is open-source. See LICENSE file for details.
#

from __future__ import absolute_import

# std library
from copy import deepcopy

# other imports
import numpy as np

# ETS imports
from traits.api import Instance, HasTraits, Array, Property, Float,\
                       List, Str, Bool,Tuple, Dict,\
                       DelegatesTo, Event

# Local imports
from ..model.survey_line import SurveyLine


class SurveyDataSession(HasTraits):
    """ Model for SurveyLineView.

    Assumes reciept of SurveyLine instance
    (Make sure surveyline has the traits delegated below from sdi dict )
    """
    # Source of survey line data to be edited
    surveyline = Instance(SurveyLine)

    # Flag to be set when valid data is passed to surveyline
    data_available = Bool(False)

    #: sample locations, an Nx2 array of lat/long (or easting/northing?)
    locations = DelegatesTo('surveyline', 'locations')

    # Easting/Northing (x,y on map-plane in meters? ). Should probably be
    # incorporated into locations attribute which could be a dictionary of
    # types of coordinates mapped to pixel values.
    E_N_positions = Property(depends_on=['surveyline.interpolated_northing',
                                         'surveyline.interpolated_easting'])

    #: a dictionary mapping frequencies to intensity arrays
    # NOTE:  assume arrays are transposed so that img_plot(array)
    # displays them correctly and array.shape gives (xsize,ysize)

    frequencies = Property(Dict, depends_on='surveyline.frequencies')

    #: relevant core samples
    core_samples = DelegatesTo('surveyline', 'core_samples')

    #: depth of the lake at each location as generated by various soruces
    lake_depths = DelegatesTo('surveyline')

    # and event fired when the lake depths are updated
    lake_depths_updated = Event

    #: pre-impoundment depth at each location as generated by various soruces
    preimpoundment_depths = DelegatesTo('surveyline', 'preimpoundment_depths')

    # and event fired when the lake depth is updated
    preimpoundment_depths_updated = Event

    #: Dictionary of all depth lines. Allows editor easy access to all lines.
    depth_dict = Property(Dict,
                          depends_on=['lake_depths', 'preimpoundment_depths',
                                      'lake_depths_items',
                                      'preimpoundment_depths_items']
                          )

    # Keys of depth_dict provides list of target choices for line editor
    target_choices = Property(depends_on='depth_dict')

    # Selected target line key from depth dict for editing
    selected_target = Str

    # Keys of frequencies dictionary.
    freq_choices = Property(List, depends_on='frequencies')

    # Selected freq key from frequencies dict for displaying image.
    selected_freq = Str

    # Y bounds should be set based on depth per pixel value of image data.
    # Y axis of depth lines should be set to match this value.
    ybounds = Property(Tuple, depends_on=['pixel_depth_offset',
                                          'pixel_depth_scale',
                                          'frequencies'])

    pixel_depth_offset = DelegatesTo('surveyline', 'draft')
    pixel_depth_scale = DelegatesTo('surveyline', 'pixel_resolution')

    # Array to be used for x axis.  Length corresponds to depth lines and
    # image horizontal sizes.  Default is index but may be changed to
    # various actual distances.  Defines xbounds.
    x_array = Property(Array)

    # xbounds used for image display (arguably could be in view class)
    xbounds = Property(Tuple, depends_on=['frequencies', 'frequencies_items'])

    ymax = Float(0)
    #==========================================================================
    # Defaults
    #==========================================================================

    def _surveyline_default(self):
        return SurveyLine()

    def _selected_freq_default(self):
        return self.frequencies.keys()[0]

    #==========================================================================
    # Notifications
    #==========================================================================

    def _surveyline_changed(self, new):
        ''' Assumes any non-None value will be a valid SurveyLine object
        In order to maintain valid delgates, when None is passed to surveyline
        we change it to an empty SurveyLine object
        '''
        if new is None:
            self.surveyline = SurveyLine()
            self.data_available = False
        else:
            self.data_available = True

    #==========================================================================
    # Get/Set
    #==========================================================================
    def _get_freq_choices(self):
        ''' Get list of available frequencies as (value,string) pair from
        frequencies dict for use in selector widget.
        Limit label string resolution to 0.1 kHz.
        '''
        s = [freq for freq in self.frequencies]
        return s

    def _get_E_N_positions(self):
        return np.array([self.surveyline.interpolated_easting,
                         self.surveyline.interpolated_northing]).T

    def _get_frequencies(self):
        new_dict = deepcopy(self.surveyline.frequencies)
        return new_dict

    def _get_depth_dict(self):
        ''' Combine lake depths and preimpoundment in to one dict.
        '''
        depth_dict = {}
        depth_dict.update(self.lake_depths)
        depth_dict.update(self.preimpoundment_depths)
        return depth_dict

    def _get_target_choices(self):
        ''' Get list of available frequencies as strings from frequencies dic
        limit resolution to 0.1 kHz.
        '''
        return self.depth_dict.keys()

    def _get_x_array(self):
        ''' Initially set as horizontal pixel number of arbitrary image'''
        N = self.frequencies.values()[0].shape[1]
        return np.arange(N)

    def _get_xbounds(self):
        bounds = (self.x_array.min(), self.x_array.max())
        return bounds

    def _get_ybounds(self):
        N = self.frequencies.values()[0].shape[0]
        min = np.mean(self.pixel_depth_offset)
        max = min + N * self.pixel_depth_scale
        return (min, max)

if __name__ == '__main__':
    pass
