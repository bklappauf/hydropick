#
# Copyright (c) 2014, Texas Water Development Board
# All rights reserved.
#
# This code is open-source. See LICENSE file for details.
#

from __future__ import absolute_import

import logging
import glob
import os
import warnings

import tables

from hydropick.io import survey_io
from hydropick.io.survey_io import (import_survey_line_from_file,
                                    read_survey_line_from_hdf)

logger = logging.getLogger(__name__)

# XXX this is not fully implemented


def get_name(directory):
    # name defaults to parent and grandparent directory names
    directory = os.path.abspath(directory)
    parent, dirname = os.path.split(directory)
    grandparent, parent_name = os.path.split(parent)
    great_grandparent, grandparent_name = os.path.split(grandparent)
    if parent_name and grandparent_name:
        name = grandparent_name + ' ' + parent_name
    elif parent_name:
        name = parent_name
    else:
        name = "Untitled"
    return name


def import_cores(directory, h5file):
    from ..model.core_sample import CoreSample

    try:
        core_dicts = survey_io.read_core_samples_from_hdf(h5file)
    except (IOError, tables.exceptions.NoSuchNodeError):
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1] == '.txt':
                corestick_file = os.path.join(directory, filename)
                survey_io.import_core_samples_from_file(corestick_file, h5file)

        core_dicts = survey_io.read_core_samples_from_hdf(h5file)

    # this is a corestick file
    return [
        CoreSample(
            core_id=core_id,
            location=(core['easting'], core['northing']),
            layer_boundaries=core['layer_interface_depths'],
        )
        for core_id, core in core_dicts.items()
    ]


def import_pick_files(directory, h5file):
    # find the GIS file in the directory
    for path in glob.glob(directory + '/*/*/*[pic,pre]'):
        survey_io.import_pick_line_from_file(path, h5file)


def import_lake(name, directory, h5file):
    try:
        shoreline = survey_io.read_shoreline_from_hdf(h5file)
    except (IOError, tables.exceptions.NoSuchNodeError):
        # find the GIS file in the directory
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1] == '.shp':
                shp_file = os.path.join(directory, filename)
                survey_io.import_shoreline_from_file(name, shp_file, h5file)
        shoreline = survey_io.read_shoreline_from_hdf(h5file)
    return shoreline


def import_sdi(directory, h5file):
    from hydropick.model.survey_line_group import SurveyLineGroup
    survey_lines = []
    survey_line_groups = []
    stepd = 0
    for root, dirs, files in os.walk(directory):
        group_lines = []
        Nd = len(dirs)
        Nf = len(files)
        stepf = 0
        stepd += 1
        for filename in files:
            stepf += 1
            if os.path.splitext(filename)[1] == '.bin':
                linename = os.path.splitext(filename)[0]
                print 'Reading line{}  ({}/{} of {}/{})'.format(linename,
                                                                stepf,
                                                                stepd, Nf, Nd)
                try:
                    line = read_survey_line_from_hdf(h5file, linename)
                except (IOError, tables.exceptions.NoSuchNodeError):
                    logger.info("Importing sdi file '%s'", filename)
                    try:
                        import_survey_line_from_file(os.path.join(root,
                                                                  filename),
                                                     h5file, linename)
                        line = read_survey_line_from_hdf(h5file, linename)
                    except Exception as e:
                        # XXX: blind except to read all the lines that we
                        # can for now
                        s = 'Reading file {} failed with error "{}"'
                        msg = s.format(filename, e)
                        warnings.warn(msg)
                        logger.warning(msg)
                        break
                group_lines.append(line)
        if group_lines:
            dirname = os.path.basename(root)
            group = SurveyLineGroup(name=dirname, survey_lines=group_lines)
            survey_lines += group_lines
            survey_line_groups.append(group)
    return survey_lines, survey_line_groups


def import_survey(directory, with_pick_files=False):
    """ Read in a project from the current directory-based format """
    from ..model.survey import Survey

    name = get_name(directory)

    # HDF5 datastore file for survey
    hdf5_file = os.path.join(directory, name + '.h5')
    print hdf5_file

    # read in core samples
    core_samples = import_cores(os.path.join(directory, 'Coring'), hdf5_file)

    # read in lake
    lake = import_lake(name, os.path.join(directory, 'ForSurvey'), hdf5_file)

    # read in sdi data
    survey_lines, survey_line_groups = import_sdi(os.path.join(directory,
                                                               'SDI_Data'),
                                                  hdf5_file)

    # read in edits to sdi data
    if with_pick_files:
        import_pick_files(os.path.join(directory, 'SDI_Edits'), hdf5_file)

    survey = Survey(
        name=name,
        lake=lake,
        survey_lines=survey_lines,
        survey_line_groups=survey_line_groups,
        core_samples=core_samples,
        hdf5_file=hdf5_file,
    )

    return survey
