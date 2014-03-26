#
# Copyright (c) 2014, Texas Water Development Board
# All rights reserved.
#
# This code is open-source. See LICENSE file for details.
#

from __future__ import absolute_import

from copy import deepcopy
import logging
import numpy as np

# ETS imports
from traits.api import (Instance, Event, Str, Property, HasStrictTraits, Int,
                        on_trait_change, Bool, Supports, List, Dict,
                        DelegatesTo, cached_property)
from traitsui.api import (View, VGroup, HGroup, Item, UItem, EnumEditor,
                          TextEditor, ListEditor, ButtonEditor, Label, Spring)

# Local imports
from ..model.depth_line import DepthLine
from ..model.i_survey_line_group import ISurveyLineGroup
from ..model.i_survey_line import ISurveyLine
from ..model.i_algorithm import IAlgorithm
#from .algorithm_presenter import AlgorithmPresenter
from .depth_line_presenters import (AlgorithmPresenter, NotesEditorPresenter,
                                    ApplyToGroupSettingsPresenter,
                                    DeletePresenter)
from .survey_data_session import SurveyDataSession
from .survey_views import MsgView

logger = logging.getLogger(__name__)

UPDATE_ARRAYS_TOOLTIP = (
    'updates array data in form but does not apply to line')
APPLY_TOOLTIP = (
    'applies current setting to line')
APPLY_TO_GROUP_TOOLTIP = (
    'applies current settings to all selected lines')

MODEL_TRAITS_TO_SAVE_ON_CHANGE = (
    'name, line_type, color, locked, notes')

CURRENT_SURFACE_FROM_BIN_NAME = 'current_surface_from_bin'
CURRENT_SURFACE_FROM_BIN_LABEL = 'POST_' + CURRENT_SURFACE_FROM_BIN_NAME


class DepthLineView(HasStrictTraits):
    """ View Class for working with survey line data to find depth profile.

    Uses a Survey class as a model and allows for viewing of various depth
    picking algorithms and manual editing of depth profiles.
    """

    #==========================================================================
    # Traits Attributes
    #==========================================================================

    # current data session with relevant info for the current line
    data_session = Instance(SurveyDataSession)

    # name of current line in editor: convenience for display
    survey_line_name = Property(depends_on=['data_session'])

    # name of hdf5_file for this survey in case we need to load survey lines
    hdf5_file = Str

    # list of available depth lines extracted from survey line for choices
    depth_lines = Property(depends_on=['data_session',
                                       'data_session.depth_lines_updated'])
    #                           cached=True)

    # name of depth_line to view chosen from pulldown of line choices.
    selected_depth_line_name = Str

    # current depth line object
    model = Instance(DepthLine)

    # convenience to disable widgets when model locked
    locked = DelegatesTo('model')

    # arrays to plot sizes: used to check they are filled and equal
    index_array_size = Property(Int, depends_on=['model.index_array, model'])
    depth_array_size = Property(Int, depends_on=['model.depth_array, model'])

    # source is edited as local trait then changed in model with apply
    # this is needed so that none of the traits modifying the model data will
    # be updated in the model until the data is actually updated.
    source = Str
    source_choices = Property(depends_on='selected_depth_line_name')

    # create local traits so that these options can be dynamically changed
    source_name = Str
    source_names = Property(depends_on=['source'])

    # Set of selected survey lines (including groups) to apply algorithm to
    selected_survey_lines = List(Supports(ISurveyLine))

    # list of selected groups and lines by name str for information only
    selected = Property(List, depends_on=['current_survey_line_group',
                                          'selected_survey_lines'])

    # dict of available algorithms
    algorithms = Dict

    # convenience property for getting algorithm arguments
    alg_arg_dict = Property()

    # currently configured algorithm: used as model for alg edit dialog
    current_algorithm = Supports(IAlgorithm)

    ##### BOOLS / FLAGS #######################################################
    # convenience condition for functions related to binary sourced line
    on_bin_line = Property(Bool, depends_on='selected_depth_line_name')

    # flag allows line creation/edit to continue in apply method
    no_problem = Bool(False)

    # determines whether to show the list of selected groups and lines
    show_selected = Bool(False)

    ##### BUTTONS FOR THE VIEW ################################################

    # applys settings to DepthLine updating object and updating survey line
    apply_button = Event

    # applys settings each survey line in selected lines
    apply_to_group = Event

    # applys settings each survey line in selected lines
    delete = Event

    # button to open algorithm configure dialog
    configure_algorithm = Event()
    configure_algorithm_done = Event()

    # used to edit text for the DepthLine notes trait. For some reason
    # custom style will not update the extended trait designated trait so we
    # need to make a local trait and update it separately.
    edit_notes = Event
    notes = Str

    # currently used but not needed as trait.  Eventually this could be
    # converted to trait that saves initial line settings when the model is
    # changed to allow undoing changes
    current_dline = Instance(DepthLine)

    #### DEPRECATED - LEAVE IN JUST IN CASE####################################

    # currently selected group (#### DEPRECATED - LEAVE IN JUST IN CASE)
    current_survey_line_group = Supports(ISurveyLineGroup)

    #==========================================================================
    # Define Views
    #==========================================================================

    traits_view = View(
        VGroup(Item('survey_line_name', style='readonly'),
               HGroup(Item('show_selected', label='Selected(show)'),
                      UItem('selected',
                            editor=ListEditor(style='readonly'),
                            style='readonly',
                            visible_when='show_selected'),
                      show_border=True
                      ),
               Item('selected_depth_line_name', label='View Depth Line',
                    editor=EnumEditor(name='depth_lines')),
               Item('_'),
               padding=5
               ),
        Label('Line Traits', emphasized=True),
        VGroup(Item('object.model.survey_line_name', style='readonly'),
               Item('object.model.name', editor=TextEditor(auto_set=False),
                    visible_when='selected_depth_line_name=="New Line"',
                    enabled_when='not locked'),
               Item('object.model.line_type', enabled_when='not locked'),
               Item('object.model.color',
                    enabled_when='not locked or on_bin_line'),
               Item('object.model.locked',
                    enabled_when='not on_bin_line'),
               Item('object.model.notes', editor=TextEditor(read_only=True),
                    style='custom', height=30, resizable=True),
               Item('edit_notes', editor=ButtonEditor(label='Edit Notes'),
                    enabled_when='not locked'),
               ),
        Label('Line Data', emphasized=True),
        VGroup(Item('object.model.edited', style='readonly'),
               Item('source', editor=EnumEditor(name='source_choices'),
                    enabled_when='not locked'),
               Item('source_name',
                    editor=EnumEditor(name='source_names'),
                    enabled_when='not locked'),
               UItem('configure_algorithm',
                     editor=ButtonEditor(label='Configure Algorithm'),
                     visible_when=('source == "algorithm"' +
                                   ' and not current_algorithm'),
                     enabled_when='not locked'),
               UItem('configure_algorithm_done',
                     editor=ButtonEditor(label='Configure Algorithm (DONE)'),
                     visible_when=('current_algorithm'),
                     enabled_when='not locked'),
               ),
        # these are the buttons to control this pane
        HGroup(UItem('apply_button',
                     editor=ButtonEditor(label='Apply'),
                     enabled_when='not locked',
                     tooltip=APPLY_TOOLTIP),
               UItem('apply_to_group',
                     editor=ButtonEditor(label='Apply To Group'),
                     enabled_when='not locked',
                     tooltip=APPLY_TO_GROUP_TOOLTIP),
               Spring(),
               UItem('delete',
                     editor=ButtonEditor(label='Delete'),
                     enabled_when='not locked',)
               ),
        height=500,
        resizable=True,
    )

    #==========================================================================
    # Defaults
    #==========================================================================

    def _selected_depth_line_name_default(self):
        ''' provide initial value for selected depth line in view'''
        return 'New Line'

    #==========================================================================
    # Notifications or Callbacks
    #==========================================================================

    @on_trait_change('configure_algorithm, configure_algorithm_done')
    def show_configure_algorithm_dialog(self):
        ''' gets/creates current algorithm object and opens configure dialog
        when buttons are pressed which edits current alg object.
        Buttons show up when alg is selected.
        '''
        alg_name = self.source_name
        logger.debug('configuring alg: {}. Alg exists={}, model args={}'
                     .format(alg_name, self.current_algorithm, self.model.args))
        if self.current_algorithm is None:
            self.set_current_algorithm(alg_name=alg_name)
        view = AlgorithmPresenter(algorithm=self.current_algorithm)
        # now edit alg args.
        view.configure_traits()
        # this needs to be reset to apply changes even though same object
        self.current_algorithm = view.algorithm
        if self.current_algorithm:
            logger.debug('configured alg: {} with args={}'
                         .format(alg_name, self.alg_arg_dict))

    def update_model_args(self):
        ''' current algorithm or its arguments have changed
        -  this updates the model.args values to match algorithm args
        -  this zeros out data arrays since the change implies new data
        -  this is done to self.model which will either be the target
           of the data update or will be a template for other depth lines.
        '''
        alg = self.current_algorithm
        if alg:
            if self.model.args == self.alg_arg_dict:
                # no change to data
                pass
            else:
                logger.debug('updating model with args {}'
                             .format(self.alg_arg_dict))
                self.model.args = self.alg_arg_dict
                self.zero_out_array_data()

    @on_trait_change('model:[{}]'.format(MODEL_TRAITS_TO_SAVE_ON_CHANGE))
    def save_depth_line_changes(self, obj, name, old, new):
        ''' if any of these traits are changed on an existing line
        The values will be saved to the surveyline and to disk
        Otherwise it is a new line and values will be save when edit is
        complete and apply is pressed.  The colon in the extended name
        trait of the listener is critical to prevent changing the
        model from messing up the action of the UI widget.
        '''

        if self.on_bin_line:
            if name != 'color':
                # this check should never be true since all other widgets
                # should not be enabled
                self.log_problem('cannot change surface line from original' +
                                 ' data, except for the color. Resetting ' +
                                 'value and selecting New Line')
                # reset trait.  Cannot reset selected line because we
                # cannot set the named traits widget inside its handler.
                dline = self.model
                trait_name = name
                prev_value = old
                # going to New Line also resets no_problem to True
                self.selected_depth_line_name = 'New Line'
                setattr(dline, trait_name, prev_value)

        if self.selected_depth_line_name != 'New Line' and self.no_problem:
            # editing existing line so save changes to disk
            survey_line = self.data_session.survey_line
            match = self.model.survey_line_name == survey_line.name
            if match:
                logger.debug('saved change to {} : from {} to {} '
                             .format(name, old, new))
                self.save_model_to_surveyline(model=self.model,
                                              survey_line=survey_line)
            else:
                logger.warning('changes not saved. data session does not' +
                               ' match depth line survey line name')
        else:
            logger.debug('trait {} not updated.  Selected line is {}'
                         .format(name, self.selected_depth_line_name))

    @on_trait_change('edit_notes')
    def note_edit_dialog(self, new):
        """ opens dialog to edit notes, then saves to model"""
        view = NotesEditorPresenter(notes=self.model.notes)
        view.edit_traits()
        self.model.notes = view.notes

    @on_trait_change('apply_button')
    def apply_to_current(self):
        """ applies current settings to a current model and survey line being
        edited. modifies data, traits, and saves to disk """
        self.apply_to_line()

    @on_trait_change('delete')
    def delete_current(self):
        """ delete current depth line from survey line """
        view = DeletePresenter(answer='No')
        view.configure_traits(kind='modal')
        if view.answer == 'Yes':
            model = self.model
            survey_line = self.data_session.survey_line
            if model.line_type == 'current surface':
                survey_line.lake_depths.pop(model.name)
            else:
                survey_line.preimpoundment_depths.pop(model.name)
            self.selected_depth_line_name = 'New Line'
            self.update_plot()
            survey_line.save_to_disk()

    @on_trait_change('apply_to_group')
    def apply_to_selected(self, new):
        ''' Apply current settings to all selected survey lines

        the will step through selected lines list and
        - check that valid algorithm selected
        - This currently overwrites any lines with same name
        - check if line is approved (apply?)
        - check if line is bad
        - create line with name and algorithm, args color etc.
        - apply data and apply to make line
        - set as final (?)
        '''
        # reset no_problem flag assuming user is ready to apply settings
        self.no_problem = True
        # list of selected lines
        selected = self.selected_survey_lines

        # self.check name is valid. Checking if used is separate check.
        self.validate_name(self.model)

        # check that algorithm is selected and valid and configured
        # (for current model and line.  This model is template for others)
        # Apply to group only makes sense for algorithms. This will be
        # rechecked for each line but this prevents errors in those checks
        if self.no_problem:
            # update model and check algorithm is consistent
            self.model.source = self.source
            self.model.source_name = self.source_name
            self.update_model_args()
            self.check_alg_ready()

        if self.no_problem:
            # log parameters
            self.log_model_params(lines=selected, model=self.model)

            # have user verify overwrite options
            view = ApplyToGroupSettingsPresenter()
            view.edit_traits()
            overwrite_name = view.overwrite_name
            overwrite_locked = view.overwrite_locked
            overwrite_approved = view.overwrite_approved
            new_name = view.new_name

            # apply to each survey line
            good_lines = [line for line in self.selected_survey_lines
                          if line.status != 'bad']

            for line in good_lines:
                if line.trace_num.size == 0:
                    # need to load line
                    line.load_data(self.hdf5_file)
                if line.status == 'approved' and not overwrite_approved:
                    self.log_problem('line {} already approved and overwrite'
                                     .format(line.name) + 'not selected' +
                                     'make a note: unapprove and redo later' +
                                     ' if desired')
                if self.no_problem:
                    # create new deep copy of model object for each survey line
                    model = deepcopy(self.model)
                    # deep copy passes reference.  need to empty array.
                    model.depth_array = np.array([])
                    model.index_array = np.array([])
                    # set new survey line for this model
                    model.survey_line_name = line.name
                    if new_name:
                        model.name = new_name
                    # apply the algorithm to this line. Resets no_problem.
                    self.apply_to_line(model=model,
                                       survey_line=line,
                                       overwrite_name=overwrite_name,
                                       overwrite_locked=overwrite_locked)
                else:
                    # continue with remaining lines
                    self.no_problem = True
        else:
            # there was a problem.  User should correct based on messages
            # and retry.  Reset no problem flag so user can continue.
            self.no_problem = True

    @on_trait_change('selected_depth_line_name')
    def change_depth_line(self, new):
        ''' selected line has changed so use the selection to change the
        current model to selected or create new one if New Line'''
        if new != 'New Line':
            # Existing line: get existing line object to edit
            self.current_dline = self.data_session.depth_dict[new]
        else:
            # New Line is Selected. Create empty template
            self.current_dline = self.create_new_line()

        self.model = self.current_dline
        self.source = self.model.source
        self.source_name = self.model.source_name
        self.current_algorithm = None
        self.no_problem = True
        logger.debug('change model to "{}", id: {}'
                     .format(self.model.name, id(self.model)))

    #==========================================================================
    # Helper functions
    #==========================================================================

    def apply_to_line(self, model=None, survey_line=None,
                      overwrite_name=False, overwrite_locked=False):
        ''' update data with current source selection and save all settings to
        appropriate dictionary in survey line.

        If called from apply_to_selected some checks are repeated.

        Overwrite for just editing current model should be false
        (user should select existing depth line to edit), but can be set to
        true for apply to group method.
        '''
        # reset no_problem flag assuming user is ready to apply settings
        self.no_problem = True

        if survey_line is None:
            survey_line = self.data_session.survey_line
        if model is None:
            # editing current model
            self.update_model_args()
            model = self.model
            model.source = self.source
            model.source_name = self.source_name
            if self.selected_depth_line_name != 'New Name':
                overwrite_name = True

        # self.check name is valid
        self.validate_name(model)

        # if depth line exists with same name for survey line, get object
        existing = self.check_if_name_is_used(model, survey_line=survey_line)

        # if not existing : continue to update
        # if existing check overwrite name and overwrite lock
        if (self.no_problem and existing is not None):
            # check if write name allowed
            if not overwrite_name:
                s = ('name already used on line {} and overwrite not' +
                     ' allowed. To overwrite on current survey line, ' +
                     'select that line, and edit it, then reapply settings.' +
                     ' If this is apply to group, then you must select ' +
                     'overwrite name when applying'.format(survey_line.name))
                self.log_problem(s)
            elif not overwrite_locked:
                # existing line with same name is locked so do not write
                s = ('line {} on survey line {} is locked and overwrite' +
                     ' locked is not checked (for apply to group). Check it' +
                     ' next time if you want to overwrite this, or unlock line'
                     .format(model.name, survey_line.name))
                self.log_problem(s)

        # now update array data
        if self.no_problem:
            self.update_arrays(model=model, survey_line=survey_line)

        if self.no_problem:
            self.save_model_to_surveyline(model=model, survey_line=survey_line)
        else:
            # notify user of problem again and reset no problem flag
            s = '''Could not make/change line.  Check log for details'''
            self.log_problem(s)
            self.no_problem = True

    def update_arrays(self, model=None, survey_line=None):
        ''' apply chosen method to fill line arrays
        assumes caller has already checked that writing is allowed
        (not locked, not current_line_from_binary, overwrite ok)
        This will update the arrays on the current self.model object,
        or the given model object,
        '''
        if model is None:
            model = self.model

        logger.debug('updating array data on {}'.format(model.name))

        if model.source == 'algorithm':
            self.check_alg_ready()
            if self.no_problem and self.current_algorithm:
                self.make_from_algorithm(model=model, survey_line=survey_line)
            else:
                self.log_problem('need to configure algorithm')

        elif model.source == 'previous depth line':
            line_name = model.source_name
            self.make_from_depth_line(line_name)

        else:
            # source is sdi line.  create only from sdi data
            s = 'source "sdi" only available at survey load'
            self.log_problem(s)

        if self.no_problem:
            # check arrays are filled and equal
            self.check_arrays()

        if self.no_problem:
            # line data reset by update so any edits are lost.
            self.model.edited = False

    def save_model_to_surveyline(self, model=None, survey_line=None):
        ''' save the given model to the appropriate place on the
        given survey line.  If no arguments given use current values
        returns depthline_dict key (with PRE / POST prepended)
        '''
        if survey_line is None:
            survey_line = self.data_session.survey_line
        if model is None:
            model = self.model
        logger.info('saving new depth line to surveyline {}'
                    .format(survey_line.name))
        if model.line_type == 'current surface':
            survey_line.lake_depths[model.name] = model
            survey_line.final_lake_depth = model.name
            key = 'POST_' + model.name
        else:
            survey_line.preimpoundment_depths[model.name] = model
            survey_line.final_preimpoundment_depth = model.name
            key = 'PRE_' + model.name
        
        # if model being saved is model being edited, update editor panes
        if model is self.model:
            # set form to the new line
            self.selected_depth_line_name = key
        self.update_plot()
        # update survey_line on disk
        survey_line.save_to_disk()

    def set_current_algorithm(self, alg_name=None):
        ''' Set current alg based on model.
        setting current alg will update model.args so need to save these
        and apply if neccesary after setting current alg.
        This should only be called by configure_algorithm'''
        if alg_name is None:
            alg_name = self.model.source_name
        model_args = self.model.args
        self.current_algorithm = self.algorithms[alg_name]()
        if model_args:
            self.set_alg_args(model_args)
            self.model.args = model_args
            logger.debug('model_args={}, alg args={}'
                         .format(self.model.args, self.alg_arg_dict))

    def make_from_algorithm(self, model=None, survey_line=None):
        ''' apply current algorithm for the given model (or self.model)
        for the given survey line.
        Assumes check algorithm was run.
        This sets problem flag if one encountered
        '''
        if model is None:
            model = self.model
        if survey_line is None:
            survey_line = self.data_session.survey_line
        # log attempt
        alg_name = model.source_name
        logger.debug('applying algorithm : {} to line {}'
                     .format(alg_name, survey_line.name))
        algorithm = self.current_algorithm
        try:
            trace_array, depth_array = algorithm.process_line(survey_line)
        except Exception as e:
            self.log_problem('Error occurred applying algoritm to line {}\n{}'
                             .format(survey_line.name, e))
        if self.no_problem:
            model.index_array = np.asarray(trace_array, dtype=np.int32) - 1
            model.depth_array = np.asarray(depth_array, dtype=np.float32)

    def make_from_depth_line(self, line_name):
        source_line = self.data_session.depth_dict[line_name]
        self.model.index_array = np.asarray(source_line.index_array,
                                            dtype=np.int32)
        self.model.depth_array = source_line.depth_array

    def create_new_line(self):
        ''' fill in some default value and return new depth line object'''
        new_dline = DepthLine(survey_line_name=self.survey_line_name,
                              name='Type New Name',
                              line_type='pre-impoundment surface',
                              source='algorithm',
                              edited=False,
                              locked=False)
        self.no_problem = True
        logger.info('creating new depthline template')
        return new_dline

    def _array_size(self, array=None):
        """ convenience for finding array size even if None"""
        if array is not None:
            size = len(array)
        else:
            size = 0
        return size

    def zero_out_array_data(self):
        ''' sets depth and index arrays for model to zero'''
        self.model.index_array = np.array([])
        self.model.depth_array = np.array([])

    def update_plot(self):
        ''' used as signal to update depth line choices from depth_lines prop
        so that ui choices will update'''
        self.data_session.depth_lines_updated = True

    #### checking methods #####################################################
    def check_arrays(self, depth_line=None):
        ''' checks arrays are equal and not empty'''
        if depth_line is None:
            depth_line = self.model
        d_array_size = self._array_size(depth_line.depth_array)
        i_array_size = self._array_size(depth_line.index_array)
        no_depth_array = d_array_size == 0
        no_index_array = i_array_size == 0
        depth_notequal_index = d_array_size != i_array_size
        name = self.survey_line_name
        if no_depth_array or no_index_array or depth_notequal_index:
            s = 'data arrays sizes are 0 or not equal for {}'.format(name)
            self.log_problem(s)

    def validate_name(self, model):
        ''' Validation here just means it exist after any whitespace is
        stripped off it.
        Sets problem flag if name can't be validated
        '''
        valid_name = model.name.strip()
        if valid_name == '':
            self.log_problem('depth line has no printable name')
        else:
            model.name = valid_name

    def check_if_name_is_used(self, model, survey_line=None):
        '''check that name is not in survey line depth line list already.

        Allow same name for PRE and POST lists since these are separate.
        Assume name is validated.
        Sets problem flag if name is used and notifies user
        If line is used, object is returned so caller can check locked status.
        '''
        if survey_line is None:
            survey_line = self.data_session.survey_line

        if model.line_type == 'current surface':
            used = model.name in survey_line.lake_depths.keys()
            if used:
                existing_line = survey_line.lake_depths[model.name]
        elif model.line_type == 'pre-impoundment surface':
            used = model.name in survey_line.preimpoundment_depths.keys()
            if used:
                existing_line = survey_line.preimpoundment_depths[model.name]
        else:
            self.log_problem('there seems to be no line type defined')
            used = False
        if not used:
            existing_line = None
        return existing_line

    def check_alg_ready(self, model=None):
        ''' check algorithm is selected and configured and args match model.
        If args don't match model, probably configure alg was not run, or
        alg name was changed and current_alg reset.
        Sets problem flag if not ready to run
        '''
        if model is None:
            model = self.model

        # check that algorithm is selected and valid
        not_alg = model.source != 'algorithm'
        alg_choices = self.algorithms.keys()
        good_alg_name = model.source_name in alg_choices
        if not_alg or not good_alg_name:
            self.log_problem('Invalid algorithm! Application Problem' +
                             ': not algorithm = {}, good alg name = {}'
                             .format(not_alg, good_alg_name))

        # if current alg not set then cofigure will create it and open edit.
        if self.no_problem and self.current_algorithm is None:
            self.log_problem('need to configure algorithm')

        # check current alg arguments match model
        if self.no_problem:
            self.check_args(model)

        # check model source name matched current algorithm
        match = model.source_name == self.current_algorithm.name
        if self.no_problem and not match:
            self.log_problem('alg name does not match configured algorithm.' +
                             'Need to configure algorithm')

    def check_args(self, model):
        ''' checks that arguments match the model. run after apply but
        before allowing apply to complete.  Run by check_alg_ready.
        Sets problem flag if args don't match
        '''
        alg = self.current_algorithm
        logger.debug('checking args for alg {} with args {}'
                     .format(alg.name, self.alg_arg_dict))
        if alg:
            tst = (self.model.args == self.alg_arg_dict)
            if not tst:
                s = ('arguments do not match - please configure algorithm.' +
                     'This should never not match so there may be bug')
                self.log_problem(s)

    #### logging and messaging methods ########################################

    def message(self, msg='my message'):
        dialog = MsgView(msg=msg)
        dialog.configure_traits()

    def log_problem(self, msg):
        ''' if there is a problem with any part of creating/updating a line,
        log it and notify user and set no_problem flag false'''
        self.no_problem = False
        logger.error(msg)
        self.message(msg)

    def log_model_params(self, lines=None, model=None):
        ''' log parameters of line for saving or other'''

        if lines:
            lines_str = '\n'.join([line.name for line in lines])
            s0 = ('Creating depth line for the following surveylines:\n' +
                  '    {lines}\n' +
                  '    with the following parameters:\n')
        else:
            s0 = "Current model has the following parameters:\n"
        s1a = ['name = {name}',
               'line_type = {ltype}',
               'color = {color}',
               'locked = {locked}',
               'notes = {notes}',
               'edited = {edited}',
               'source = {source}',
               'source name = {sourcename}',
               'args = {args}'
               ]
        s1 = '\n'.join(s1a)
        s = s0 + s1.format(lines=lines_str,
                           name=model.name,
                           ltype=model.line_type,
                           color=model.color,
                           locked=model.locked,
                           notes=model.notes,
                           edited=model.edited,
                           source=model.source,
                           sourcename=model.source_name,
                           args=model.args)
        logger.info(s)

    #==========================================================================
    # Get/Set methods
    #==========================================================================

    def set_alg_args(self, model_args):
        ''' if possible, sets default arguments for current algorithm configure
        dialog according to model.args dict. Otherwise warns user and continues'''
        alg = self.current_algorithm
        logger.debug('set arg defaults to model: args={}'.format(model_args))
        default = self.alg_arg_dict
        try:
            for arg in alg.arglist:
                setattr(alg, arg, model_args[arg])
        except Exception as e:
            for arg in alg.arglist:
                setattr(alg, arg, default[arg])
            s = ('cannot set arguments from model.args ({}):\nuse default'
                 .format(e))
            logger.warning(s)

    def _get_alg_arg_dict(self):
        if self.current_algorithm:
            alg = self.current_algorithm
            d = dict([(arg, getattr(alg, arg)) for arg in alg.arglist])
        else:
            d = {}
        return d

    def _get_source_names(self):
        """ update possible source names when source selection is changed"""
        source = self.source
        if source == 'algorithm':
            names = self.data_session.algorithms.keys()
        elif source == 'previous depth line':
            names = self.data_session.depth_dict.keys()
        else:
            # if source is sdi the source name is just the file it came from
            names = [self.model.source_name]
        #logger.debug('updated source {} names to {}'.format(source, names))
        return names

    def _get_survey_line_name(self):
        if self.data_session:
            name = self.data_session.survey_line.name
        else:
            name = 'No Survey Line Selected'
        return name

    @cached_property
    def _get_depth_lines(self):
        # get list of names of depthlines for the UI
        if self.data_session:
            lines = ['New Line'] + self.data_session.depth_dict.keys()
        else:
            lines = []
        logger.debug('updated depthline choices {}'.format(lines))
        return lines

    def _get_index_array_size(self):
        return self._array_size(self.model.index_array)

    def _get_depth_array_size(self):
        return self._array_size(self.model.depth_array)

    def _get_selected(self):
        '''make list of selected lines for display to user
        '''
        all_lines = []
        if self.selected_survey_lines:
            all_lines = [line.name for line in self.selected_survey_lines]
            num_lines = len(all_lines)
        else:
            num_lines = 0
        return ['LINES: {}'.format(num_lines)] + all_lines

    def _get_on_bin_line(self):
        on_bin_line = (self.selected_depth_line_name ==
                       CURRENT_SURFACE_FROM_BIN_LABEL)
        return on_bin_line

    def _get_source_choices(self):
        """ gets source choices from DepthLine class and removes sdi as an
        option if the selected line is not a line from the sdi raw data
        """
        choices = list(DepthLine.class_traits()['source'].get_validate()[1])
        if not self.on_bin_line:
            choices.remove('sdi_file')
        return choices


if __name__ == '__main__':
    from .tests.utils import get_data_session

    session = get_data_session()
    depth_line = session.lake_depths[session.lake_depths.keys()[0]]

    from hydropick.model import algorithms
    algo_dict = algorithms.get_algorithm_dict()

    #FIXME: we have set the algorithms at two different places.
    session.algorithms = algo_dict

    view = DepthLineView(
        data_session=session, model=depth_line, algorithms=algo_dict
    )
    view.configure_traits()
