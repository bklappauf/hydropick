""" Algorithm presenters. """
from traits.api import HasStrictTraits, Instance, Str, Bool, Enum
from traitsui.api import View, Item, TextEditor, Group, UItem, Label

from hydropick.model.i_algorithm import IAlgorithm


class AlgorithmPresenter(HasStrictTraits):

    algorithm = Instance(IAlgorithm)

    def traits_view(self):
        view = View(
            Group(
                Label('Instructions:', emphasized=True),
                UItem('object.algorithm.instructions',
                      editor=TextEditor(), style='readonly',
                      emphasized=True)
            ),
            Item('_'),
            *['object.algorithm.{}'.format(arg)
              for arg in self.algorithm.arglist],
            buttons=['OK', 'Cancel'],
            kind='modal'
        )
        return view


class NotesEditorPresenter(HasStrictTraits):
    ''' view to edit notes for a depth line'''
    notes = Str
    traits_view = View(Item('notes',
                            editor=TextEditor(), style='custom'),
                       buttons=['OK', 'Cancel'],
                       kind='modal',
                       resizable=True
                       )


class ApplyToGroupSettingsPresenter(HasStrictTraits):
    ''' view to set over-write options for apply to group method'''
    overwrite_name = Bool(False)
    overwrite_locked = Bool(False)
    overwrite_approved = Bool(False)
    new_name = Str

    traits_view = View(Item('overwrite_name',
                            tooltip=('will overwrite existing lines with' +
                                     ' same name (depending on locked)')),
                       Item('overwrite_locked',
                            tooltip=('will overwrite existing name if' +
                                     ' overwrite name check even if line' +
                                     ' is locked')),
                       Item('overwrite_approved',
                            tooltip=('will apply settings even to approved' +
                                     ' lines (based on above settings)')),
                       'new_name',
                       buttons=['OK', 'Cancel'],
                       kind='modal',
                       resizable=True
                       )


class DeletePresenter(HasStrictTraits):
    ''' view to edit notes for a depth line'''
    answer = Enum('No', 'Yes')
    traits_view = View(Label('Are you sure you want to delete this line?'),
                       Item('answer', style='custom'),
                       buttons=['OK', 'Cancel'],
                       kind='modal',
                       resizable=True
                       )

    def _answer_default(self):
        return 'No'


if __name__ == '__main__':
    from hydropick.model.algorithms import XDepthAlgorithm
    presenter = AlgorithmPresenter(algorithm=XDepthAlgorithm())
    presenter.configure_traits()
