""" Algorithm presenters. """
from traits.api import HasStrictTraits, Instance, Str, Bool
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

if __name__ == '__main__':
    from hydropick.model.algorithms import XDepthAlgorithm
    presenter = AlgorithmPresenter(algorithm=XDepthAlgorithm())
    presenter.configure_traits()


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
                       buttons=['OK', 'Cancel'],
                       kind='modal',
                       resizable=True
                       )
