""" Algorithm presenters. """
from traits.api import HasStrictTraits, Instance
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
