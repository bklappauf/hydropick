import os
import tempfile

from ..survey_data_session import SurveyDataSession
from hydropick.io.import_survey import import_sdi, import_cores

def get_data_session():
    """ Returns a default data session.

    FIXME: add caching ...
    """

    data_dir = 'SurveyData'
    survey_name = '12030221'
    tempdir = tempfile.mkdtemp()
    h5file = os.path.join(tempdir, 'test.h5')

    test_dir = os.path.dirname(__file__)
    data_path = os.path.join(test_dir, data_dir)
    lines, groups, _, _ = import_sdi(data_path, h5file)
    core_samples = import_cores(os.path.join(data_path, 'Coring'), h5file)
    for line in lines:
        if line.name == survey_name:
            survey_line = line
            survey_line.core_samples = core_samples
    survey_line.load_data(h5file)
    session = SurveyDataSession(survey_line=survey_line)

    return session
