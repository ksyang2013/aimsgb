from pathlib import Path
from aimsgb import Grain
from pymatgen.util.testing import PymatgenTest

INPUT_DIR = Path(__file__).parent.absolute() / "input"

class GrainTest(PymatgenTest):
    def setUp(self):
        self.structure = Grain.from_file(f'{INPUT_DIR}/POSCAR_mp-13')

    def test_make_supercell(self):
        self.structure.make_supercell([2, 1, 1])
        assert self.structure.formula == 'Fe4'
        
    # def test_from_mp_id(self):
    #     s = Grain.from_mp_id('mp-13')
    #     assert s == self.structure
