import re
import os
from fnmatch import fnmatch

from aimsflow.util import file_to_str
from aimsflow.util.string_utils import find_re_pattern

JOURNAL_NAME = {
        'ac': ["ac", "Appl. Catal.", "Applied Catalysis"],
        'aca': ["aca", "Appl. Catal. A: Gen.", "Applied Catalysis A: General"],
        'acb': ["acb", "Appl. Catal. B: Environ.", "Applied Catalysis B: Environmental"],
        'ami': ["ami", "ACS Appl. Mater. Inter.", "ACS Appl. Mater. Interfaces",
                "ACS Applied Materials and Interfaces",
                "ACS Applied Materials \& Interfaces"],
        'acsnano': ["acsnano", "ACS Nano"],
        'acscata': ["acscata", "ACS Catal."],
        'acscombsci': ["acscombsci", "ACS Comb. Sci."],
        'actaca': ["actaca", "Acta Crystallogr. Sect. A",
                   "Acta Crystallographica Section A"],
        'actacb': ["actacb", "Acta Crystallogr. Sect. B",
                   "Acta Crystallographica Section B"],
        'actacryst': ["actacryst", "Acta Cryst.", "Acta Crystal"],
        'actamat': ["actamat", "Acta Mater.", "Acta Materialia"],
        'advmat': ["advmat", "Adv. Mater.", "Advanced Materials"],
        'advfunmat': ["advfunmat", "Adv. Fun. Mater.", "Advanced Functional Materials"],
        'advmatif':['advmatif', 'Advanced Materials Interfaces'],
        'am': ["am", "APL Mater.", "APL Materials"],
        'ass': ["ass", "Appl. Sur. Sci.", "Applied Surface Science"],
        'aipa': ["aipa", "AIP Advances"],
        'aipcp': ["aipcp", "AIP Conference Proceedings"],
        'assp': ["assp", "Adv. Solid State Phys."],
        'apl': ["apl", "Appl. Phys. Lett.", "Applied Physics Letters"],
        'ape': ["ape", "Appl. Phys. Express", "Applied Physics Express"],
        'armr': ["armr", "Annu. Rev. Mater. Sci.", "Annual Review of Materials Research"],
        'arcmp': ["arcmp", "Annual Review of Condensed Matter Physics"],
        'bpas': ["bpas", "Bull. Pol. Acad. Sci.-Tech. Sci.",
                 "Bulletin of The Polish Academy of Sciences-Technical Sciences"],
        'catalett': ["catalett", "Catal. Lett.", "Catalysis Letters"],
        'cc': ["cc", "Chem. Commun.", "Chemical Communications"],
        'cl': ["cl", "Chem. Lett."],
        'cm': ["cm", "Chem. Mater.", "CHEMISTRY OF MATERIALS", "Chemistry of Materials"],
        'cms': ["cms", "Comput. Mater. Sci.", "Computational Materials Science", "COMPUTATIONAL MATERIALS SCIENCE"],
        'cp': ["cp", "Chem. Phys.", "Chemical Physics"],
        'cpc': ["cpc", "ChemPhysChem"],
        'cpl': ["cpl", "Chem. Phys. Lett."],
        'crse': ["crse", "Cat. Rev.-Sci. Eng."],
        'ct': ["ct", "Catal. Today"],
        'csr': ["csr", "Chem. Soc. Rev.", "Chemical Society Reviews"],
        'chemrev': ["chemrev", "Chem. Rev.", "CHEMICAL REVIEWS", "Chemical Reviews"],
        'dt': ["dt", "Dalton T.", "Dalton Transactions"],
        'ecs': ["ecs", "ECS Transactions", "ECS Transactions"],
        'ec': ["ec", "Electrochem. Commun.", "Electrochemistry Communications"],
        'ees': ["ees", "Energy Environ. Sci."],
        'epjd': ["epjd", "Eur. Phys. J. D", "The European Physical Journal D"],
        'epjb': ["epjb", "Eur. Phys. J. B", "The European Physical Journal B"],
        'epjst': ["epjst", "Eur. Phys. J.-Spec. Top.", "European Physical Journal-Special Topics"],
        'epl': ["epl", "EPL-Europhys. Lett.", "Europhysics Letters"],
        'est': ["est", "Environ. Sci. Technol"],
        'fardis': ["fardis", "Faraday Discuss.", "Faraday Discussions"],
        'ferro': ["ferro", "Ferroelectrics"],
        'ieeeml': ["ieeeml", "IEEE Trans. Magn.", "IEEE Transactions on Magnetics"],
        'ieeetm': ["ieeetm", "IEEE Magn. Lett.", "IEEE Magnetics Letters"],
        'ic': ["ic", "Inorg. Chem.", "Inorganic Chemistry"],
        'ijhe': ["ijhe", "Int. J. Hydrogen Energy", "International Journal of Hydrogen Energy"],
        'ijp': ["ijp", "Int. J. Photoenergy", "International Journal of Photoenergy"],
        'ijqc': ["ijqc", "Int. J. Quantum. Chem.", "International Journal of Quantum Chemistry"],
        'iecr': ["iecr", "Ind. Eng. Chem. Res.", "Industrial & Engineering Chemistry Research"],
        'jac': ["jac", "J. Alloys Compound.", "Journal of Alloys and Compounds"],
        'jacs': ["jacs", "J. Am. Chem. Soc."],
        'jacers': ["jacers", "J. Am. Ceram. Soc.", "Journal of the American Ceramic Society"],
        'jap': ["jap", "J. Appl. Phys.", "Journal of Applied Physics"],
        'jc': ["jc", "J. Catal.", "Journal of Catalysis"],
        'jcc': ["jcc", "J. Comput. Chem.", "Journal of Computational Chemistry"],
        'jcg': ["jcg", "J. Cryst. Growth"],
        'jcp': ["jcp", "J. Chem. Phys."],
        'je': ["je", "J. Electroceram.", "Journal of Electroceramics"],
        'jem': ["jem", "J. Elec. Mat."],
        'jesoc': ["jesoc", "J. Electrochem. Soc.", "Journal of The Electrochemical Society"],
        'jhm': ["J. Hazard. Mater."],
        'jim': ["jim", "J. Inst. Met.", "Journal of the Institute of Metals"],
        'jjap': ["jjap", "Jpn. J. Appl. Phys", "Japanese JAP", "Japanese Journal of Applied Physics"],
        'jlum': ["jlum", "J. Luminescence"],
        'jmmm': ["jmmm", "J. Magn. Magn. Mater.", "Journal of Magnetism and Magnetic Materials"],
        'jmc': ["jmc", "J. Mater. Chem.", "Journal of Materials Chemistry"],
        'jmca': ["jmca", "J. Mater. Chem. A", "Journal of Materials Chemistry A"],
        'jmacb': ["jmacb", "J. Mater. Chem. B", "Journal of Materials Chemistry A"],
        'jmacc': ["jmacc", "J. Mater. Chem. C", "Journal of Materials Chemistry A"],
        'jmoca': ["jmoca", "J. Mol. Catal. A: Chem.", "Journal of Molecular Catalysis A: Chemical"],
        'jmocb': ["jmocb", "J. Mol. Catal. B: Enzym.", "Journal of Molecular Catalysis B: Enzymatic"],
        'jmr': ["jmr", "J. Mater. Res.", "Journal of Materials Research"],
        'jms': ["jms", "J. Mater. Sci.", "Journal of Materials Science"],
        'jmos': ["jmos", "J. Mol. Struct.", "Journal of Molecular Structure"],
        'jmsl': ["jmsl", "J. Mater. Sci. Lett.", "Journal of Materials Science Letters"],
        'jpc': ["jpc", "J. Phys. Chem."],
        'jpca': ["jpca", "J. Phys. Chem. A", "The Journal of Physical Chemistry A"],
        'jpcb': ["jpcb", "J. Phys. Chem. B", "The Journal of Physical Chemistry B"],
        'jpcc': ["jpcc", "J. Phys. Chem. C", "The Journal of Physical Chemistry C"],
        'jpcl': ["jpcl", "J. Phys. Chem. Lett.", "The Journal of Physical Chemistry Letter"],
        'jpcs': ["jpcs", "J. Phys. Chem. Solids", "Journal of Physics and Chemistry of Solids"],
        'jpcm': ["jpcm", "J. Phys.: Condens. Matter", "Journal of Physics: Condensed Matter"],
        'jpconser': ["jpconser", "J. Phys.: Conf. Ser.", "Journal of Physics: Conference Series"],
        'jpf': ["jpf", "J. Phys. F-Met. Phys.", "Journal of Physics F: Metal Physics"],
        'jpd': ["jpd", "J. Phys. D: Appl. Phys.", "Journal of Physics D-APPLIED PHYSICS",
                "Journal of Physics D: Applied Physics"],
        'jpe': ["jpe", "J. Phase Equilib.", "JOURNAL OF PHASE EQUILIBRIA"],
        'jped': ["jped", "J. Phase Equilib. Diffus.", "JOURNAL OF PHASE EQUILIBRIA AND DIFFUSION"],
        'jppa': ["jppa", "J. Photochem. Photobio. A", "Journal of Photochemistry and Photobiology A: Chemistry"],
        'jps': ["jps", "J. Power Source", "Journal of Power Sources"],
        'jssc': ["jssc", "J. Solid State Chem.", "JOURNAL OF SOLID STATE CHEMISTRY"],
        'jtac': ["jtac", "Journal of Thermal Analysis and Calorimetry"],
        'jvstb': ["jvstb", "J. Vac. Sci. Technol. B", "Journal of Vacuum Science \& Technology B"],
        'kri': ["kri", "Kristallografiya", "Kristallografiya, English Title: Crystallography Reports"],
        'microeng': ["microeng", "Microelec. Eng."],
        'ml': ["ml", "Mater. Lett."],
        'mrs': ["mrs", "MRS Bull.", "MRS BULLETIN"],
        'mrb': ["mrb", "Mater. Res. Bull.", "Materials Research Bulletin"],
        'mtd': ["mtd", "Mater. Today.", "Materials Today"],
        'mt': ["mt", "Mater. Trans.", "Materials Transactions"],
        'mtr': ["mtr", "Metall. Trans.", "METALLURGICAL TRANSACTIONS"],
        'mmtra': ["mmtra", "Metall. Mater. Trans. A", "METALLURGICAL AND MATERIALS TRANSACTIONS A-PHYSICAL METALLURGY AND MATERIALS SCIENCE"],
        'nanotech': ["nanotech", "Nanotechnology", "Nanotechnology"],
        'nature': ["nature", "Nature"],
        'nchem': ["nchem", "Nat. Chem.", "Nature Chemistry"],
        'ncom': ["ncom", "Nat. Commun.", "Nature Communications"],
        'nima': ["nima", "Nucl. Instrum. Methods Phys. Res., Sect. A",
                 "Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors"],
        'nimb': ["nimb", "Nucl. Instrum. Methods Phys. Res., Sect. B",
                 "Nuclear Instruments and Methods in Physics Research Section B: Beam Interactions with Materials and"],
        'njp': ["njp", "New J. Phys.", "New Journal of Physics"],
        'nl': ["nl", "Nano Lett.", "Nano Letters"],
        'nmat': ["nmat", "Nat. Mater.", "Nat Mater", "Nature Materials"],
        'nnano': ["nnano", "Nat. Nanotechnol.", "Nature Nanotechnology"],
        'nphys': ["nphys", "Nat. Phys.", "Nature Physics"],
        'npho': ["npho", "Nat. Photon.", "Nature photonics"],
        'nphoton': ["nphoton", "Nat. Photon.", "Nature Photon"],
        'nt': ["nt", "Nano. Today"],
        'pccp': ["pccp", "Phys. Chem. Chem. Phys.", "Physical Chemistry Chemical Physics"],
        'phm': ["phm", "Phil. Mag."],
        'phmb': ["phmb", "Phil. Mag. B"],
        'pnas': ["pnas", "Proc. Natl. Acad. Sci."],
        'pla': ["pla", "Phys. Lett. A", "Physics Letters A"],
        'ppsb': ["ppsb", "Proc. Phys. Soc. B", "Proceedings of the Physical Society"],
        'pr': ["pr", "Phys. Rev.", "Physical Review (OLD STUFF)"],
        'pra': ["pra", "Phys. Rev. A", "Physical Review A"],
        'prb': ["prb", "Phys. Rev. B", "Physical Review B"],
        'prc': ["prc", "Phys. Rev. C", "Physical Review C"],
        'prd': ["prd", "Phys. Rev. D", "Physical Review D"],
        'pre': ["pre", "Phys. Rev. E", "Physical Review E"],
        'prm': ["prm", "Phys. Rev. Mater.", "Phys. Rev. Materials", "Physical Review Materials"],
        'prl': ["prl", "Phys. Rev. Lett.", "Physical Review Letters"],
        'prapp': ["prapp", "Phys. Rev. Applied", "Physical Review Applied"],
        'prx': ["prx", "Phys. Rev. X", "Physical Review X"],
        'pwd': ["pwd", "Physics World", "Physics World"],
        'psca': ["psca", "Physica A"],
        'psce': ["psce", "Physica E"],
        'pms': ["pms", "Prog. Mater. Sci.", "Progress in Materials Science"],
        'pss': ["pss", "Phys. Solid State", "Physics of the Solid State"],
        'pssprl': ["pssprl", "Phys. Stat. Solidi (PRL)"],
        'pssa': ["pssa", "Phys. Stat. Solidi (A)"],
        'pssb': ["pssb", "Phys. Stat. Solidi (B)"],
        'pssc': ["pssc", "Prog. Solid State Chem.", "PROGRESS IN SOLID STATE CHEMISTRY"],
        'rmp': ["rmp", "Rev. Mod. Phys.", "Reviews of Modern Physics"],
        'rpp': ["rpp", "Rep. Prog. Phys.", "Reports on Progress in Physics"],
        'rsc': ["rsc", "RSC Adv.,", "RSC advances"],
        'rser': ["rser", "Renew. Sust. Energ. Rev.", "Reports on Progress in Physics"],
        'sm': ["sm", "Superlattices Microstruct.", "Superlattices and Microstructures"],
        'scrmat': ["scrmat", "Scripta Mater.", "Scripta Materialia"],
        'sr': ["sr", "Sci. Rep.", "Scientific Reports"],
        'sst': ["sst", "Semicond. Sci. Tech.", "Semiconductor Science and Technology"],
        'sct': ["sct", "Surf. Coat. Technol.", "Surface and Coatings Technology"],
        'ssr': ["ssr", "Surf. Sci. Rep.", "Surface Science Report"],
        'ssc': ["ssc", "Solid State Commun.", "SOLID STATE COMMUNICATIONS"],
        'sss': ["sss", "Solid State Sci.", "Solid State Sciences"],
        'science': ["science", "Science"],
        'sa': ["Sci. Adv.", "Science Advances"],
        'ss': ["ss", "Surf. Sci.", "Surface Science"],
        'tsf': ["tsf", "{Thin Solid Films"],
        'tns': ["tns", "IEEE Trans. on Nucl. Sci."],
        'wst': ["wst", "Water Sci. Technol.", "Water Science and Technology"],
        'zac': ["zac", "Z. Anorganische Chemie", "Zeitschrift fur Anorganische Chemie"],
        'zaac': ["zaac", "Z. Anorganische und Allgemeine Chemie", "Zeitschrift fur Anorganische und Allgemeine Chemie"],
        'zm': ["zm", "Z. Metallkd.", "Zeitschrift fur Metallkunde"]
    }
BIB_PATTERN = {
    'author': re.compile('author\s*=\s*[{"](.*?)[}"],*\s*\n', re.DOTALL | re.I),
    'title': re.compile('title\s*=\s*[{"](.*?)[}"],\s*\n\s*[a-z]', re.DOTALL | re.I),
    'journal': re.compile('journal\s*=\s*[{"]*(.*?)[}"]*[,\n]', re.I),
    'year': re.compile('year\s*=\s*[{"]*(.*?)[}"]*[,\n]', re.I),
    'volume': re.compile('volume\s*=\s*[{"](.*?)[}"][,\n]', re.I),
    'pages': re.compile('[^a-z]pages\s*=\s*[{"](.*?)[}"][,\n]', re.I),
    'doi': re.compile('doi\s*=\s*[{"](.*?)[}"][,\n]', re.I)
}
RIS_PATTERN = {
    'author': re.compile('AU\s*-\s*(.*?)\s*\n', re.DOTALL | re.I),
    'title': re.compile('TI\s*-\s*(.*?)\s*\n', re.DOTALL | re.I),
    'journal': re.compile('J[A|O]\s*-\s*(.*?)\s*\n', re.DOTALL | re.I),
    'year': re.compile('PY\s*-\s*(.*?)[/|\n]', re.DOTALL | re.I),
    'volume': re.compile('VL\s*-\s*(.*?)\s*\n', re.DOTALL | re.I),
    'pages': re.compile('SP\s*-\s*(.*?)\s*\n', re.DOTALL | re.I),
    'doi': re.compile('UR\s*-\s*(.*?)\s*\n', re.DOTALL | re.I)
    }


class Citation(object):
    def __init__(self, filename):
        fname = os.path.basename(filename)
        str_cite = file_to_str(filename)
        if fnmatch(fname.lower(), "*.ris*"):
            cite = find_re_pattern(RIS_PATTERN, str_cite)
            if isinstance(cite['author'], list):
                cite['author'] = " and ".join(cite['author'])
        elif fnmatch(fname.lower(), "*.bib*"):
            cite = find_re_pattern(BIB_PATTERN, str_cite)
        else:
            raise IOError("'%s' is not a citation file. Please make sure the "
                          "file formate is either '*.ris*' or '*.bib*'")
        self.cite = cite

    def __str__(self):
        bib_str = '''@article{%s,
    author = {%s},
    title = {%s},
    journal = %s,
    year = {%s},
    volume = {%s},
    pages = {%s},
    doi = {%s}
}''' % (self.bibkey, self.author, self.title, self.journal, self.year,
        self.volume, self.pages, self.doi)
        return bib_str

    @property
    def author(self):
        return self.cite['author']

    @property
    def year(self):
        return self.cite['year']

    @property
    def volume(self):
        return self.cite['volume']

    @property
    def pages(self):
        return self.cite['pages'].replace("--", "-").replace(" ", "")

    @property
    def doi(self):
        doi = self.cite['doi']
        for i in range(len(doi)):
            if doi[i].isdigit():
                doi = doi[i:]
                break
        return doi

    @property
    def journal(self):
        find_journal = False
        for k, v in JOURNAL_NAME.items():
            if self.cite['journal'] in v:
                journal = k
                find_journal = True
                break
        if not find_journal:
            journal = "{%s}" % self.cite['journal']
        return journal

    @property
    def title(self):
        ignore_text = ["a", "an", "the", "at", "by", "for", "in", "of", "on",
                       "to", "up", "and", "as", "but", "or", "nor",
                       "with", "from", "between", "via", "across"]
        new_title = self.cite['title'].split()
        for i in range(0, len(new_title)):
            if new_title[i] not in ignore_text:
                if '-' in new_title[i]:
                    sub_new_title = new_title[i].split('-')
                    try:
                        new_title[i] = '-'.join([j[0].upper() + j[1:]
                                                 for j in sub_new_title])
                    except IndexError:
                        new_title[i] = new_title[i][0].upper() + new_title[i][1:]
                elif '/' in new_title[i]:
                    sub_new_title = new_title[i].split('/')
                    try:
                        new_title[i] = '/'.join([j[0].upper() + j[1:]
                                                 for j in sub_new_title])
                    except IndexError:
                        new_title[i] = new_title[i][0].upper() + new_title[i][1:]
                else:
                    new_title[i] = new_title[i][0].upper() + new_title[i][1:]

        return ' '.join(new_title)

    @property
    def bibkey(self):
        if ',' in self.author:
            first_author = re.findall('(.*?)[\s,]', self.author)[0]
        elif not self.author:
            first_author = ''
        elif 'and' in self.author:
            first_author = re.findall('\s(.*?)\s', self.author)[0]
        else:
            first_author = re.findall('\s(.*)', self.author)[0]

        first_author = re.sub('[^a-zA-Z]+', '', first_author)
        key = '%s_%s_%s' % (first_author, self.year, self.journal)
        return key.replace(" ", "")
