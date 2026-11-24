"""Sequence generation utilities for testing."""

import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Real antibody variable-region sequences for antibody-specific models such as
# ablang2, which reject random amino-acid strings. The heavy chain is an
# IGHV3-23-based VH; the light chain is the trastuzumab kappa VL. Both are valid
# natural antibody variable domains.
ANTIBODY_HEAVY_SEQUENCE = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKYYYYGMDVWGQGTTVTVSS"
ANTIBODY_LIGHT_SEQUENCE = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

# Human IgG1 CH1 through the upper hinge. ANARCI numbers antibody variable
# domains only, so appending this to a VH gives a sequence whose last 103
# residues carry no antibody position.
ANTIBODY_HEAVY_CONSTANT_SEQUENCE = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC"


def mutate_sequence(
    sequence: str, mutation_rate: float = 0.05, ensure_mutation: bool = True
) -> str:
    """
    Return a copy of ``sequence`` with random substitutions applied.

    Substitution-only, so antibody variable domains keep a valid length and
    numbering, and each test run produces a unique sequence that busts
    server-side caches.

    When ``ensure_mutation`` is True, at least one residue is guaranteed to be
    substituted, so callers can rely on the result differing from the input.
    """
    if not sequence:
        return sequence

    while True:
        residues = []
        mutated = False
        for aa in sequence:
            if random.random() < mutation_rate:
                residues.append(random.choice([a for a in AMINO_ACIDS if a != aa]))
                mutated = True
            else:
                residues.append(aa)
        if mutated or not ensure_mutation:
            return "".join(residues)


def random_sequence_fake(length: int) -> str:
    """
    Generate a random protein sequence of a given length.
    Uses the 20 standard amino acids.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(random.choice(amino_acids) for _ in range(length))


# Window sources for `random_sequence_real`. Real, full-length, reviewed
# sequences: a signal-peptide-bearing transcription factor, human insulin
# preproprotein, human haemoglobin alpha, human serum albumin, and E. coli
# beta-galactosidase. The last is 1024 residues, so every length the tests ask
# for can be windowed without reaching the network.
WINDOW_SOURCE_PROTEINS = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL",
    "MTMITDSLAVVLQRRDWENPGVTQLNRLAAHPPFASWRNSEEARTDRPSQQLRSLNGEWRFAWFPAPEAVPESWLECDLPEADTVVVPSNWQMHGYDAPIYTNVTYPITVNPPFVPTENPTGCYSLTFNVDESWLQEGQTRIIFDGVNSAFHLWCNGRWVGYGQDSRLPSEFDLSAFLRAGENRLAVMVLRWSDGSYLEDQDMWRMSGIFRDVSLLHKPTTQISDFHVATRFNDDFSRAVLEAEVQMCGELRDYLRVTVSLWQGETQVASGTAPFGGEIIDERGGYADRVTLRLNVENPKLWSAEIPNLYRAVVELHTADGTLIEAEACDVGFREVRIENGLLLLNGKPLLIRGVNRHEHHPLHGQVMDEQTMVQDILLMKQNNFNAVRCSHYPNHPLWYTLCDRYGLYVVDEANIETHGMVPMNRLTDDPRWLPAMSERVTRMVQRDRNHPSVIIWSLGNESGHGANHDALYRWIKSVDPSRPVQYEGGGADTTATDIICPMYARVDEDQPFPAVPKWSIKKWLSLPGETRPLILCEYAHAMGNSLGGFAKYWQAFRQYPRLQGGFVWDWVDQSLIKYDENGNPWSAYGGDFGDTPNDRQFCMNGLVFADRTPHPALTEAKHQQQFFQFRLSGQTIEVTSEYLFRHSDNELLHWMVALDGKPLASGEVPLDVAPQGKQLIELPELPQPESAGQLWLTVRVVQPNATAWSEAGHISAWQQWRLAENLSVTLPAASHAIPHLTTSEMDFCIELGNKRWQFNRQSGFLSQMWIGDKKQLLTPLRDQFTRAPLDNDIGVSEATRIDPNAWVERWKAAGHYQAEAALLQCTADTLADAVLITTAHAWQHQGKTLFISRKTYRIDGSGQMAITVDVEVASDTPHPARIGLNCQLAQVAERVNWLGLGPQENYPDRLTAACFDRWDLPLSDMYTPYVFPSENGLRCGTRELNYGPHQWRGDFQFNISRYSQQQLMETSHRHLLHAEEGTWLNIDGFHMGIGGDDSWSPSVSAEFQLSAGRYHYQLVWCQK",
]

_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_fetched_sources: list[str] = []


def _fetch_window_sources(min_length: int, size: int = 25) -> list[str]:
    """Reviewed UniProt sequences of at least ``min_length`` residues."""
    import urllib.parse
    import urllib.request

    query = f"reviewed:true AND length:[{min_length} TO {min_length * 3}]"
    url = f"{_UNIPROT_SEARCH}?" + urllib.parse.urlencode(
        {"query": query, "format": "fasta", "size": size}
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        fasta = response.read().decode()
    return [
        "".join(record.splitlines()[1:])
        for record in fasta.split(">")
        if record.strip()
    ]


def _window_sources(length: int) -> list[str]:
    """Proteins long enough to yield a ``length``-residue window.

    The bundled set covers the shorter lengths without a network call. Anything
    longer is drawn from UniProt once and reused for the rest of the session.
    """
    usable = [p for p in WINDOW_SOURCE_PROTEINS + _fetched_sources if len(p) >= length]
    if usable:
        return usable
    _fetched_sources.extend(_fetch_window_sources(length))
    usable = [p for p in _fetched_sources if len(p) >= length]
    if not usable:
        raise RuntimeError(
            f"no source protein of at least {length} residues is available"
        )
    return usable


def random_sequence_real(length: int, real_proteins: list[str] | None = None) -> str:
    """
    Return a contiguous window of one real protein sequence.

    A window of a single protein keeps the homology a search needs to find:
    stitching snippets of unrelated proteins together, or padding with random
    residues, yields a sequence with no relatives in any database.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return ""

    sources = real_proteins or _window_sources(length)
    candidates = [p for p in sources if len(p) >= length]
    if not candidates:
        raise ValueError(f"no supplied protein is at least {length} residues long")
    protein = random.choice(candidates)
    start = random.randint(0, len(protein) - length)
    return protein[start : start + length]


def mutated_antibody_sequences(
    base_seq: str, num_sequences: int = 3, mutation_rate: float = 0.02
) -> list[str]:
    """
    Generate unique antibody sequences via substitution-only mutation.

    Returns the base sequence plus mutated copies. Substitution keeps the
    variable-domain length and framing intact, so every sequence stays a valid
    antibody that AbNumber/ANARCI can number, while making each test run unique
    to bust server-side caches.
    """
    sequences = {base_seq}
    while len(sequences) < num_sequences:
        sequences.add(mutate_sequence(base_seq, mutation_rate=mutation_rate))
    return list(sequences)


def random_mutated_sequences(
    base_seq: str, num_sequences: int = 3, mutation_rate: float = 0.05
) -> list[str]:
    """
    Generates a list of sequences by introducing random mutations (substitutions,
    insertions, and deletions) to a base sequence. Includes the original base
    sequence in the returned list.
    """
    sequences = {base_seq}
    while len(sequences) < num_sequences:
        mutated_seq_list = []
        i = 0
        while i < len(base_seq):
            # Check for mutation
            if random.random() < mutation_rate:
                mutation_type = random.choice(["substitute", "insert", "delete"])

                if mutation_type == "substitute":
                    original_aa = base_seq[i]
                    new_aa = random.choice(
                        [aa for aa in AMINO_ACIDS if aa != original_aa]
                    )
                    mutated_seq_list.append(new_aa)

                elif mutation_type == "insert":
                    # insert before current residue
                    mutated_seq_list.append(random.choice(AMINO_ACIDS))
                    mutated_seq_list.append(base_seq[i])  # keep the original

                # "delete" appends nothing, dropping the residue.

            else:  # No mutation
                mutated_seq_list.append(base_seq[i])

            i += 1

        # Add the new unique sequence
        sequences.add("".join(mutated_seq_list))

    return list(sequences)
