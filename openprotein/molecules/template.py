import dataclasses
from collections.abc import Mapping
from types import NoneType

from .complex import Complex
from .protein import Protein

TemplateSource = Protein | Complex
TargetMolecule = Protein | Complex
ChainMapping = Mapping[str, str] | str | None


@dataclasses.dataclass(frozen=True)
class Template:
    """
    A structural template used to guide the folding of a target chain or complex.

    This class wraps a structural source (Protein or Complex) and defines how it
    should map to the target(s).

    Attributes:
        template (Protein | Complex): The structural object to be used as a template.
            Must contain structural data (coordinates).
        mapping (Mapping[str, str] | str | None): The rule for assigning this template
            to the target.
            - Mapping[str, str]: Explicitly maps {template_chain_id: target_chain_id}.
            - str: Apply this template to a specific target_chain_id. (If template is
              a Complex, a selection algorithm is used to pick the best source chain).
            - None: Automatic assignment. The folding algorithm will determine which
              chain(s) this template applies to.
    """

    template: TemplateSource
    mapping: ChainMapping = None

    def __post_init__(self) -> None:
        """Validates the template upon initialization."""
        self._validate_self()

    def _validate_self(self) -> None:
        """Checks internal consistency of the Template."""
        if isinstance(self.template, Protein):
            if not self.template.has_structure:
                raise ValueError("Provided template Protein has no structural data.")
            # A single Protein object is treated as an atomic unit (anonymous chain).
            # It cannot support a dictionary mapping because it has no internal Chain IDs
            # to map *from*.
            if not isinstance(self.mapping, (str, NoneType)):
                raise ValueError(
                    f"Invalid mapping type '{type(self.mapping)}' for Protein template. "
                    "Expected 'str' (target ID) or 'None'. A dict mapping is only valid "
                    "if the template is a Complex with named chains."
                )
        elif isinstance(self.template, Complex):
            # Ensure all parts of the complex have structure
            for chain_id, protein in self.template.get_proteins().items():
                if not protein.has_structure:
                    raise ValueError(
                        f"Template Chain '{chain_id}' has no structural data."
                    )
            # If mapping is explicit (dict), ensure source keys exist in the template
            if not isinstance(self.mapping, (str, NoneType)):
                template_chains = set(self.template.get_chains().keys())
                mapping_keys = set(self.mapping.keys())
                if not mapping_keys.issubset(template_chains):
                    missing = mapping_keys - template_chains
                    raise ValueError(
                        f"Mapping contains source chain IDs {missing} that do not "
                        f"exist in the template complex (available: {template_chains})."
                    )
        else:
            raise TypeError(
                f"Template source must be Protein or Complex, got {type(self.template)}"
            )

    def validate_for_target(self, target: TargetMolecule) -> None:
        """
        Ensures this Template is compatible with a specific target Molecule.

        Args:
            target: The Protein or Complex that is being folded.

        Raises:
            ValueError: If this Template is invalid, or if chain IDs referenced in
                        mapping do not exist in the target.
            TypeError: If the template/target combination is structurally incompatible.
        """
        self._validate_self()
        if isinstance(target, Protein):
            # Target is a single Protein (implies anonymous/single context).
            # We cannot map to a specific chain ID because the target Protein object
            # doesn't have a chain ID.
            if self.mapping is not None:
                raise ValueError(
                    "Cannot use a specific chain mapping when the target is a standalone Protein. "
                    "Mapping must be None."
                )
        elif isinstance(target, Complex):
            target_chains = {
                chain_id
                for chain_id, chain in target.get_chains().items()
                if isinstance(self.template, Complex)
                or isinstance(chain, type(self.template))
            }
            if isinstance(self.mapping, str):
                # Mapping points to a specific target chain ID
                if self.mapping not in target_chains:
                    raise ValueError(
                        f"Template maps to target chain '{self.mapping}', but this chain "
                        f"does not exist in the target Complex (available: {target_chains})."
                    )
            elif self.mapping is not None:
                # Mapping points from Source -> specific target chain IDs
                target_values = set(self.mapping.values())
                if not target_values.issubset(target_chains):
                    missing = target_values - target_chains
                    raise ValueError(
                        f"Template maps to target chains {missing} which do not exist "
                        f"in the target Complex (available: {target_chains})."
                    )
        else:
            raise TypeError(f"Target must be Protein or Complex, got {type(target)}")
