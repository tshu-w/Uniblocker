from pathlib import Path
from typing import Literal

from jnius import autoclass
from jsonargparse import CLI
from rich import print

List = autoclass("java.util.List")
Set = autoclass("java.util.Set")
EntityReader = autoclass("org.scify.jedai.datareader.entityreader.AbstractEntityReader")
EntityCSVReader = autoclass("org.scify.jedai.datareader.entityreader.EntityCSVReader")
EntityProfile = autoclass("org.scify.jedai.datamodel.EntityProfile")
GroundTruthReader = autoclass(
    "org.scify.jedai.datareader.groundtruthreader.AbstractGtReader"
)
GtCSVReader = autoclass("org.scify.jedai.datareader.groundtruthreader.GtCSVReader")
IdDuplicates = autoclass("org.scify.jedai.datamodel.IdDuplicates")
DuplicatePropagation = autoclass(
    "org.scify.jedai.utilities.datastructures.AbstractDuplicatePropagation"
)
BilateralDuplicatePropagation = autoclass(
    "org.scify.jedai.utilities.datastructures.BilateralDuplicatePropagation"
)
UnilateralDuplicatePropagation = autoclass(
    "org.scify.jedai.utilities.datastructures.UnilateralDuplicatePropagation"
)

StandardBlocking = autoclass("org.scify.jedai.blockbuilding.StandardBlocking")
ComparisonsBasedBlockPurging = autoclass(
    "org.scify.jedai.blockprocessing.blockcleaning.ComparisonsBasedBlockPurging"
)
ComparisonPropagation = autoclass(
    "org.scify.jedai.blockprocessing.comparisoncleaning.ComparisonPropagation"
)

QGramsBlocking = autoclass("org.scify.jedai.blockbuilding.QGramsBlocking")
BlockFiltering = autoclass(
    "org.scify.jedai.blockprocessing.blockcleaning.BlockFiltering"
)
WeightedEdgePruning = autoclass(
    "org.scify.jedai.blockprocessing.comparisoncleaning.WeightedEdgePruning"
)
WeightingScheme = autoclass("org.scify.jedai.utilities.enumerations.WeightingScheme")

Block = autoclass("org.scify.jedai.datamodel.AbstractBlock")
BlocksPerformance = autoclass("org.scify.jedai.utilities.BlocksPerformance")


def get_profiles(
    file_path: str,
    *,
    Reader: EntityReader = EntityCSVReader,
    attribute_names_in_first_row: bool = True,
    separator: str = ",",
    id_index: int = 0,
) -> list[EntityProfile]:
    reader = Reader(file_path)
    reader.setAttributeNamesInFirstRow(attribute_names_in_first_row)
    reader.setSeparator(separator)
    reader.setIdIndex(id_index)

    profiles = reader.getEntityProfiles()
    return profiles


def get_duplicate_propagation(
    file_path: str,
    profiles_list: list[list[EntityProfile]],
    *,
    Reader: GroundTruthReader = GtCSVReader,
    separator: str = ",",
    ignore_first_row: bool = True,
) -> DuplicatePropagation:
    reader = Reader(file_path)
    reader.setIgnoreFirstRow(ignore_first_row)
    reader.setSeparator(separator)

    duplicate_pairs = reader.getDuplicatePairs(*profiles_list)
    if len(profiles_list) == 1:
        duplicate_propagation = UnilateralDuplicatePropagation(duplicate_pairs)
    else:
        duplicate_propagation = BilateralDuplicatePropagation(duplicate_pairs)

    return duplicate_propagation


def get_blocks_stats(
    blocks: list[Block],
    duplicate_propagation: DuplicatePropagation,
) -> dict[str, float]:
    blocks_stats = BlocksPerformance(blocks, duplicate_propagation)
    blocks_stats.setStatistics()

    return {
        "PC": blocks_stats.getPc(),
        "PQ": blocks_stats.getPq(),
        "F1": blocks_stats.getFMeasure(),
    }


def parameter_free_blocking_workflows(
    profiles_list: list[list[EntityProfile]],
) -> list[Block]:
    standard_blocking = StandardBlocking()
    comparisons_based_block_purging = ComparisonsBasedBlockPurging(True)
    comparison_propagation = ComparisonPropagation()

    blocks = standard_blocking.getBlocks(*profiles_list)
    blocks = comparisons_based_block_purging.refineBlocks(blocks)
    blocks = comparison_propagation.refineBlocks(blocks)

    return blocks


def default_qgrams_blocking_workflows(
    profiles_list: list[list[EntityProfile]],
) -> list[Block]:
    qgrams_blocking = QGramsBlocking(6)
    block_filtering = BlockFiltering(0.5)
    weighted_edge_pruning = WeightedEdgePruning(WeightingScheme.ECBS)

    blocks = qgrams_blocking.getBlocks(*profiles_list)
    blocks = block_filtering.refineBlocks(blocks)
    blocks = weighted_edge_pruning.refineBlocks(blocks)

    return blocks


BLOCKING_WORKFLOWS = {
    "parameter_free": parameter_free_blocking_workflows,
    "default_qgrams": default_qgrams_blocking_workflows,
}


def blocking_workflows(
    data_dir: str = "./data/blocking/cora",
    workflow: Literal["parameter_free", "default_qgrams"] = "default_qgrams",
) -> dict[str, float]:
    profiles_paths = map(str, sorted(Path(data_dir).glob("[1-2]*.csv")))
    profiles_list = [get_profiles(path) for path in profiles_paths]
    matches_path = str(Path(data_dir) / "matches.csv")
    duplicate_propagation = get_duplicate_propagation(matches_path, profiles_list)

    blocks = BLOCKING_WORKFLOWS[workflow](profiles_list)
    metrics = get_blocks_stats(blocks, duplicate_propagation)

    print(metrics)
    return metrics


if __name__ == "__main__":
    CLI(blocking_workflows)
