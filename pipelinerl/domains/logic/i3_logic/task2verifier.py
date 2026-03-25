from .corpus.tasks.arc_agi.scripts.arc_agi_verifier import ArcAGIVerifier
from .corpus.tasks.bbeh.scripts.bbeh_verifier import BBEHVerifier
from .corpus.tasks.bbh.scripts.boolean_expressions_verifier import BBHBooleanExpressionsVerifier
from .corpus.tasks.bbh.scripts.causal_judgement_verifier import BBHCausalJudgementVerifier
from .corpus.tasks.bbh.scripts.date_understanding_verifier import BBHDateUnderstandingVerifier
from .corpus.tasks.bbh.scripts.dyck_languages_verifier import BBHDyckLanguagesVerifier
from .corpus.tasks.bbh.scripts.formal_fallacies_verifier import BBHFormalFallaciesVerifier
from .corpus.tasks.bbh.scripts.multistep_arithmetic_two_verifier import BBHMultistepArithmeticVerifier
from .corpus.tasks.bbh.scripts.sports_understanding_verifier import BBHSportsUnderstandingVerifier
from .corpus.tasks.bbh.scripts.web_of_lies_verifier import BBHWebOfLiesVerifier
from .corpus.tasks.bbh.scripts.word_sorting_verifier import BBHWordSortingVerifier
from .corpus.tasks.gpqa.scripts.gpqa_verifier import GPQAVerifier
from .corpus.tasks.zebra_puzzle.scripts.zebra_puzzle_verifier import ZebraPuzzleVerifier
from .games.tasks.arrow_maze.scripts.arrow_maze_verifier import ArrowMazeVerifier
from .games.tasks.boolean_expressions.scripts.boolean_expressions_verifier import BooleanExpressionsVerifier
from .games.tasks.buggy_tables.scripts.game_of_buggy_tables_verifier import BuggyTableVerifier
from .games.tasks.calcudoko.scripts.calcudoko_verifier import CalcudokoVerifier
from .games.tasks.campsite.scripts.campsite_verifier import CampsiteVerifier
from .games.tasks.cipher.scripts.cipher_verifier import CipherVerifier
from .games.tasks.cryptarithm.scripts.cryptarithm_verifier import CryptarithmVerifier
from .games.tasks.dyck_language.scripts.dyck_language_verifier import DyckLanguageVerifier
from .games.tasks.dyck_language_errors.scripts.dyck_language_errors_verifier import DyckLanguageErrorsVerifier
from .games.tasks.dyck_language_reasoning_errors.scripts.dyck_language_reasoning_errors_verifier import (
    DyckLanguageReasoningErrorsVerifier,
)
from .games.tasks.futoshiki.scripts.futoshiki_verifier import FutoshikiVerifier
from .games.tasks.game_of_24.scripts.game_of_24_verifier import GameOf24Verifier
from .games.tasks.goods_exchange.scripts.goods_exchange_verifier import GoodsExchangeVerifier
from .games.tasks.kukurasu.scripts.kukurasu_verifier import KukurasuVerifier
from .games.tasks.math_path.scripts.math_path_verifier import MathPathVerifier
from .games.tasks.minesweeper.scripts.minesweeper_verifier import MinesweeperVerifier
from .games.tasks.norinori.scripts.norinori_verifier import NorinoriVerifier
from .games.tasks.number_wall.scripts.number_wall_verifier import NumberWallVerifier
from .games.tasks.numbrix.scripts.numbrix_verifier import NumbrixVerifier
from .games.tasks.object_counting.scripts.object_counting_verifier import ObjectCountingVerifier
from .games.tasks.object_properties.scripts.object_properties_verifier import ObjectPropertiesVerifier
from .games.tasks.operation.scripts.operation_verifier import OperationVerifier
from .games.tasks.skyscraper_puzzle.scripts.skyscraper_puzzle_verifier import SkyscraperPuzzleVerifier
from .games.tasks.space_reasoning.scripts.space_reasoning_verifier import SpaceReasoningVerifier
from .games.tasks.space_reasoning_tree.scripts.space_reasoning_tree_verifier import SpaceReasoningTreeVerifier
from .games.tasks.star_placement_puzzle.scripts.star_placement_puzzle_verifier import StarPlacementPuzzleVerifier
from .games.tasks.sudoku.scripts.sudoku_verifier import SudokuVerifier
from .games.tasks.survo.scripts.survo_verifier import SurvoVerifier
from .games.tasks.time_sequence.scripts.time_sequence_verifier import TimeSequenceVerifier
from .games.tasks.web_of_lies.scripts.web_of_lies_verifier import WebOfLiesVerifier
from .games.tasks.word_sorting.scripts.word_sorting_verifier import WordSortingVerifier
from .games.tasks.word_sorting_mistake.scripts.word_sorting_mistake_verifier import WordSortingMistakeVerifier
from .games.tasks.wordscapes.scripts.wordscapes_verifier import WordscapesVerifier

bbh_classes = {
    "bbh_boolean_expressions": BBHBooleanExpressionsVerifier,
    "bbh_causal_judgement": BBHCausalJudgementVerifier,
    "bbh_date_understanding": BBHDateUnderstandingVerifier,
    "bbh_disambiguation_qa": BBHDateUnderstandingVerifier,
    "bbh_dyck_languages": BBHDyckLanguagesVerifier,
    "bbh_formal_fallacies": BBHFormalFallaciesVerifier,
    "bbh_geometric_shapes": BBHDateUnderstandingVerifier,
    "bbh_hyperbaton": BBHDateUnderstandingVerifier,
    "bbh_logical_deduction_five_objects": BBHDateUnderstandingVerifier,
    "bbh_logical_deduction_seven_objects": BBHDateUnderstandingVerifier,
    "bbh_logical_deduction_three_objects": BBHDateUnderstandingVerifier,
    "bbh_movie_recommendation": BBHDateUnderstandingVerifier,
    "bbh_multistep_arithmetic_two": BBHMultistepArithmeticVerifier,
    "bbh_navigate": BBHCausalJudgementVerifier,
    "bbh_object_counting": BBHMultistepArithmeticVerifier,
    "bbh_penguins_in_a_table": BBHDateUnderstandingVerifier,
    "bbh_reasoning_about_colored_objects": BBHDateUnderstandingVerifier,
    "bbh_ruin_names": BBHDateUnderstandingVerifier,
    "bbh_salient_translation_error_detection": BBHDateUnderstandingVerifier,
    "bbh_snarks": BBHDateUnderstandingVerifier,
    "bbh_sports_understanding": BBHSportsUnderstandingVerifier,
    "bbh_temporal_sequences": BBHDateUnderstandingVerifier,
    "bbh_tracking_shuffled_objects_five_objects": BBHDateUnderstandingVerifier,
    "bbh_tracking_shuffled_objects_seven_objects": BBHDateUnderstandingVerifier,
    "bbh_tracking_shuffled_objects_three_objects": BBHDateUnderstandingVerifier,
    "bbh_web_of_lies": BBHWebOfLiesVerifier,
    "bbh_word_sorting": BBHWordSortingVerifier,
}
bbeh_classes = {
    "bbeh_boardgame_qa": BBEHVerifier,
    "bbeh_boolean_expressions": BBEHVerifier,
    "bbeh_buggy_tables": BBEHVerifier,
    "bbeh_causal_understanding": BBEHVerifier,
    "bbeh_disambiguation_qa": BBEHVerifier,
    "bbeh_dyck_languages": BBEHVerifier,
    "bbeh_geometric_shapes": BBEHVerifier,
    "bbeh_hyperbaton": BBEHVerifier,
    "bbeh_linguini": BBEHVerifier,
    "bbeh_movie_recommendation": BBEHVerifier,
    "bbeh_multistep_arithmetic": BBEHVerifier,
    "bbeh_nycc": BBEHVerifier,
    "bbeh_object_counting": BBEHVerifier,
    "bbeh_object_properties": BBEHVerifier,
    "bbeh_sarc_triples": BBEHVerifier,
    "bbeh_shuffled_objects": BBEHVerifier,
    "bbeh_spatial_reasoning": BBEHVerifier,
    "bbeh_sportqa": BBEHVerifier,
    "bbeh_temporal_sequence": BBEHVerifier,
    "bbeh_time_arithmetic": BBEHVerifier,
    "bbeh_web_of_lies": BBEHVerifier,
    "bbeh_word_sorting": BBEHVerifier,
    "bbeh_zebra_puzzles": BBEHVerifier,
}
verifier_classes = {
    "arrow_maze": ArrowMazeVerifier,
    "boolean_expressions": BooleanExpressionsVerifier,
    "buggy_tables": BuggyTableVerifier,
    "calcudoko": CalcudokoVerifier,
    "campsite": CampsiteVerifier,
    "cipher": CipherVerifier,
    "cryptarithm": CryptarithmVerifier,
    "dyck_language": DyckLanguageVerifier,
    "dyck_language_errors": DyckLanguageErrorsVerifier,
    "dyck_language_reasoning_errors": DyckLanguageReasoningErrorsVerifier,
    "futoshiki": FutoshikiVerifier,
    "goods_exchange": GoodsExchangeVerifier,
    "gpqa_diamond": GPQAVerifier,
    "kukurasu": KukurasuVerifier,
    "math_path": MathPathVerifier,
    "arc_agi": ArcAGIVerifier,
    "arc_agi_2": ArcAGIVerifier,
    "mathador": GameOf24Verifier,
    "minesweeper": MinesweeperVerifier,
    "norinori": NorinoriVerifier,
    "number_wall": NumberWallVerifier,
    "numbrix": NumbrixVerifier,
    "object_counting": ObjectCountingVerifier,
    "object_properties": ObjectPropertiesVerifier,
    "operation": OperationVerifier,
    "skyscraper_puzzle": SkyscraperPuzzleVerifier,
    "space_reasoning": SpaceReasoningVerifier,
    "space_reasoning_tree": SpaceReasoningTreeVerifier,
    "star_placement_puzzle": StarPlacementPuzzleVerifier,
    "sudoku": SudokuVerifier,
    "survo": SurvoVerifier,
    "time_sequence": TimeSequenceVerifier,
    "web_of_lies": WebOfLiesVerifier,
    "word_sorting": WordSortingVerifier,
    "word_sorting_mistake": WordSortingMistakeVerifier,
    "wordscapes": WordscapesVerifier,
    "zebra_puzzle": ZebraPuzzleVerifier,
    **bbeh_classes,
    **bbh_classes,
}

__all__ = ["verifier_classes"]
