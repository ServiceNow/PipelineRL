import argparse
import itertools
import json
import logging
import pathlib
import random
import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .....base.data import Data
from .....games.base.game import Game
from .....games.tasks.web_of_lies.scripts.web_of_lies_prompt import prompt_web_of_lies
from .....games.tasks.web_of_lies.scripts.web_of_lies_verifier import WebOfLiesVerifier

if TYPE_CHECKING:
    from typing import List

    class Person:
        id: int
        name: str
        location: str
        is_truth_teller: bool
        statements: List[Any]

        def reference(self, use_chinese: bool = False) -> str: ...


class Statement:
    def __init__(self, speaker_id: int):
        self.speaker_id = speaker_id

    def gen_real_statement(self, people: List["Person"], use_chinese: bool) -> str:
        raise NotImplementedError("子类必须实现此方法")

    @classmethod
    def New(cls, speaker_id: int, *args, **kwargs):
        raise NotImplementedError("子类必须实现此方法")


class SimpleStatement(Statement):
    def __init__(self, speaker_id: int, target_id: int):
        super().__init__(speaker_id)
        self.target_id = target_id

    @classmethod
    def New(cls, speaker_id: int, target_id: int, people: List["Person"]):
        return cls(speaker_id, target_id)

    def gen_real_statement(self, people: List["Person"], use_chinese: bool) -> str:
        speaker = people[self.speaker_id]
        target = people[self.target_id]

        speaker_ref = speaker.reference(use_chinese)
        target_ref = target.reference(use_chinese)

        is_claim_truth = target.is_truth_teller
        if not speaker.is_truth_teller:
            is_claim_truth = not is_claim_truth

        if use_chinese:
            truth_text = "真话" if is_claim_truth else "假话"
            return f"{speaker_ref}说{target_ref}说的是{truth_text}。"
        else:
            if is_claim_truth:
                return f"{speaker_ref} says {target_ref} tells the truth."
            else:
                return f"{speaker_ref} says {target_ref} lies."


class AtLeastOneStatement(Statement):
    def __init__(self, speaker_id: int, target_ids: List[int]):
        super().__init__(speaker_id)
        self.target_ids = target_ids

    @classmethod
    def New(cls, speaker_id: int, target_ids: List[int], people: List["Person"]):
        if len(target_ids) != 2:
            raise ValueError("AtLeastOneStatement需要正好两个目标ID")
        return cls(speaker_id, target_ids)

    def gen_real_statement(self, people: List["Person"], use_chinese: bool) -> str:
        speaker = people[self.speaker_id]
        targets = [people[tid] for tid in self.target_ids]

        speaker_ref = speaker.reference(use_chinese)
        target_refs = [t.reference(use_chinese) for t in targets]

        actual_at_least_one_truth = any(t.is_truth_teller for t in targets)

        is_claim_truth = actual_at_least_one_truth
        if not speaker.is_truth_teller:
            is_claim_truth = not is_claim_truth

        if use_chinese:
            truth_text = "真话" if is_claim_truth else "假话"
            return f"{speaker_ref}说{target_refs[0]}和{target_refs[1]}中至少有一个总是说{truth_text}。"
        else:
            truth_text = "truth" if is_claim_truth else "lies"
            return f"{speaker_ref} says at least one of {target_refs[0]} and {target_refs[1]} always tells the {truth_text}."


class SameTypeStatement(Statement):
    def __init__(self, speaker_id: int, target_id: int):
        super().__init__(speaker_id)
        self.target_id = target_id

    @classmethod
    def New(cls, speaker_id: int, target_id: int, people: List["Person"]):
        return cls(speaker_id, target_id)

    def gen_real_statement(self, people: List["Person"], use_chinese: bool) -> str:
        speaker = people[self.speaker_id]
        target = people[self.target_id]

        speaker_ref = speaker.reference(use_chinese)
        target_ref = target.reference(use_chinese)

        actual_same_type = speaker.is_truth_teller == target.is_truth_teller

        is_claim_same_type = actual_same_type
        if not speaker.is_truth_teller:
            is_claim_same_type = not is_claim_same_type

        if use_chinese:
            type_text = "是" if is_claim_same_type else "不是"
            return f"{speaker_ref}说自己和{target_ref}{type_text}一类人。"
        else:
            type_text = "is" if is_claim_same_type else "is not"
            return f"{speaker_ref} says they {type_text} the same type as {target_ref}."


class Person:
    def __init__(self, id: int, name: str, location: str, is_truth_teller: bool = None):
        self.id = id
        self.name = name
        self.location = location
        self.is_truth_teller = is_truth_teller
        self.statements = []

    def reference(self, use_chinese: bool = False) -> str:
        if random.random() < 0.7:
            if use_chinese:
                return f"在{self.location}的人"
            else:
                return f"the person at the {self.location}"
        else:
            return self.name


class WebOfLies(Game):
    def __init__(self):
        super().__init__("Web of Lies", WebOfLiesVerifier)

        self.chinese_names = [
            "李明",
            "王华",
            "张伟",
            "赵芳",
            "钱程",
            "孙文",
            "周丽",
            "吴强",
            "郑晓",
            "冯雅",
            "陈刚",
            "褚健",
            "卫平",
            "蒋红",
            "沈阳",
            "韩梅",
            "杨光",
            "朱琳",
            "秦松",
            "许江",
            "何方",
            "吕坤",
            "施汉",
            "张丽",
            "孔明",
            "曹操",
            "严肃",
            "华光",
            "金铭",
            "魏然",
        ]

        self.english_names = [
            "Emma",
            "Noah",
            "Olivia",
            "Liam",
            "Ava",
            "Jacob",
            "Sophia",
            "Mason",
            "Isabella",
            "William",
            "Mia",
            "Ethan",
            "Charlotte",
            "James",
            "Amelia",
            "Alexander",
            "Harper",
            "Michael",
            "Evelyn",
            "Benjamin",
            "Abigail",
            "Daniel",
            "Emily",
            "Matthew",
            "Elizabeth",
            "Henry",
            "Sofia",
            "Joseph",
            "Madison",
            "Samuel",
        ]

        self.chinese_locations = [
            "公园",
            "医院",
            "学校",
            "图书馆",
            "餐厅",
            "超市",
            "电影院",
            "体育馆",
            "咖啡店",
            "博物馆",
            "广场",
            "花园",
            "车站",
            "机场",
            "银行",
            "商场",
            "酒店",
            "游泳池",
            "动物园",
            "美术馆",
            "剧院",
            "书店",
            "健身房",
            "办公室",
            "社区中心",
            "夜市",
            "水族馆",
            "植物园",
            "展览中心",
            "游乐园",
        ]

        self.english_locations = [
            "park",
            "hospital",
            "school",
            "library",
            "restaurant",
            "supermarket",
            "movie theater",
            "stadium",
            "cafe",
            "museum",
            "square",
            "garden",
            "train station",
            "airport",
            "bank",
            "mall",
            "hotel",
            "swimming pool",
            "zoo",
            "art gallery",
            "theater",
            "bookstore",
            "gym",
            "office",
            "community center",
            "night market",
            "aquarium",
            "botanical garden",
            "exhibition center",
            "amusement park",
        ]

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        num_person: int = 10,
        difficulty: int = 3,
        statement_type: int = 0,
    ):
        logger.info(
            f"开始生成题目，参数：num_of_questions={num_of_questions}, max_attempts={max_attempts}, num_person={num_person}, difficulty={difficulty}, statement_type={statement_type}"
        )

        if num_person <= 3:
            raise ValueError("人物数量必须大于3")
        if difficulty < 1 or difficulty > 5:
            raise ValueError("难度必须在1-5之间")
        if statement_type not in [0, 1]:
            raise ValueError("陈述类型必须为0或1")

        game_data_list = []
        attempts = 0

        while len(game_data_list) < num_of_questions and attempts < max_attempts:
            try:
                attempts += 1
                logger.info(f"第 {attempts} 次尝试生成题目")

                use_chinese = random.choice([True, False])

                tar_people_count = num_person // 2

                logger.info(f"当前生成参数：使用中文={use_chinese}, 目标人物数量={tar_people_count}")

                question, answer, metadata = self._generate_game_instance(
                    num_person, tar_people_count, difficulty, use_chinese, statement_type
                )

                logger.info(f"生成的题目：{question}")
                logger.info(f"答案：{answer}")

                game_data = Data(question=question, answer=answer, difficulty=difficulty, metadata=metadata)

                game_data_list.append(game_data)
                logger.info(f"成功生成第 {len(game_data_list)}/{num_of_questions} 个题目")

            except Exception as e:
                logger.error(f"生成题目时出错: {str(e)}")
                continue

        return game_data_list

    def _statement_to_dict(self, statement: Statement) -> Dict:
        if isinstance(statement, SimpleStatement):
            return {"type": "simple", "speaker_id": statement.speaker_id, "target_id": statement.target_id}
        elif isinstance(statement, AtLeastOneStatement):
            return {"type": "at_least_one", "speaker_id": statement.speaker_id, "target_ids": statement.target_ids}
        elif isinstance(statement, SameTypeStatement):
            return {"type": "same_type", "speaker_id": statement.speaker_id, "target_id": statement.target_id}
        else:
            return {"type": "unknown"}

    def _generate_game_instance(
        self, num_person: int, tar_people_count: int, difficulty: int, use_chinese: bool, statement_type: int = 0
    ) -> Tuple[str, str, Dict]:
        if use_chinese:
            names = random.sample(self.chinese_names, num_person)
            locations = random.sample(self.chinese_locations, num_person)
        else:
            names = random.sample(self.english_names, num_person)
            locations = random.sample(self.english_locations, num_person)

        people = [
            Person(id=i, name=names[i], location=locations[i], is_truth_teller=random.choice([True, False]))
            for i in range(num_person)
        ]

        tar_people_count = max(1, min(num_person, tar_people_count))
        tar_people_indices = random.sample(range(num_person), tar_people_count)
        tar_people = [people[i] for i in tar_people_indices]

        base_max = num_person // 2
        difficulty_factor = (6 - difficulty) / 5
        max_truth_narration = min(max(1, int(base_max * difficulty_factor)), num_person - tar_people_count)

        truth_narration_count = random.randint(0, max_truth_narration)

        non_target_indices = [i for i in range(num_person) if i not in tar_people_indices]
        truth_narration_indices = random.sample(non_target_indices, min(truth_narration_count, len(non_target_indices)))

        for i in range(num_person):
            possible_targets = [j for j in range(num_person) if j != i]

            prob = random.random()
            if prob < 0.1:
                statement_count = 0
            elif prob < 0.8:
                statement_count = 1
            elif prob < 0.95:
                statement_count = 2
            else:
                statement_count = 3

            if statement_count == 0 or not possible_targets:
                continue

            if statement_type == 0 or (statement_type == 1 and random.random() > 0.1):
                targets = random.sample(possible_targets, min(statement_count, len(possible_targets)))

                for target_id in targets:
                    statement = SimpleStatement.New(i, target_id, people)
                    people[i].statements.append(statement)
            else:
                if random.random() < 0.5 and len(possible_targets) >= 2:
                    target_ids = random.sample(possible_targets, 2)
                    statement = AtLeastOneStatement.New(i, target_ids, people)
                else:
                    target_id = random.choice(possible_targets)
                    statement = SameTypeStatement.New(i, target_id, people)

                people[i].statements.append(statement)

        for i in range(num_person):
            is_mentioned = i in truth_narration_indices

            if not is_mentioned:
                for person in people:
                    for statement in person.statements:
                        if isinstance(statement, SimpleStatement) and statement.target_id == i:
                            is_mentioned = True
                            break
                        elif isinstance(statement, AtLeastOneStatement) and i in statement.target_ids:
                            is_mentioned = True
                            break
                        elif isinstance(statement, SameTypeStatement) and statement.target_id == i:
                            is_mentioned = True
                            break

                    if is_mentioned:
                        break

            if not is_mentioned:
                speaker_idx = random.choice([j for j in range(num_person) if j != i])

                if statement_type == 0 or random.random() > 0.1:
                    statement = SimpleStatement.New(speaker_idx, i, people)
                else:
                    if (
                        random.random() < 0.5
                        and len([j for j in range(num_person) if j != speaker_idx and j != i]) >= 1
                    ):
                        other_targets = [j for j in range(num_person) if j != speaker_idx and j != i]
                        other_target = random.choice(other_targets)
                        target_ids = [i, other_target]
                        statement = AtLeastOneStatement.New(speaker_idx, target_ids, people)
                    else:
                        statement = SameTypeStatement.New(speaker_idx, i, people)

                people[speaker_idx].statements.append(statement)

        solution_count, all_solutions, valid_solutions = self._find_solution_count(
            people, tar_people_indices, truth_narration_indices
        )
        if solution_count == 0:
            raise ValueError("游戏无解，需要重新生成")
        elif solution_count > 1:
            raise ValueError("游戏有多解，需要重新生成")

        statements = []

        for person in people:
            if use_chinese:
                statements.append(f"{person.name}在{person.location}。")
            else:
                statements.append(f"{person.name} is at the {person.location}.")

        for idx in truth_narration_indices:
            person = people[idx]
            reference = person.reference(use_chinese)

            if use_chinese:
                templates_truth = [
                    f"{reference}说的是真话。",
                    f"{reference}总是诚实的。",
                    f"{reference}从不撒谎。",
                    f"你可以相信{reference}说的每一句话。",
                ]
                templates_lie = [
                    f"{reference}说的是假话。",
                    f"{reference}总是撒谎。",
                    f"{reference}从不说真话。",
                    f"不要相信{reference}说的任何话。",
                ]
                if person.is_truth_teller:
                    statements.append(random.choice(templates_truth))
                else:
                    statements.append(random.choice(templates_lie))
            else:
                templates_truth = [
                    f"{reference} tells the truth.",
                    f"{reference} is always honest.",
                    f"{reference} never lies.",
                    f"You can trust everything {reference} says.",
                ]
                templates_lie = [
                    f"{reference} lies.",
                    f"{reference} always lies.",
                    f"{reference} never tells the truth.",
                    f"Don't trust anything {reference} says.",
                ]
                if person.is_truth_teller:
                    statements.append(random.choice(templates_truth))
                else:
                    statements.append(random.choice(templates_lie))

        for person in people:
            for statement in person.statements:
                statement_text = statement.gen_real_statement(people, use_chinese)
                statements.append(statement_text)

        irrelevant_count = random.randint(1, 2 + difficulty)
        for _ in range(irrelevant_count):
            if random.random() < 0.5:
                irrelevant = self._generate_irrelevant_statement(people, use_chinese)
            else:
                if len(people) >= 2:
                    person1, person2 = random.sample(people, 2)
                    irrelevant = self._generate_person_to_person_statement(person1, person2, use_chinese)
                else:
                    irrelevant = self._generate_irrelevant_statement(people, use_chinese)

            statements.append(irrelevant)

        random.shuffle(statements)

        tar_people_references = [person.reference(use_chinese) for person in tar_people]

        question = prompt_web_of_lies(statements, tar_people_references, num_person, use_chinese)

        answer_values = []
        for person in tar_people:
            if person.is_truth_teller:
                answer_values.append("yes" if not use_chinese else "是")
            else:
                answer_values.append("no" if not use_chinese else "否")

        answer = ", ".join(answer_values)

        metadata = {
            "trace_id": str(uuid.uuid4()),
            "num_person": num_person,
            "difficulty": difficulty,
            "people": [
                {
                    "id": p.id,
                    "name": p.name,
                    "location": p.location,
                    "is_truth_teller": p.is_truth_teller,
                    "is_target": p.id in tar_people_indices,
                    "has_truth_narration": p.id in truth_narration_indices,
                    "statements": [self._statement_to_dict(stmt) for stmt in p.statements],
                }
                for p in people
            ],
            "statements": statements,
            "use_chinese": use_chinese,
            "all_possible_solutions": all_solutions,
            "valid_solutions": {",".join(str(x) for x in k): v for k, v in valid_solutions.items()},
        }

        return question, answer, metadata

    def _find_solution_count(
        self, people: List["Person"], tar_people_indices: List[int], truth_narration_indices: List[int]
    ) -> Tuple[int, List[Dict[str, bool]], Dict[Tuple[bool, ...], int]]:
        logger.info(
            f"开始计算解的个数，目标人物索引：{tar_people_indices}，真假表述旁白索引：{truth_narration_indices}"
        )

        all_enum_indices = [i for i in range(len(people)) if i not in truth_narration_indices]
        logger.debug(f"需要枚举的人物索引：{all_enum_indices}")

        valid_solutions = {}
        all_solutions = []

        total_possibilities = 2 ** len(all_enum_indices)
        logger.info(f"总共需要枚举 {total_possibilities} 种可能性")

        for possibility in itertools.product([True, False], repeat=len(all_enum_indices)):
            assumed_truth_map = {}
            for i, person in enumerate(people):
                if i in truth_narration_indices:
                    assumed_truth_map[i] = person.is_truth_teller

            for i, is_truth in enumerate(possibility):
                idx = all_enum_indices[i]
                assumed_truth_map[idx] = is_truth

            if self._check_statements_consistency_with_map(people, assumed_truth_map):
                target_solution = tuple(assumed_truth_map[idx] for idx in tar_people_indices)
                if target_solution not in valid_solutions:
                    solution_dict = {people[idx].name: assumed_truth_map[idx] for idx in tar_people_indices}
                    all_solutions.append(solution_dict)
                valid_solutions[target_solution] = valid_solutions.get(target_solution, 0) + 1

        logger.info(f"找到 {len(valid_solutions)} 个不同的解")
        if valid_solutions:
            logger.debug(f"有效解：{valid_solutions}")

        return len(valid_solutions), all_solutions, valid_solutions

    def _check_statements_consistency_with_map(
        self, people: List["Person"], assumed_truth_map: Dict[int, bool]
    ) -> bool:
        for person in people:
            if person.id not in assumed_truth_map:
                continue

            speaker_is_truth = assumed_truth_map[person.id]

            for statement in person.statements:
                if isinstance(statement, SimpleStatement):
                    target_id = statement.target_id

                    if target_id not in assumed_truth_map:
                        continue

                    target_is_truth = assumed_truth_map[target_id]

                    actual_speaker = people[person.id]
                    actual_target = people[target_id]
                    is_claim_truth = actual_target.is_truth_teller
                    if not actual_speaker.is_truth_teller:
                        is_claim_truth = not is_claim_truth

                    expected_is_claim_truth = target_is_truth
                    if not speaker_is_truth:
                        expected_is_claim_truth = not expected_is_claim_truth

                    if is_claim_truth != expected_is_claim_truth:
                        return False

                elif isinstance(statement, AtLeastOneStatement):
                    if not all(tid in assumed_truth_map for tid in statement.target_ids):
                        continue

                    actual_speaker = people[person.id]
                    actual_targets = [people[tid] for tid in statement.target_ids]
                    actual_at_least_one_truth = any(t.is_truth_teller for t in actual_targets)
                    is_claim_truth = actual_at_least_one_truth
                    if not actual_speaker.is_truth_teller:
                        is_claim_truth = not is_claim_truth

                    assumed_targets_states = [assumed_truth_map[tid] for tid in statement.target_ids]
                    assumed_at_least_one_truth = any(assumed_targets_states)
                    expected_is_claim_truth = assumed_at_least_one_truth
                    if not speaker_is_truth:
                        expected_is_claim_truth = not expected_is_claim_truth

                    if is_claim_truth != expected_is_claim_truth:
                        return False

                elif isinstance(statement, SameTypeStatement):
                    target_id = statement.target_id

                    if target_id not in assumed_truth_map:
                        continue

                    actual_speaker = people[person.id]
                    actual_target = people[target_id]
                    actual_same_type = actual_speaker.is_truth_teller == actual_target.is_truth_teller
                    is_claim_same_type = actual_same_type
                    if not actual_speaker.is_truth_teller:
                        is_claim_same_type = not is_claim_same_type

                    assumed_same_type = speaker_is_truth == assumed_truth_map[target_id]
                    expected_is_claim_same_type = assumed_same_type
                    if not speaker_is_truth:
                        expected_is_claim_same_type = not expected_is_claim_same_type

                    if is_claim_same_type != expected_is_claim_same_type:
                        return False

        return True

    def _check_statements_consistency(self, people: List["Person"]) -> bool:
        truth_map = {i: person.is_truth_teller for i, person in enumerate(people) if person.is_truth_teller is not None}

        return self._check_statements_consistency_with_map(people, truth_map)

    def _generate_person_to_person_statement(self, person: "Person", target: "Person", use_chinese: bool) -> str:
        person_ref = person.reference(use_chinese)
        target_ref = target.reference(use_chinese)

        if use_chinese:
            templates = [
                f"{person_ref}说{target_ref}很友好。",
                f"{person_ref}表示认识{target_ref}。",
                f"{person_ref}说{target_ref}是个好人。",
                f"{person_ref}提到{target_ref}很聪明。",
            ]
        else:
            templates = [
                f"{person_ref} says {target_ref} is friendly.",
                f"{person_ref} claims to know {target_ref}.",
                f"{person_ref} says {target_ref} is a good person.",
                f"{person_ref} mentions that {target_ref} is smart.",
            ]

        statement = random.choice(templates)

        return statement

    def _generate_irrelevant_statement_about_person(self, person: "Person", target: "Person", use_chinese: bool) -> str:
        person_ref = person.reference(use_chinese)
        target_ref = target.reference(use_chinese)

        if use_chinese:
            templates = [
                f"{person_ref}说{target_ref}喜欢看电影。",
                f"{person_ref}提到{target_ref}养了一只猫。",
                f"{person_ref}说{target_ref}喜欢读书。",
                f"{person_ref}表示{target_ref}昨天看到了一辆消防车。",
                f"{person_ref}认为{target_ref}对天气很敏感。",
                f"{person_ref}听说{target_ref}刚从旅行回来。",
                f"{person_ref}说{target_ref}喜欢吃水果。",
            ]
        else:
            templates = [
                f"{person_ref} says {target_ref} likes watching movies.",
                f"{person_ref} mentions that {target_ref} has a cat.",
                f"{person_ref} says {target_ref} enjoys reading books.",
                f"{person_ref} states that {target_ref} saw a firetruck yesterday.",
                f"{person_ref} thinks {target_ref} is sensitive to weather.",
                f"{person_ref} heard that {target_ref} just returned from a trip.",
                f"{person_ref} says {target_ref} likes eating fruits.",
            ]

        return random.choice(templates)

    def _generate_irrelevant_statement(self, people: List["Person"], use_chinese: bool) -> str:
        person = random.choice(people)
        person_ref = person.reference(use_chinese)

        if use_chinese:
            irrelevant_templates = [
                f"{person_ref}总是很准时。",
                f"{person_ref}从不对任何人发脾气。",
                f"{person_ref}对每个人都很善良。",
                f"{person_ref}从不隐瞒自己的癖好。",
                f"{person_ref}做事很有原则。",
                f"{person_ref}对朋友很忠诚。",
                f"{person_ref}从不做违背良心的事。",
                f"{person_ref}做事情很认真负责。",
                f"{person_ref}昨天遇到了一件有趣的事。",
                f"{person_ref}最近在思考人生。",
                f"{person_ref}认为自己的朋友在撒谎。",
                f"{person_ref}觉得孩子们编造的故事很有趣。",
                f"{person_ref}发现同事在工作报告中写错了数据。",
                f"{person_ref}怀疑邻居家的狗偷吃了他的早餐。",
                f"{person_ref}觉得电视剧里的剧情太假了。",
                f"{person_ref}认为朋友在开玩笑。",
                f"{person_ref}发现市场上有人在卖假货。",
                f"{person_ref}觉得小说里的情节不够真实。",
            ]
        else:
            irrelevant_templates = [
                f"{person_ref} is always punctual.",
                f"{person_ref} never gets angry at anyone.",
                f"{person_ref} is nice to everyone.",
                f"{person_ref} never hides their habits.",
                f"{person_ref} has strong principles.",
                f"{person_ref} is loyal to friends.",
                f"{person_ref} never does anything against their conscience.",
                f"{person_ref} is very responsible.",
                f"{person_ref} had an interesting encounter yesterday.",
                f"{person_ref} has been contemplating life lately.",
                f"{person_ref} thinks their friend is lying.",
                f"{person_ref} finds children's made-up stories amusing.",
                f"{person_ref} discovered errors in their colleague's work report.",
                f"{person_ref} suspects the neighbor's dog stole their breakfast.",
                f"{person_ref} thinks the TV drama is too unrealistic.",
                f"{person_ref} believes their friend is joking.",
                f"{person_ref} found counterfeit goods in the market.",
                f"{person_ref} thinks the novel's plot isn't convincing.",
            ]

        return random.choice(irrelevant_templates)

    def extract_answer(self, test_solution: str) -> str:
        if not test_solution:
            return ""

        cn_patterns = [
            r"答案是[：:]\s*\*\*([^*]+)\*\*[.。]*$",
        ]

        en_patterns = [
            r"[Tt]he answer is[：:=]\s*\*\*([^*]+)\*\*[.。]*$",
        ]

        patterns = cn_patterns + en_patterns

        for pattern in patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                return matches[-1].strip()

        lines = test_solution.strip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            bold_match = re.search(r"\*\*([^*]+)\*\*", last_line)
            if bold_match:
                return bold_match.group(1).strip()

            answer_match = re.search(r"(?:答案是|[Tt]he answer is)[：:=]?\s*(.*?)(?:[.。]|$)", last_line)
            if answer_match:
                return answer_match.group(1).strip()

        yes_no_pattern = r"(?:\b(?:yes|no|是|否)\b[,，\s]*)+"
        matches = re.findall(yes_no_pattern, test_solution.lower())
        if matches:
            return matches[-1].strip()

        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="谎言之网游戏生成器")
    parser.add_argument("--num_of_data", type=int, default=100, help="生成的题目数量")
    parser.add_argument("--max_attempts", type=int, default=1000, help="每个题目的最大尝试次数")
    parser.add_argument("--num_person", type=int, default=10, help="人物数量")
    parser.add_argument("--difficulty", type=int, default=3, help="难度级别，1-5")
    parser.add_argument("--statement_type", type=int, default=0, help="陈述类型，0为基础类型，1为扩展类型")
    args = parser.parse_args()

    data_dir = pathlib.Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    output_dir = (
        data_dir
        / f"num_person_{args.num_person}/difficulty_{args.difficulty}/statement_type_{args.statement_type}/num_of_data_{args.num_of_data}"
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.jsonl"

    game = WebOfLies()

    game_data_list = game.generate(
        num_of_questions=args.num_of_data,
        max_attempts=args.max_attempts,
        num_person=args.num_person,
        difficulty=args.difficulty,
        statement_type=args.statement_type,
    )

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for game_data in game_data_list:
                f.write(json.dumps(game_data.to_json(), ensure_ascii=False) + "\n")
    except Exception:
        pass
