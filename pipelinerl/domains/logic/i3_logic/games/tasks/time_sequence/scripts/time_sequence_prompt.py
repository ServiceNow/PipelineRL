import random


def concat_schedule_description(raw_schedule: dict, is_chinese=True):
    time_des_free = ["仅在以下时间有空：", "以下时间空闲：", "除以下时间之外，都没空："]
    time_des_book = ["已预定了以下时间：", "以下时间已经占用：", "以下时间无法安排会议："]
    time_des_free_english = [
        "only available at the following times: ",
        "free at the following times: ",
        "unavailable except for the following times: ",
    ]
    time_des_book_english = [
        "already booked the following times: ",
        "schedule is occupied at the following times: ",
        "cannot schedule meetings at the following times: ",
    ]

    def trans_time(start, end, merge_str):
        base_hour, _ = (9, 0)
        start_hour = base_hour + start * 5 // 60
        start_minute = start * 5 % 60
        end_hour = base_hour + end * 5 // 60
        end_minute = end * 5 % 60
        return f"{start_hour:02}:{start_minute:02} {merge_str} {end_hour:02}:{end_minute:02}"

    def concat_one_schedule(days_schedules):
        weekday_map = {"一": "Monday", "二": "Tuesday", "三": "Wednesday", "四": "Thursday", "五": "Friday"}
        concat_string = ""
        for key in days_schedules:
            concat_string += f"星期{key}：" if is_chinese else f"{weekday_map[key]}: "
            if days_schedules[key][0] == "空闲":
                concat_string += random.choice(time_des_free) if is_chinese else random.choice(time_des_free_english)
            elif days_schedules[key][0] == "占用":
                concat_string += random.choice(time_des_book) if is_chinese else random.choice(time_des_book_english)
            merge_str = random.choice(["-"])
            for sche in days_schedules[key][1]:
                concat_string += trans_time(sche[0], sche[1], merge_str)
                concat_string += "、" if is_chinese else ", "
            concat_string += "\n"
        return concat_string

    final_string = ""
    for name in raw_schedule:
        final_string += name
        final_string += "、" if is_chinese else ", "
    final_string = final_string[:-1]
    final_string += (
        "的工作时间为工作日当地时间早上9点到下午5点，他们想在本周的某个时间一起开会。\n\n"
        if is_chinese
        else " work from 9 to 5 their local time on weekdays. They want to have a meeting some time this week."
    )
    for name in raw_schedule:
        final_string += (
            f"{name} 当地时间本周的日程安排如下：\n"
            if is_chinese
            else f"The schedule for {name}'s week in their local time is as follows: "
        )
        final_string += concat_one_schedule(raw_schedule[name])
    return final_string


def concat_rule_description(rules_desc: dict, is_chinese: bool = True):
    final_string = ""
    for name in rules_desc:
        if rules_desc[name]:
            final_string += f"{name}{rules_desc[name]}\n"
    final_string += (
        "如果会议必须在整点或半点开始，请给出可以安排的会议的最长时间（以分钟为单位），以及可安排的选项数(必须是最长时间的)。如果无法安排符合条件的会议，都回答0即可。"
        if is_chinese
        else "If the meeting must start on the hour or half-hour, please provide the maximum duration of the meeting that can be scheduled (in minutes), as well as the number of available options (which must be for the maximum duration).If no meetings that meet the conditions can be scheduled, respond with 0.\n"
    )
    return final_string


def prompt_timeSequence(raw_schedule, rules_desc, is_chinese=True):
    prompt = ""
    prompt = concat_schedule_description(raw_schedule=raw_schedule, is_chinese=is_chinese)
    prompt += concat_rule_description(rules_desc=rules_desc, is_chinese=is_chinese)
    prompt += (
        "\n\n请在回答的末尾将可安排的会议最长时间和选项数，按顺序放在[ ]中，无需其他内容，形如：\n[45, 3]"
        if is_chinese
        else "\n\nPlease put the maximum duration of the meeting that can be scheduled and the number of options, in that order, in [ ] at the end of your answer, with no other content, like this:\n[45, 3]"
    )
    return prompt
