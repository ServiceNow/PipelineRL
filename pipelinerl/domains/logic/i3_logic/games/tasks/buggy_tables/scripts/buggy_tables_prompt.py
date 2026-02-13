import random

query_templates = {
    "Compute the absolute difference between the {stat_type} of column {col1} and column {col2}, {conditions}. Round the result to 2 decimal places. The answer MUST always have exactly 2 decimal places (e.g., 42.00, 0.50). If it cannot be determined whether the condition is satisfied or not (e.g., due to null values that cannot be inferred from the rest of the table), assume the condition is not satisfied. If a value is null, ignore it from the computations. {special_instruction}": "en",
    "Calculate the absolute difference between the {stat_type} of {col1} and {col2}, where {conditions}. The result should be rounded to 2 decimal places (e.g., 42.00, 0.50). If conditions cannot be verified due to null values, assume they are not satisfied. Null values should be excluded from calculations. {special_instruction}": "en",
    "Determine the absolute difference between the {stat_type} values of columns {col1} and {col2}, for rows where {conditions}. Round your answer to 2 decimal places, always showing both decimal digits (e.g., 42.00). Exclude null values from calculations and assume conditions with nulls are not satisfied. {special_instruction}": "en",
    "Find the absolute difference when comparing the {stat_type} of {col1} with the {stat_type} of {col2}, filtered by {conditions}. Present the result with exactly 2 decimal places. Skip null values in your calculations and treat unverifiable conditions as not satisfied. {special_instruction}": "en",
    "Work out the absolute difference between columns {col1} and {col2} for their {stat_type} values, where {conditions}. The answer should have exactly 2 decimal places. Ignore null values in your calculations. If a condition cannot be verified due to nulls, consider it not met. {special_instruction}": "en",
    "Evaluate the absolute difference between the {stat_type} of {col1} and the {stat_type} of {col2}, considering only rows where {conditions}. Express the result with 2 decimal places. Exclude null values from your analysis and assume conditions with nulls aren't satisfied. {special_instruction}": "en",
    "Analyze the {stat_type} values of columns {col1} and {col2}, then calculate their absolute difference for entries where {conditions}. Round to 2 decimal places, always showing both digits. Null values should be ignored, and conditions that can't be verified should be treated as false. {special_instruction}": "en",
    "Measure the absolute difference between the {stat_type} values found in column {col1} and column {col2}, restricted to records where {conditions}. Your answer must have exactly 2 decimal places. Skip nulls in your calculations and treat ambiguous conditions as not satisfied. {special_instruction}": "en",
    "Compare the {stat_type} of columns {col1} and {col2} by finding their absolute difference, but only consider rows where {conditions}. Round to 2 decimal places, always writing both decimal digits. Ignore nulls in your calculations and treat unverifiable conditions as false. {special_instruction}": "en",
    "Quantify the absolute difference between the {stat_type} of {col1} and {col2}, filtering for records where {conditions}. The result should show exactly 2 decimal places. Null values should be excluded from calculations, and conditions that can't be verified should be treated as not satisfied. {special_instruction}": "en",
    "计算列 {col1} 和列 {col2} 的 {stat_type} 之间的绝对差值，{conditions}。将结果四舍五入到小数点后2位。答案必须始终精确到2位小数（例如：42.00、0.50）。如果无法确定条件是否满足（例如，由于无法从表格其余部分推断的空值），则假设条件不满足。如果值为空，则在计算中忽略它。{special_instruction}": "zh",
    "求出列 {col1} 与列 {col2} 的 {stat_type} 之间的绝对差，其中 {conditions}。结果应四舍五入到2位小数（如42.00、0.50）。若因空值无法验证条件，视为不满足。计算时应排除空值。{special_instruction}": "zh",
    "确定满足条件 {conditions} 的行中，列 {col1} 和列 {col2} 的 {stat_type} 值之间的绝对差。将答案四舍五入到2位小数，始终显示两位小数位（如42.00）。计算中排除空值，并假设含空值的条件不满足。{special_instruction}": "zh",
    "找出列 {col1} 的 {stat_type} 与列 {col2} 的 {stat_type} 之间的绝对差值，筛选条件为 {conditions}。结果必须精确到2位小数。计算中跳过空值，并将无法验证的条件视为不满足。{special_instruction}": "zh",
    "计算出列 {col1} 和列 {col2} 的 {stat_type} 值之间的绝对差，其中 {conditions}。答案应保留2位小数。计算时忽略空值。如果由于空值无法验证条件，则视为不满足。{special_instruction}": "zh",
    "评估列 {col1} 和列 {col2} 的 {stat_type} 之间的绝对差，仅考虑满足 {conditions} 的行。以2位小数表示结果。分析时排除空值，并假设含空值的条件不满足。{special_instruction}": "zh",
    "分析列 {col1} 和列 {col2} 的 {stat_type} 值，然后计算满足 {conditions} 的记录的绝对差。四舍五入到2位小数，始终显示两位数字。计算中应忽略空值，无法验证的条件应视为不满足。{special_instruction}": "zh",
    "测量列 {col1} 和列 {col2} 中找到的 {stat_type} 值之间的绝对差，限制为满足 {conditions} 的记录。答案必须精确到2位小数。计算中跳过空值，将模糊条件视为不满足。{special_instruction}": "zh",
    "比较列 {col1} 和列 {col2} 的 {stat_type}，找出它们的绝对差，但仅考虑满足 {conditions} 的行。四舍五入到2位小数，始终显示两位小数。计算中忽略空值，将无法验证的条件视为不满足。{special_instruction}": "zh",
    "量化列 {col1} 和列 {col2} 的 {stat_type} 之间的绝对差，筛选满足 {conditions} 的记录。结果应精确显示2位小数。计算中应排除空值，无法验证的条件应视为不满足。{special_instruction}": "zh",
}
bug_description_templates = {
    'However, the code to convert the table into markdown format was buggy and mistakenly replaced some values with "ERROR". The values for those cells in column-order are respectively as follows: {affected_values}.': {
        "bug_type": "error",
        "lang": "en",
    },
    'Unfortunately, due to a bug in the conversion code, certain values were incorrectly replaced with "ERROR". The affected cells (in column-order) are: {affected_values}.': {
        "bug_type": "error",
        "lang": "en",
    },
    'The table conversion had a glitch that substituted "ERROR" for some values. The affected cell positions in column-order are: {affected_values}.': {
        "bug_type": "error",
        "lang": "en",
    },
    'A fault in the markdown formatter replaced specific values with "ERROR". The cells affected (listed in column-order) are: {affected_values}.': {
        "bug_type": "error",
        "lang": "en",
    },
    'The table contains errors where some values were mistakenly replaced with "ERROR" during conversion. The column-ordered list of affected cells is: {affected_values}.': {
        "bug_type": "error",
        "lang": "en",
    },
    '然而，将表格转换为markdown格式的代码有错误，错误地将某些值替换为"ERROR"。按列顺序，受影响的单元格值分别如下：{affected_values}。': {
        "bug_type": "error",
        "lang": "zh",
    },
    '不幸的是，由于转换代码中的错误，某些值被错误地替换为"ERROR"。受影响的单元格（按列顺序）是：{affected_values}。': {
        "bug_type": "error",
        "lang": "zh",
    },
    '表格转换过程中出现了故障，将一些值替换为"ERROR"。按列顺序，受影响的单元格位置是：{affected_values}。': {
        "bug_type": "error",
        "lang": "zh",
    },
    'markdown格式化程序中的错误导致特定值被替换为"ERROR"。受影响的单元格（按列顺序列出）是：{affected_values}。': {
        "bug_type": "error",
        "lang": "zh",
    },
    '表格中包含错误，在转换过程中某些值被错误地替换为"ERROR"。按列顺序排列的受影响单元格列表是：{affected_values}。': {
        "bug_type": "error",
        "lang": "zh",
    },
    'However, the code to convert the table into markdown format was buggy and mistakenly merged every two rows of values (not including the header) into one row by placing the values of the second row after the first row separated by " && ". In the case of an odd number of rows, the last row was merged with a row full of "null" values.': {
        "bug_type": "merge_rows",
        "lang": "en",
    },
    'Due to a bug, the conversion code incorrectly merged pairs of rows (excluding the header), placing the second row\'s values after the first row with " && " between them. For tables with odd row counts, the final row was combined with a row of "null" values.': {
        "bug_type": "merge_rows",
        "lang": "en",
    },
    'A defect in the markdown formatter caused it to combine every two rows (after the header), placing the second row\'s values after the first with " && " separators. When dealing with an odd number of rows, the last one was paired with synthesized "null" values.': {
        "bug_type": "merge_rows",
        "lang": "en",
    },
    'The table conversion contained an error that joined consecutive rows (except the header) by appending the second row\'s values to the first with " && " between them. With an odd row count, the last row was joined with "null" values.': {
        "bug_type": "merge_rows",
        "lang": "en",
    },
    'During format conversion, a bug caused pairs of data rows to be merged by concatenating the second row\'s values to the first with " && " separators. For tables with odd-numbered rows, the final row was paired with "null" values.': {
        "bug_type": "merge_rows",
        "lang": "en",
    },
    '然而，将表格转换为markdown格式的代码有错误，错误地将每两行值（不包括标题行）合并为一行，方法是将第二行的值放在第一行之后，用" && "分隔。如果行数为奇数，则最后一行与一行全是"null"值的行合并。': {
        "bug_type": "merge_rows",
        "lang": "zh",
    },
    '由于一个错误，转换代码错误地合并了每对行（不包括标题），将第二行的值放在第一行之后，中间用" && "分隔。对于奇数行的表格，最后一行与一行"null"值合并。': {
        "bug_type": "merge_rows",
        "lang": "zh",
    },
    'markdown格式化程序中的缺陷导致它合并了每两行（标题行之后），将第二行的值放在第一行之后，用" && "分隔符隔开。当处理奇数行时，最后一行与合成的"null"值配对。': {
        "bug_type": "merge_rows",
        "lang": "zh",
    },
    '表格转换包含一个错误，它将连续的行（标题行除外）连接起来，将第二行的值附加到第一行，中间用" && "分隔。如果行数为奇数，最后一行与"null"值连接。': {
        "bug_type": "merge_rows",
        "lang": "zh",
    },
    '在格式转换过程中，一个错误导致成对的数据行合并，方法是将第二行的值连接到第一行，用" && "分隔符分隔。对于奇数行的表格，最后一行与"null"值配对。': {
        "bug_type": "merge_rows",
        "lang": "zh",
    },
}
bug_description_templates.update(
    {
        "However, the code to convert the table into markdown format was buggy and, except for the header row that was formatted correctly, the values in the i-th column were rotated down i times (e.g., for the first column of values, the value at row j was moved to row j+1 and the value in the last row was moved to the first row).": {
            "bug_type": "rotate_columns",
            "lang": "en",
        },
        "Unfortunately, a bug in the table formatter caused each column's values to be shifted downward. Specifically, values in the i-th column (0-indexed) were rotated down i positions, while the header row remained intact.": {
            "bug_type": "rotate_columns",
            "lang": "en",
        },
        "A glitch occurred during conversion where each column i (0-indexed) had its values circularly shifted downward by i positions. The header row was unaffected by this rotation.": {
            "bug_type": "rotate_columns",
            "lang": "en",
        },
        "The table contains a formatting error where each column's values were circularly rotated. For column i (0-indexed), values shifted down i spaces (wrapping from bottom to top). The header row was correctly preserved.": {
            "bug_type": "rotate_columns",
            "lang": "en",
        },
        "During conversion, a bug caused column values to rotate. In column i (0-indexed), each value moved down i positions, with values at the bottom wrapping to the top. The header row remained in the correct position.": {
            "bug_type": "rotate_columns",
            "lang": "en",
        },
        "然而，将表格转换为markdown格式的代码有错误，除了正确格式化的标题行外，第i列中的值向下旋转了i次（例如，对于第一列值，第j行的值被移动到第j+1行，最后一行的值被移动到第一行）。": {
            "bug_type": "rotate_columns",
            "lang": "zh",
        },
        "不幸的是，表格格式化程序中的错误导致每列的值向下移动。具体来说，第i列（0索引）的值向下旋转了i个位置，而标题行保持不变。": {
            "bug_type": "rotate_columns",
            "lang": "zh",
        },
        "在转换过程中出现了故障，每列i（0索引）的值都被循环向下移动了i个位置。标题行不受此旋转的影响。": {
            "bug_type": "rotate_columns",
            "lang": "zh",
        },
        "表格包含一个格式错误，每列的值都被循环旋转。对于第i列（0索引），值向下移动了i个空格（从底部包装到顶部）。标题行被正确保留。": {
            "bug_type": "rotate_columns",
            "lang": "zh",
        },
        "在转换过程中，一个错误导致列值旋转。在第i列（0索引）中，每个值向下移动i个位置，底部的值包装到顶部。标题行保持在正确位置。": {
            "bug_type": "rotate_columns",
            "lang": "zh",
        },
        "However, the code to convert the table into markdown format was buggy and, except for the header row that was formatted correctly, the values in the i-th row were rotated right i times (e.g., for the first row of values, the value at column j was moved to column j+1 and the value in the last column was moved to the first column).": {
            "bug_type": "rotate_rows",
            "lang": "en",
        },
        "A bug in the table conversion caused each row's values to shift horizontally. The i-th data row (0-indexed, after the header) had its values rotated right by i positions, with values wrapping from right to left.": {
            "bug_type": "rotate_rows",
            "lang": "en",
        },
        "During formatting, a glitch caused values in each row to be circularly shifted. Specifically, in the i-th row after the header (0-indexed), values were rotated right i times. The header row remained correctly formatted.": {
            "bug_type": "rotate_rows",
            "lang": "en",
        },
        "The table contains an error where each row's values were horizontally rotated. For row i after the header (0-indexed), values shifted right i positions with end-of-row values wrapping to the beginning. The header row was unaffected.": {
            "bug_type": "rotate_rows",
            "lang": "en",
        },
        "A conversion defect caused each data row i (0-indexed) to have its values circularly shifted right by i places. Values at the rightmost column wrapped back to the leftmost position. The header row remained properly formatted.": {
            "bug_type": "rotate_rows",
            "lang": "en",
        },
        "然而，将表格转换为markdown格式的代码有错误，除了正确格式化的标题行外，第i行中的值向右旋转了i次（例如，对于第一行值，第j列的值被移动到第j+1列，最后一列的值被移动到第一列）。": {
            "bug_type": "rotate_rows",
            "lang": "zh",
        },
        "表格转换中的一个错误导致每行的值水平移动。第i个数据行（0索引，标题行之后）的值向右旋转了i个位置，值从右到左包装。": {
            "bug_type": "rotate_rows",
            "lang": "zh",
        },
        "在格式化过程中，一个故障导致每行中的值被循环移动。具体来说，在标题行之后的第i行（0索引），值向右旋转了i次。标题行保持正确格式。": {
            "bug_type": "rotate_rows",
            "lang": "zh",
        },
        "表格包含一个错误，每行的值都被水平旋转。对于标题行之后的第i行（0索引），值向右移动i个位置，行尾的值包装到开头。标题行不受影响。": {
            "bug_type": "rotate_rows",
            "lang": "zh",
        },
        "一个转换缺陷导致每个数据行i（0索引）的值向右循环移动i个位置。最右列的值回到最左边的位置。标题行保持正确格式。": {
            "bug_type": "rotate_rows",
            "lang": "zh",
        },
        "However, the code to save the table was buggy and after saving each row, it appended some random values to the end of the row. Specifically, at the end of the i-th row (0-indexed), i random values were appended to the end of the row.": {
            "bug_type": "add_end_row",
            "lang": "en",
        },
        "A bug in the table saving process caused extra values to be appended to each row. For the i-th row (0-indexed), i random values were added to the end of the row.": {
            "bug_type": "add_end_row",
            "lang": "en",
        },
        "The table saving code contained an error that added surplus values at the end of rows. Specifically, for row i (0-indexed), i random values were appended after the legitimate data.": {
            "bug_type": "add_end_row",
            "lang": "en",
        },
        "During the save operation, a fault caused the system to add random values to the end of each row. For the i-th row (where i is 0-indexed), exactly i random values were appended.": {
            "bug_type": "add_end_row",
            "lang": "en",
        },
        "A glitch in the table storage process appended spurious values to row endings. Row i (0-indexed) received i random additional values at its end.": {
            "bug_type": "add_end_row",
            "lang": "en",
        },
        "然而，保存表格的代码有错误，在保存每行后，它在行尾附加了一些随机值。具体来说，在第i行（0索引）的末尾，添加了i个随机值。": {
            "bug_type": "add_end_row",
            "lang": "zh",
        },
        "表格保存过程中的一个错误导致每行末尾附加了额外的值。对于第i行（0索引），在行尾添加了i个随机值。": {
            "bug_type": "add_end_row",
            "lang": "zh",
        },
        "表格保存代码包含一个错误，在行尾添加了多余的值。具体来说，对于第i行（0索引），在合法数据之后附加了i个随机值。": {
            "bug_type": "add_end_row",
            "lang": "zh",
        },
        "在保存操作期间，一个故障导致系统在每行末尾添加随机值。对于第i行（其中i是0索引），在末尾添加了恰好i个随机值。": {
            "bug_type": "add_end_row",
            "lang": "zh",
        },
        "表格存储过程中的一个故障在行尾附加了虚假值。第i行（0索引）在其末尾收到了i个随机附加值。": {
            "bug_type": "add_end_row",
            "lang": "zh",
        },
        "However, the code to save the table was buggy and after saving each column, it appended some random values to the end of the column. Specifically, at the end of the j-th column (0-indexed), j random values were appended to the end of the column.": {
            "bug_type": "add_end_column",
            "lang": "en",
        },
        "A bug in the table saving routine caused extra values to be added at the bottom of each column. For column j (0-indexed), j random values were appended below the existing data.": {
            "bug_type": "add_end_column",
            "lang": "en",
        },
        "During the save process, a glitch caused the addition of spurious data at the end of columns. Column j (0-indexed) had j random values appended to its bottom.": {
            "bug_type": "add_end_column",
            "lang": "en",
        },
        "The table storage code mistakenly appended random values to the end of each column. Specifically, the j-th column (0-indexed) received j random values appended after the legitimate data.": {
            "bug_type": "add_end_column",
            "lang": "en",
        },
        "A defect in the saving mechanism resulted in random data being inserted at column endings. For each column j (0-indexed), j extra random values were appended to the bottom.": {
            "bug_type": "add_end_column",
            "lang": "en",
        },
        "然而，保存表格的代码有错误，在保存每列后，它在列尾附加了一些随机值。具体来说，在第j列（0索引）的末尾，添加了j个随机值。": {
            "bug_type": "add_end_column",
            "lang": "zh",
        },
        "表格保存程序中的一个错误导致在每列底部添加了额外的值。对于第j列（0索引），在现有数据下面附加了j个随机值。": {
            "bug_type": "add_end_column",
            "lang": "zh",
        },
        "在保存过程中，一个故障导致在列末尾添加了虚假数据。第j列（0索引）在其底部附加了j个随机值。": {
            "bug_type": "add_end_column",
            "lang": "zh",
        },
        "表格存储代码错误地在每列末尾附加了随机值。具体来说，第j列（0索引）在合法数据之后收到了j个随机值。": {
            "bug_type": "add_end_column",
            "lang": "zh",
        },
        "保存机制中的一个缺陷导致在列末尾插入随机数据。对于每列j（0索引），在底部附加了j个额外的随机值。": {
            "bug_type": "add_end_column",
            "lang": "zh",
        },
        "However, the code to save the table was buggy and failed to save the null values. There were null values at the following locations (0-indexed -- table header included): {affected_values}.": {
            "bug_type": "null",
            "lang": "en",
        },
        "A bug in the table saving code failed to properly handle null values. The following positions (0-indexed, including the header row) contained null data: {affected_values}.": {
            "bug_type": "null",
            "lang": "en",
        },
        "The table storage process had an error in null value handling. Null values were present at these locations (0-indexed, header row included): {affected_values}.": {
            "bug_type": "null",
            "lang": "en",
        },
        "During saving, the system failed to properly store null values. The cells at these coordinates (0-indexed, with header row) contained nulls: {affected_values}.": {
            "bug_type": "null",
            "lang": "en",
        },
        "A defect in the saving routine caused null values to be mishandled. Null data was present at the following positions (0-indexed, including header): {affected_values}.": {
            "bug_type": "null",
            "lang": "en",
        },
        "然而，保存表格的代码有错误，未能保存空值。以下位置（0索引 -- 包括表头）有空值：{affected_values}。": {
            "bug_type": "null",
            "lang": "zh",
        },
        "表格保存代码中的一个错误导致无法正确处理空值。以下位置（0索引，包括标题行）包含空数据：{affected_values}。": {
            "bug_type": "null",
            "lang": "zh",
        },
        "表格存储过程在空值处理方面存在错误。以下位置（0索引，包括标题行）存在空值：{affected_values}。": {
            "bug_type": "null",
            "lang": "zh",
        },
        "在保存过程中，系统未能正确存储空值。这些坐标（0索引，包括标题行）的单元格包含空值：{affected_values}。": {
            "bug_type": "null",
            "lang": "zh",
        },
        "保存例程中的一个缺陷导致空值被错误处理。以下位置（0索引，包括标题）存在空数据：{affected_values}。": {
            "bug_type": "null",
            "lang": "zh",
        },
    }
)


def get_bug_description(bug_type, affected_values=None, error_positions=None, language="en", order="column"):
    matching_templates = {
        k: v for k, v in bug_description_templates.items() if v["bug_type"] == bug_type and v["lang"] == language
    }
    if not matching_templates:
        matching_templates = {
            k: v for k, v in bug_description_templates.items() if v["bug_type"] == bug_type and v["lang"] == "en"
        }
    if not matching_templates:
        if language == "zh":
            return "表格包含未指定的错误。"
        else:
            return "The table contains an unspecified bug."
    template = random.choice(list(matching_templates.keys()))
    if bug_type == "error" and affected_values:
        if error_positions and len(error_positions) == len(affected_values):
            error_info = list(zip(error_positions, affected_values))
            if order == "column":
                error_info.sort(key=lambda x: (x[0][1], x[0][0]))
            else:
                error_info.sort(key=lambda x: (x[0][0], x[0][1]))
            sorted_values = [info[1] for info in error_info]
            return template.format(affected_values=sorted_values)
        elif isinstance(affected_values, list) and affected_values and isinstance(affected_values[0], tuple):
            values_only = [value for _, value in affected_values]
            return template.format(affected_values=values_only)
        else:
            return template.format(affected_values=affected_values)
    elif bug_type == "null" and affected_values:
        return template.format(affected_values=affected_values)
    elif bug_type in ["merge_rows", "rotate_columns", "rotate_rows", "add_end_row", "add_end_column"]:
        return template
    if language == "zh":
        return "表格包含未指定的错误。"
    else:
        return "The table contains an unspecified bug."


def format_query_template(query_data, language="en"):
    lang_templates = {k: v for k, v in query_templates.items() if v == language}
    if not lang_templates:
        lang_templates = {k: v for k, v in query_templates.items() if v == "en"}
    template = random.choice(list(lang_templates.keys()))
    cols_to_compare = query_data.get("cols_to_compare", [])
    if len(cols_to_compare) < 2:
        raise ValueError(f"cols_to_compare must have at least 2 elements, got {len(cols_to_compare)}")
    col1, col2 = cols_to_compare[0], cols_to_compare[1]
    stat_type = query_data["stat_type"]
    conditions = query_data["conditions"]
    conditions_text = ", ".join(
        [f"the {condition} was {cond_info['op']} {cond_info['value']}" for condition, cond_info in conditions.items()]
    )
    special_instruction = ""
    if stat_type == "stdev" or stat_type == "variance":
        special_instruction = f'If any of the {stat_type}s cannot be computed due to there being no values or only one value, the final answer must be "0.00".'
    else:
        special_instruction = f'If any of the {stat_type}s cannot be computed due to there being no values, the final answer must be "0.00".'
    return template.format(
        stat_type=stat_type, col1=col1, col2=col2, conditions=conditions_text, special_instruction=special_instruction
    )


question_framework_templates = {
    "I have a table with {num_rows} rows (including the header) and {num_columns} columns. The table was {format_verb} {table_format} format as follows:\n{table_section}.\n{bug_description}\n{query}": "en",
    "Below is a table containing {num_rows} rows (header included) and {num_columns} columns, {format_verb} {table_format} format:\n{table_section}.\n{bug_description}\n{query}": "en",
    "The following table has {num_rows} rows (with header) and {num_columns} columns. It was {format_verb} {table_format} representation:\n{table_section}.\n{bug_description}\n{query}": "en",
    "Here's a dataset with {num_rows} rows (including header row) and {num_columns} columns, which was {format_verb} {table_format} format:\n{table_section}.\n{bug_description}\n{query}": "en",
    "Examine this table containing {num_rows} rows (header included) and {num_columns} columns. The data was {format_verb} {table_format} notation:\n{table_section}.\n{bug_description}\n{query}": "en",
    "This is a table with {num_rows} rows (counting the header) and {num_columns} columns that was {format_verb} {table_format} format:\n{table_section}.\n{bug_description}\n{query}": "en",
    "The data table below has {num_rows} rows (including the header row) and {num_columns} columns, {format_verb} {table_format} representation:\n{table_section}.\n{bug_description}\n{query}": "en",
    "I'm working with a table of {num_rows} rows (header included) and {num_columns} columns, {format_verb} {table_format} structure:\n{table_section}.\n{bug_description}\n{query}": "en",
    "Let's analyze a table with {num_rows} rows (including the header) and {num_columns} columns. The table was {format_verb} {table_format} notation:\n{table_section}.\n{bug_description}\n{query}": "en",
    "We have a data table containing {num_rows} rows (with header) and {num_columns} columns, which was {format_verb} {table_format} format:\n{table_section}.\n{bug_description}\n{query}": "en",
    "我有一个表格，包含 {num_rows} 行（包括表头）和 {num_columns} 列。该表格以 {table_format} 格式 {format_verb} 如下：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "以下是一个包含 {num_rows} 行（包括表头）和 {num_columns} 列的表格，以 {table_format} 格式 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "下面的表格有 {num_rows} 行（带表头）和 {num_columns} 列。它以 {table_format} 表示方式 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "这是一个数据集，有 {num_rows} 行（包括表头行）和 {num_columns} 列，它以 {table_format} 格式 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "请查看这个包含 {num_rows} 行（包括表头）和 {num_columns} 列的表格。数据以 {table_format} 表示法 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "这是一个有 {num_rows} 行（计算表头）和 {num_columns} 列的表格，以 {table_format} 格式 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "下面的数据表有 {num_rows} 行（包括表头行）和 {num_columns} 列，以 {table_format} 表示方式 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "我正在处理一个表格，有 {num_rows} 行（包括表头）和 {num_columns} 列，以 {table_format} 结构 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "让我们分析一个有 {num_rows} 行（包括表头）和 {num_columns} 列的表格。该表格以 {table_format} 表示法 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
    "我们有一个数据表格，包含 {num_rows} 行（带表头）和 {num_columns} 列，它以 {table_format} 格式 {format_verb}：\n{table_section}。\n{bug_description}\n{query}": "zh",
}
format_verb_translations = {
    "saved in a": {"en": "saved in a", "zh": "保存为"},
    "converted to": {"en": "converted to", "zh": "转换为"},
}
table_format_translations = {
    "row-major": {"en": "row-major", "zh": "行主序"},
    "markdown": {"en": "markdown", "zh": "markdown"},
}


def format_question_template(question_data, language="en"):
    lang_templates = {k: v for k, v in question_framework_templates.items() if v == language}
    if not lang_templates:
        lang_templates = {k: v for k, v in question_framework_templates.items() if v == "en"}
    template = random.choice(list(lang_templates.keys()))
    num_rows = question_data["num_rows"]
    num_columns = question_data["num_columns"]
    bug_type = question_data["bug_type"]
    table_format = question_data["table_format"]
    table_section = question_data["table_section"]
    bug_description = question_data["bug_description"]
    query = question_data["query"]
    if bug_type in ["add_end_row", "add_end_column", "null"]:
        format_verb_key = "saved in a"
    else:
        format_verb_key = "converted to"
    format_verb = format_verb_translations[format_verb_key][language]
    translated_table_format = table_format_translations[table_format][language]
    return template.format(
        num_rows=num_rows,
        num_columns=num_columns,
        format_verb=format_verb,
        table_format=translated_table_format,
        table_section=table_section,
        bug_description=bug_description,
        query=query,
    )
