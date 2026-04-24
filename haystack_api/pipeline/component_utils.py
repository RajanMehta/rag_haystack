import re
from typing import Dict, List, Optional

import pydash
from pydash.arrays import find_last_index
from pydash.strings import (
    clean,
    ensure_ends_with,
    ensure_starts_with,
    lines,
    replace_end,
    trim,
)


def line_contains_substring(line, strlist):
    """
    line: str
    strlist: List[str]
    returns True if any string from `strlist` is present in `line`
    """
    return any(substring in line for substring in strlist)


def custom_make_translation(text, translation):
    regex = re.compile("|".join(map(re.escape, translation)))
    return regex.sub(lambda match: translation[match.group(0)], text)


def handle_substring_replacements(
    content: str, replace_substrings_concurrent: Dict[str, str], replace_substrings_ordered: Dict[str, str]
):
    if replace_substrings_concurrent:
        content = custom_make_translation(content, replace_substrings_concurrent)
    if replace_substrings_ordered:
        for source, target in replace_substrings_ordered.items():
            content = content.replace(source, target)
    return content


def get_cleaned_lines(content):
    """
    Splits lines in text into an array.
    Trims and replaces multiple spaces with a single space.
    Removes all empty lines.
    """
    lines_list = lines(content)
    lines_list = pydash.chain(lines_list).map(lambda x: trim(clean(x))).compact().value()
    return lines_list


def handle_bullets(
    full_text,
    bullets_primary: Optional[List[str]],
    bullets_nested: Optional[List[str]],
    bullet_titles: Optional[List[str]],
):
    """
    handles bullet-points such that they stay together during splitting
    process in the PreProcessor step of pipeline.
    """
    bullet_titles_indices = []

    for title in bullet_titles:
        last_index = find_last_index(full_text, lambda x: x == title)
        if last_index > -1:
            bullet_titles_indices.append(last_index)

    if bullet_titles_indices:
        for i, line in enumerate(full_text):
            # add title to primary-bullets as titles are on separate lines and primary-bullets
            # might miss the context otherwise. We don't add titles to nested-bulltes.
            last_index = find_last_index(bullet_titles_indices, lambda x: x < i)
            if line_contains_substring(line, bullets_primary):
                closest_title = full_text[bullet_titles_indices[last_index]]
                full_text[i] = ensure_starts_with(line, f"{closest_title} ")

    for i, line in enumerate(full_text):
        # if line contains a nested bullet-point, do not end the sentence.
        # but if that's the last nested bullet-point before a primary bullet-point, end it.
        try:
            if line_contains_substring(line, bullets_nested):
                full_text[i] = replace_end(line, ".", "")
            if line_contains_substring(full_text[i + 1], bullets_primary):
                full_text[i] = ensure_ends_with(line, ".")
        except IndexError:
            pass

        # if line is in between two nested bullet-points, do not end the sentence.
        try:
            if line_contains_substring(full_text[i + 1], bullets_nested) and not line_contains_substring(
                line, bullets_nested
            ):
                full_text[i] = replace_end(line, ".", "")
            if line_contains_substring(full_text[i + 1], bullets_primary):
                full_text[i] = ensure_ends_with(line, ".")
        except IndexError:
            pass

    return full_text
