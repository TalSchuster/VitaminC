"""Simple script to post-process dataset sentences."""

import argparse
import json
import regex

from collections import defaultdict, Counter


def strip_headings(text):
    # Filter starting headings.                                                                                                                                                                                                                              
    match = regex.match(r"^(={2,})[^=]{1,50}(\1)(.*)", text)
    while match:
        text = match.groups()[2]
        match = regex.match(r"^(={2,})[^=]{1,50}(\1)(.*)", text)

    # Filter ending headings.                                                                                                                                                                                                                                
    match = regex.match(r"(.*?)(={2,})[^=]{1,50}(\2)$", text)
    if match:
        text = match.groups()[0]

    # Filter middle headings.
    match = regex.match(r"(^.*?)([=]{2,})[^=]{2,50}(\2)(.*$)", text)
    if match:
        text = match.groups()[0] + " " + match.groups()[-1]

    return text


def strip_urls(text):
    url = r"https?\s?:\s?\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    text = regex.sub(url, " ", text)
    return text


def fix_whitespace(text):
    text = text.strip()
    text = " ".join(text.split(" "))
    return text


def fix_brackets(text):
    if text.startswith("} } "):
        text = text[len("} } "):]
    return text


def filter_nonalpha(text):
    if not regex.search("[\p{L}\p{N}\p{M}]", text):
        return ""
    return text


def main(args):
    with open(args.input_file, "r") as f:
        dataset = [json.loads(line) for line in f]

    id2evs = defaultdict(lambda: set())
    id_counter = Counter()
    filtered = []
    adjusted = 0
    skipped = 0
    total = len(dataset)
    for example in dataset:
        evidence = example["evidence"]
        original = evidence
        evidence = strip_headings(evidence)
        evidence = strip_urls(evidence)
        evidence = fix_brackets(evidence)
        evidence = fix_whitespace(evidence)
        evidence = filter_nonalpha(evidence)
        if not evidence:
            skipped += 1
            continue
        if evidence != original:
            if adjusted < 20:
                print (f"Before: {original} \nAfter: {evidence}")
                print ('-' * 30)
            adjusted += 1
        example["evidence"] = evidence
        filtered.append(example)

        id2evs[example["case_id"]].add(evidence)
        id_counter[example["case_id"]] += 1

    exclude_ids = set()
    for cid, evs in id2evs.items():
        if len(evs) < 2 and id_counter[cid] > 1:
            exclude_ids.add(cid)

    exclude = 0
    with open(args.output_file, "w") as f:
        for example in filtered:
            if example["case_id"] in exclude_ids:
                exclude += 1
            else:
                f.write(json.dumps(example) + "\n")
            
    print(f"Adjusted: {adjusted}\nSkipped no evidence: {skipped}\nSkipped same evidence: {exclude}\nTotal: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()
    main(args)
