# -*- coding: utf-8 -*-
"""Write and validate the frozen Tier A query set.

Corpus-only: source ids are resolved by exact title match. No retriever runs.

This script is the *construction* record for stratum A. It is kept in the repo
so the set is reproducible: re-running it must produce a byte-identical file,
and the hash in queries/MANIFEST.md is that byte content's digest. If you edit a
query here you have edited the evaluation, which is a different act from fixing a
bug -- do it deliberately, and re-record the hash.

Stratum B is NOT built here. It needs a retrieval pool, and a pool is only
legitimate once the judgments were made blind of the systems being compared.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from semantic_search.data.loader import DocumentStore  # noqa: E402

# --------------------------------------------------------------------------- #
# The set. (query_id, source title, query, derivation, notes)
#
# Every "lead" query was written by reading the article's own opening text and
# asking a question of it. That is a real judgment -- it is just a judgment made
# by the person who also reads the retrieval output later, which is why the
# limitation is stated in MANIFEST.md rather than left implicit.
# --------------------------------------------------------------------------- #
ROWS: list[tuple[str, str, str, str, str]] = [
    # --- A-restated: a natural question about the article ---------------
    ("A01", "Piano Trios, Op. 1 (Beethoven)",
     "what were the three piano trios Beethoven published in 1795",
     "lead", "Op. 1 designation; dedicated to Prince Lichnowsky"),
    ("A02", "Book of Alma",
     "which book of the Book of Mormon is the longest and named after a Nephite prophet",
     "lead", "Alma the Younger; 63 chapters"),
    ("A03", "AIM120 AMRAAM",
     "what guidance system does the American beyond-visual-range air-to-air missile use",
     "lead", "active transmit-receive radar; fire-and-forget"),
    ("A04", "Comparison of American and British English",
     "how did the English language reach the Americas",
     "lead", "British colonisation, late 16th-17th c."),
    ("A05", "Demographics of Angola",
     "what are the main ethnic groups of Angola",
     "lead", "Ovimbundu 37%, Ambundu; Bantu languages"),
    ("A06", "Ælle of Sussex",
     "who was the first king of the South Saxons",
     "lead", "reigned 477 to c.514 in what is now Sussex"),
    ("A07", "Andersonville, Georgia",
     "where was the Civil War prisoner of war camp in Georgia",
     "lead", "Andersonville National Historic Site; Sumter County"),
    ("A08", "The Amazing SpiderMan",
     "when did the Spider-Man title comic series begin publication",
     "lead", "March 1963 cover date"),
    ("A09", "Severus Alexander",
     "who was the last emperor from the Severan dynasty",
     "lead", "reigned 222-235; succeeded Elagabalus"),
    ("A10", "Saint Titus",
     "who was the early Christian missionary consecrated Bishop of Crete",
     "lead", "companion of Paul; Gentile convert"),
    ("A11", "ANSI C",
     "which bodies publish the standards for the C programming language",
     "lead", "ANSI and ISO/IEC JTC 1/SC 22/WG 14"),
    ("A12", "Demographics of American Samoa",
     "what population statistics are recorded for the US territory in the South Pacific",
     "lead", "unincorporated territory; censuses 1900-1950"),
    ("A13", "Bill Clinton",
     "which US president served from 1993 to 2001",
     "lead", "42nd president; governor of Arkansas"),
    ("A14", "Agar",
     "what is the jellylike substance made from red algae cell walls",
     "lead", "agar-agar; agarose and agaropectin"),
    ("A15", "Black Forest",
     "which mountain range in southwest Germany is the source of the Danube",
     "lead", "Baden-Wurttemberg; Feldberg peak"),
    ("A16", "Belgium",
     "which country in Northwestern Europe borders the Netherlands to the north",
     "lead", "11.5 million; 22nd most densely populated"),
    ("A17", "Brass",
     "what two metals make up the substitutional alloy used since prehistoric times",
     "lead", "copper and zinc; similar to bronze"),
    ("A18", "Army of Darkness",
     "which Sam Raimi film follows Ash Williams trapped in the Middle Ages",
     "lead", "1992; sequel to Evil Dead II"),
    ("A19", "Big Dipper",
     "what are the seven brightest stars of Ursa Major called in American English",
     "lead", "the Plough in British English"),
    ("A20", "Athlon",
     "which AMD processor brand was the first desktop chip to reach one gigahertz",
     "lead", "Athlon Classic, 23 June 1999"),
    ("A21", "Alexandria",
     "which Egyptian city on the Mediterranean coast was founded by Alexander the Great",
     "lead", "second largest city in Egypt; replaced Memphis"),
    ("A22", "Ānanda",
     "which disciple of the Buddha was known for having the best memory",
     "lead", "primary attendant; First Buddhist Council"),
    ("A23", "Boron nitride",
     "what is the chemical formula of the refractory compound of boron and nitrogen",
     "lead", "BN; hexagonal form used as lubricant"),
    ("A24", "IBM AIX",
     "which IBM series of proprietary Unix operating systems was first released for the RT PC in 1986",
     "lead", "AIX; RS/6000, Power, System i"),
    ("A25", "Beer",
     "what are the main cereal grains used to brew the world's most widely consumed alcoholic drink",
     "lead", "malted barley, wheat, maize, rice, oats"),
    # --- A-verbatim: the query IS the article title ---------------------
    ("V01", "Piano Trios, Op. 1 (Beethoven)", "Piano Trios, Op. 1 (Beethoven)", "title", ""),
    ("V02", "Book of Alma", "Book of Alma", "title", ""),
    ("V03", "AIM120 AMRAAM", "AIM120 AMRAAM", "title", ""),
    ("V04", "Comparison of American and British English", "Comparison of American and British English", "title", ""),
    ("V05", "Demographics of Angola", "Demographics of Angola", "title", ""),
    ("V06", "Ælle of Sussex", "Ælle of Sussex", "title", ""),
    ("V07", "Andersonville, Georgia", "Andersonville, Georgia", "title", ""),
    ("V08", "The Amazing SpiderMan", "The Amazing SpiderMan", "title", ""),
    ("V09", "Severus Alexander", "Severus Alexander", "title", ""),
    ("V10", "Saint Titus", "Saint Titus", "title", ""),
    ("V11", "ANSI C", "ANSI C", "title", ""),
    ("V12", "Demographics of American Samoa", "Demographics of American Samoa", "title", ""),
    ("V13", "Bill Clinton", "Bill Clinton", "title", ""),
    ("V14", "Agar", "Agar", "title", ""),
    ("V15", "Black Forest", "Black Forest", "title", ""),
    # --- A-typo: frozen misspellings -------------------------------------
    # Lowercase content words only. The corrector's proper-noun gate refuses
    # capitalised terms, so a typo in "Alexandria" could never be repaired and
    # the row would measure nothing. T02 is deliberately unmisspelled: not
    # every word is typosable, and asserting that uniformly would be false.
    ("T01", "Piano Trios, Op. 1 (Beethoven)",
     "what were the three piano trios Beethoven publshed in 1795",
     "typo:A01", "transposition, unobservable on a keyboard"),
    ("T02", "Book of Alma",
     "which book of the Book of Mormon is the longest and named after a Nephite prophet",
     "typo:none", "A02 unchanged: unobservable typo, superset of A02"),
    ("T03", "AIM120 AMRAAM",
     "what gudiance system does the American beyond-visual-range air-to-air missle use",
     "typo:A03", "dropped letter x2"),
    ("T04", "Comparison of American and British English",
     "how did the Engish language reach the Americas",
     "typo:A04", "dropped letter"),
    ("T05", "Demographics of Angola",
     "what are the main ethnic grpus of Angola",
     "typo:A05", "dropped letter"),
    ("T06", "Ælle of Sussex",
     "who was the first kng of the South Saxons",
     "typo:A06", "dropped letter"),
    ("T07", "Andersonville, Georgia",
     "where was the Civil War prsoner of war camp in Georgia",
     "typo:A07", "dropped letter"),
    ("T08", "The Amazing SpiderMan",
     "when did the Spider-Man ttle comic series begin publication",
     "typo:A08", "dropped letter"),
    ("T09", "Severus Alexander",
     "who was the last emporer from the Severan dynasty",
     "typo:A09", "dropped letter"),
    ("T10", "Bill Clinton",
     "which US president srved from 1993 to 2001",
     "typo:A13", "dropped letter"),
]

TIER = {"A": "A_restated", "V": "A_verbatim", "T": "A_typo"}


def main() -> int:
    store = DocumentStore.load_jsonl(Path("data/processed/corpus.jsonl"))
    by_title = {d.title: d.id for d in store}
    if len(by_title) != len(store):
        print(f"ERROR: titles are not unique ({len(by_title)}/{len(store)})")
        return 1

    rows = []
    for query_id, title, query, derivation, notes in ROWS:
        if title not in by_title:
            print(f"ERROR: {query_id}: no corpus document titled {title!r}")
            return 1
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "stratum": TIER[query_id[0]],
                "source_doc_id": by_title[title],
                "derivation": derivation,
                "notes": notes,
            }
        )

    # --- validate the typo stratum ---------------------------------------
    by_id = {r["query_id"]: r for r in rows}
    tokenize = lambda s: re.findall(r"[a-z0-9]+", s.lower())  # noqa: E731
    distinct = 0
    for row in rows:
        if row["stratum"] != "A_typo":
            continue
        if row["derivation"] == "typo:none":
            if row["query"] != by_id["A02"]["query"]:
                print(f"ERROR: {row['query_id']} claims typo:none but differs from A02")
                return 1
            print(f"  {row['query_id']}: verbatim copy of A02 (unobservable typo, intentional)")
            continue
        parent = by_id.get(row["derivation"].split(":", 1)[1])
        if parent is None:
            print(f"ERROR: {row['query_id']} names a missing parent {row['derivation']}")
            return 1
        expected, actual = tokenize(parent["query"]), tokenize(row["query"])
        diffs = [(a, b) for a, b in zip(expected, actual, strict=False) if a != b]
        if not diffs:
            print(f"ERROR: {row['query_id']} is identical to its parent")
            return 1
        for _, misspelt in diffs:
            if not misspelt.islower():
                print(f"ERROR: {row['query_id']}: {misspelt!r} is not lowercase -- untestable")
                return 1
        distinct += 1
        print(f"  {row['query_id']}: {', '.join(f'{a}>{b}' for a, b in diffs)}")
    print(f"typo rows with a genuinely distinct misspelling: {distinct}/10")

    reuse = Counter(r["source_doc_id"] for r in rows)
    print(f"source docs used by more than one stratum: {sum(1 for v in reuse.values() if v > 1)}")

    out = Path("data/evaluation/queries/queries.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} queries -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
