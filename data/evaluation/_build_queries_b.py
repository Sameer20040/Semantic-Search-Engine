"""Construction record for stratum B (hand-written topical queries).

Run BEFORE any retrieval. The point of writing the queries down and hashing
them first is that the judgments cannot then be fitted to the systems: the pool
is built afterwards from their output, and the qrels file is written from the
pool. A script that wrote queries and labels in one pass, after seeing a
ranking, would produce a qrels file that measures the retriever.

Protocol
--------
Ten articles drawn by ``random.Random(20251001).sample`` from the 275 corpus
articles not used as stratum-A sources, then three questions per article. The
sampling is mechanical for the same reason stratum A's is: choosing articles by
hand means choosing the ones I find interesting to write about, which is a
choice correlated with the ones the retrievers find easy.

Each question is written so the article is a genuinely good answer -- and, so
far as the lead shows, possibly the only one in the corpus. That is a
limitation, recorded in the qrels manifest, not a hidden assumption: this corpus
is 300 loosely-related Wikipedia articles, so "topical" mostly means "about
this article". Stratum B's value here is that the query does not name the
article, so title matching cannot trivially solve it.

Deliberately avoided in every question: the article's title words, its
disambiguator, and the phrasing of its first sentence. Where a country name was
the only natural way to ask, the question was rephrased around the
institution instead ("which house can override the other when a money bill is
rejected" rather than "what can Barbados do with a money bill").

The answer to each question is recorded in ``ANSWERS`` below but is *not* used
to build the judgments -- it is written so the judging pass can check that the
pooled document really does contain the answer, which is the only way to keep
"relevant" from silently meaning "topically related".
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:  # pragma: no cover - script bootstrap
    sys.path.insert(0, str(SRC))

from semantic_search.data.loader import DocumentStore  # noqa: E402

CORPUS = Path("data/processed/corpus.jsonl")
QUERIES_OUT = Path("data/evaluation/queries/queries.jsonl")
SEED = 20251001
N_ARTICLES = 10
QUERIES_PER_ARTICLE = 3

TIER = "B_topical"

# (source title, [(query, expected answer as found in the article), ...])
SPEC: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Government of Barbados",
        [
            ("how many seats does the upper chamber have and who appoints them",
             "21 Senate members; 12 on the Prime Minister's advice, 2 on the "
             "Opposition Leader's advice, 7 at the President's sole discretion"),
            ("which house can override the other when a money bill is rejected",
             "the House of Assembly may override Senate's rejection of money "
             "bills and other bills except bills amending the Constitution"),
            ("who holds the two different head offices in this system",
             "the President is head of state and the Prime Minister is head of "
             "government"),
        ],
    ),
    (
        "Tank destroyer",
        [
            ("how does this vehicle differ from a main battle tank",
             "tanks are designed for frontline combat combining mobility and "
             "offensive and defensive capability; the tank destroyer is designed "
             "specifically to engage and destroy enemy tanks"),
            ("what kind of chassis are these vehicles usually built on",
             "many are based on a tracked tank chassis, while others are wheeled"),
            ("what kind of missiles are used for long range antitank work today",
             "lightly armoured antitank guided missile (ATGM) carriers"),
        ],
    ),
    (
        "Northern cavefish",
        [
            ("how did one species become two distinct evolutionary lineages",
             "a 2013 study found the species divided into two lineages, one "
             "north of the Ohio River in Indiana and one south of it in Kentucky"),
            ("which journal published the paper that gave the northern lineage "
             "a new name",
             "a 2014 paper published in the journal ZooKeys"),
            ("which river is the northern limit of where either form is found",
             "neither is found north of the White River, south of Bedford, "
             "Indiana"),
        ],
    ),
    (
        "Politics of Angola",
        [
            ("how many members sit in the national assembly",
             "220 parliamentarians elected to the National Assembly"),
            ("who appoints the judges of the two highest courts",
             "the President of the Republic, together with the parliament"),
            ("which party has held executive power for decades",
             "the People's Movement for the Liberation of Angola"),
        ],
    ),
    (
        "Baltic Sea",
        [
            ("which two artificial waterways connect this sea to other waters",
             "the White Sea-Baltic Canal to the White Sea and the Kiel Canal to "
             "the German Bight of the North Sea"),
            ("which gulfs and bays break up its coastline",
             "the Gulf of Bothnia, the Gulf of Finland, the Gulf of Riga and the "
             "Bay of Gdansk"),
            ("why is it described as an inland sea",
             "limited water exchange with the Atlantic makes it an inland sea"),
        ],
    ),
    (
        "Geography of Antarctica",
        [
            ("how much of the world's fresh water is locked up in the ice",
             "90 percent of Earth's total ice volume and 70 percent of its fresh "
             "water"),
            ("which lake of liquid water lies under the ice",
             "subglacial lakes of liquid water also occur, e.g. Lake Vostok"),
            ("how much would sea level rise if the ice were to melt",
             "enough water to raise global sea level by 200 ft"),
        ],
    ),
    (
        "Blood on the Tracks",
        [
            ("which two cities were the recording sessions held in",
             "recorded in New York City in September 1974 and rerecorded in "
             "Minneapolis in December"),
            ("which record label was it released on after a two album stint "
             "elsewhere",
             "Columbia Records, after a two album stint with Asylum Records"),
            ("which of the artist's children described the songs as his parents "
             "talking",
             "Jakob Dylan"),
        ],
    ),
    (
        "British Columbia",
        [
            ("which indigenous peoples were among the earliest inhabitants",
             "the Coast Salish, Tsilhqot'in, and Haida peoples, among many others"),
            ("which city is the seat of government and which is the largest",
             "the capital is Victoria, while the largest city is Vancouver"),
            ("how many people live there and what coastline does it face",
             "an estimated population of over 5.5 million, situated between the "
             "Pacific Ocean and the Rocky Mountains"),
        ],
    ),
    (
        "George Abbot (bishop)",
        [
            ("which university did he serve as the fourth chancellor of",
             "the University of Dublin, from 1612 to 1633"),
            ("which of his brothers became lord mayor of a city",
             "Maurice became Lord Mayor of London"),
            ("which version of the Bible did he work on",
             "he was a translator of the King James Version of the Bible"),
        ],
    ),
    (
        "Mexican tetra",
        [
            ("what happens to the eyes and the colouring of the cave form",
             "its blind cave form is notable for having no eyes or pigment; it "
             "has a pinkish white colour resembling an albino"),
            ("which rivers in texas is it native to",
             "the lower Rio Grande and the Nueces and Pecos Rivers in Texas"),
            ("what water conditions does it prefer in a tank",
             "water with 6.5-8 pH, hardness of up to 30 dGH, and a temperature "
             "range given in the article"),
        ],
    ),
]


def main() -> int:
    store = DocumentStore.load_jsonl(CORPUS)
    by_title: dict[str, str] = {}
    for doc in store:
        if doc.title in by_title:
            raise SystemExit(f"duplicate title in corpus: {doc.title}")
        by_title[doc.title] = doc.id

    queries_path = Path(QUERIES_OUT)
    existing = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(row["stratum"] == TIER for row in existing):
        raise SystemExit(
            f"{TIER} queries are already in {queries_path}; delete them first rather "
            "than appending a second set with overlapping ids"
        )
    used = {row["source_doc_id"] for row in existing}
    pool = [d for d in store if d.id not in used]
    rng = random.Random(SEED)
    drawn = rng.sample(pool, N_ARTICLES)
    drawn_titles = sorted(d.title for d in drawn)

    spec_titles = sorted(title for title, _ in SPEC)
    if drawn_titles != spec_titles:
        raise SystemExit(
            "the sampled articles and the hand-written questions have drifted "
            "apart -- regenerate the questions for: "
            f"{[t for t in drawn_titles if t not in spec_titles]}"
        )

    rows: list[dict[str, object]] = []
    index = 0
    for title, questions in SPEC:
        doc_id = by_title[title]
        for query, answer in questions:
            index += 1
            rows.append(
                {
                    "query_id": f"B{index:02d}",
                    "query": query,
                    "stratum": TIER,
                    "source_doc_id": doc_id,
                    "derivation": "handwritten",
                    "notes": f"answer present in source: {answer}",
                }
            )
    if len(rows) != N_ARTICLES * QUERIES_PER_ARTICLE:
        raise SystemExit(f"expected {N_ARTICLES * QUERIES_PER_ARTICLE} rows, got {len(rows)}")

    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with queries_path.open("w", encoding="utf-8") as handle:
        for row in [*existing, *rows]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    sha_path = queries_path.parent / "queries.sha256"
    sha_path.write_text(f"{digest}  {queries_path.name}\n", encoding="utf-8")
    print(f"wrote {len(rows)} stratum-B queries -> {queries_path} ({len(existing)} kept)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
