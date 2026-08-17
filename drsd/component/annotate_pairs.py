import textwrap
from enum import Enum

from pydantic import BaseModel, RootModel

from drsd.llm import LLMClient
from drsd.schema import NewsPair, RelationshipSchema

LABELING_PROMPT_TEMPLATE = """
You are an expert in news recommendation systems. Your task is to annotate the relationship between pairs of clicked news articles.

Defined Relationship Types:
{relationships}
- Unknown: Use this label if the pair does not clearly fit any of the predefined schemas above.

Task:
Analyze the news pairs below. For each pair, choose the single relationship type from the list above that best explains the logical connection between the articles.

Strict Rules for Annotation:
1. Single Choice: Select exactly one relationship type per pair.
2. Strict Matching (Conservative Annotation): Do not force a fit. If the relationship is weak, ambiguous, or requires a schema not present in the defined types, you MUST choose "Unknown".
3. Exact Labels: The output relationship must exactly match one of the provided names or "Unknown".

Return ONLY a valid JSON array of objects with fields `no` and `relationship`.

Pairs to analyze:
{pairs}
"""


def annotate_pairs_component(
    llm: LLMClient,
    schema: RelationshipSchema,
    news_pairs: list[NewsPair],
    seed: int,
) -> tuple[list[NewsPair], str]:

    prompt = LABELING_PROMPT_TEMPLATE.format(
        relationships="\n".join(f"- {r.name}: {r.definition}" for r in schema.accepted_relationships),
        pairs="\n".join(
            textwrap.dedent(
                """
                {
                    "no": %d,
                    "query_title": "%s",
                    "target_title": "%s"
                }
                """
                % (i, article.query_news.title, article.target_news.title)
            ).strip()
            for i, article in enumerate(news_pairs)
        ),
    )

    RelationshipType = Enum(
        "Relationship", {r.name: r.name for r in schema.accepted_relationships} | {"Unknown": "Unknown"}
    )

    class Response(BaseModel):
        no: int
        relationship: RelationshipType  # type: ignore

    response = llm.generate_content(
        prompt=prompt,
        response_schema=RootModel[list[Response]],  # type: ignore
        seed=seed,
    )
    responses = RootModel[list[Response]].model_validate(response).root

    outputs: list[NewsPair] = []
    for r in responses:
        news_pair = news_pairs[r.no]
        news_pair.relationship = "Unknown" if r.relationship == RelationshipType.Unknown else r.relationship.name
        outputs.append(news_pair)

    return outputs, prompt
