import textwrap

from pydantic import BaseModel, RootModel

from drsd.llm import LLMClient
from drsd.schema import NewsPair, RelationshipSchema

PROPOSAL_PROMPT_TEMPLATE = """
You are an expert in news recommendation systems. Your task is to identify reusable semantic relationship schemas between clicked news article pairs.

Existing relationship types (DO NOT propose these or their synonyms):
{existing}

Previously rejected types (DO NOT propose these or their synonyms):
{rejected}

Task:
Analyze the news pairs below to discover complementary relationship schemas. Do NOT force the creation of a new label if existing ones suffice.

Strict Evaluation Rules:
1. Strictly Filter First: If a pair fits ANY Existing or Rejected type, ignore it. Do NOT paraphrase existing concepts.
2. Simplicity & Brevity (CRITICAL): Label names must be extremely short and fundamental (1-3 words max, e.g., "Background", "Local to Global"). Definitions must be a single, simple sentence. 
   - BAD Example: "Systemic Constraint to Alternative Resource Reclamation" (Too specific/verbose)
3. Domain-Agnostic Abstraction: Base relationships on structural intent, not specific topics.
4. Frequency Requirement: A new schema MUST clearly apply to MULTIPLE pairs in the provided list. Do not create a bespoke schema for just one unique pair.
5. Default to Empty: If no truly new, simple, and broadly applicable relationship exists across multiple pairs, you MUST return exactly: []

Return ONLY a valid JSON array of objects with fields name and definition.

Pairs to analyze:
{pairs}
"""


def propose_hypothesis_component(
    llm: LLMClient,
    schema: RelationshipSchema,
    news_pairs: list[NewsPair],
    seed: int,
) -> tuple[dict[str, str], str]:

    prompt = PROPOSAL_PROMPT_TEMPLATE.format(
        existing="\n".join(
            f"- {relationship.name}: {relationship.definition}" for relationship in schema.accepted_relationships
        ),
        rejected=(
            "\n".join(
                f"- {relationship.name}: {relationship.definition}" for relationship in schema.rejected_relationships
            )
            if schema.rejected_relationships
            else "None"
        ),
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

    class Response(BaseModel):
        name: str
        definition: str

    response = llm.generate_content(
        prompt=prompt,
        response_schema=RootModel[list[Response]],  # type: ignore
        seed=seed,
    )
    responses = RootModel[list[Response]].model_validate(response).root
    hypothesis = {r.name: r.definition for r in responses}

    return hypothesis, prompt
