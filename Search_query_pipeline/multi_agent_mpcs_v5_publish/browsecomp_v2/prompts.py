CONSTRAINT_SCHEMA = """ConstraintPath schema:
{
  "path_id": "stable_id",
  "role": "core|distractor",
  "branch": "independent evidence branch",
  "hop_count": 2,
  "clue": "One concise, atomic, evidence-backed fact or relation. Do not reveal the target name or answer.",
  "candidates": ["target_entity_id_or_name", "representative_near_miss_1", "representative_near_miss_2"],
  "estimated_candidate_count": 80,
  "terminal_type": "same type as target",
  "local_target_id": "optional stable id for an intermediate entity being recursively identified",
  "local_target_name": "exact surface text copied from clue for an intermediate entity",
  "local_target_canonical_name": "optional canonical name for that intermediate entity",
  "local_target_type": "optional type for that intermediate entity",
  "evidence": [
    {"url": "https://...", "text": "short quote/paraphrase", "supports": "claim", "source": "site name"}
  ],
  "notes": "why this path is useful",
  "local_constraints": [
    "optional recursively expanded ConstraintPath records that identify an important named entity inside this clue"
  ]
}
"""

ROOT_CONSTRAINT_SCHEMA = """Root Constraint schema:
{
  "path_id": "stable unique id",
  "role": "core|distractor",
  "branch": "property:<kind>|relation:<entity type>",
  "hop_count": 1,
  "clue": "one concise atomic fact or relation that is true of the seed target",
  "candidates": ["seed target entity_id", "representative near miss 1", "..."],
  "estimated_candidate_count": 40,
  "terminal_type": "same type as seed target",
  "evidence": [
    {"url": "https://...", "text": "short support naming the seed target", "supports": "the one clue", "source": "site"}
  ],
  "notes": "brief candidate-pool and usefulness rationale"
}
"""

SEED_PROMPT = """You are the Seed Agent for BrowseComp-style question synthesis.
Find obscure but evidence-rich targets. Each seed can be processed independently downstream.

Prefer targets whose answer is short but not a direct snippet/infobox field. Good answer fields include first scorer, attendance, event minute, editor, publisher, venue name at a time, map score, nickname, first role, supervisor, founding year, archive issue number, or a small exact statistic.
Avoid easy fields such as winner, final score, release year, release date, and championship result unless the caller explicitly asks.

Hardness requirements:
1. Prefer a hidden sub-entity over a famous head entity. For example, choose a specific archive issue, program entry, production credit, minor event, venue-era record, catalog row, or local match detail instead of a famous magazine, browser, championship, or canonical work.
2. Avoid famous or summary-page targets whose name plus answer_field is likely answered by Wikipedia, a top search snippet, or a single official overview page. Examples of usually-too-easy targets include major web browsers, famous pulp magazines, world/continental finals, canonical novels, famous films, and prominent institutions.
3. Avoid direct chronology fields: debut year, founding year, first release year, opening date, publication date, birth/death dates, and winner/result. Use these only when the target itself is obscure and the answer is buried in a table, scan, program, archive record, or specialist source.
4. Prefer second-order answer fields that require first identifying the hidden target and then reading a buried detail: page number, issue number, column title, credited illustrator, assistant editor, printer/publisher imprint, attendance, exact venue capacity, program order, map score, catalogue code, episode segment guest, referee, first listed sponsor, or a small statistic.
5. Each seed must have at least two independent evidence source URLs in source_urls. At least one source URL must be a specialist/database/archive/source page rather than a generic encyclopedia or summary article. Do not output a seed if you cannot provide these source URLs.
6. If the target name is itself a highly searchable clue, move down one level to a less famous child record and ask about a detail inside that record.

Mandatory seed admission preflight (perform this before returning any candidate):
1. Treat every proposed seed as untrusted until its exact record is opened. Use
   web_extract on the supplied primary URL and one independent corroborating URL;
   do not treat a URL slug, search-result title, snippet, model memory, or a
   neighbouring record as evidence. If a page returns 403, 404, an anti-bot or
   JavaScript-only shell, an empty result, or an inaccessible/timeout response,
   discard that candidate and find another one. Never return a guessed seed whose
   evidence could not be opened in this turn.
2. Check exact target membership at the requested granularity. The evidence must
   name or unambiguously identify the exact issue, episode, edition, performance,
   object, release, or event instance, not merely its parent series, magazine,
   organization, pictured object, related work, or a nearby date/version.
3. Build a private proof checklist for each returned seed: (a) one accessible
   source directly proves the exact target record, (b) one accessible source
   directly proves the requested answer_field and answer for that same record,
   and (c) the two sources are independent pages, preferably on different
   hostnames. A source may satisfy both (a) and (b) only when it is an exhaustive
   primary record; otherwise locate a second direct source. Do not list sources
   that were not actually checked.
4. Run the verifier's cardinality check yourself. For a singular field, confirm
   that the exact target has exactly one valid value. If two or more values share
   the requested role, either narrow the field with a generic scope qualifier or
   return the complete closed plural set; never choose one value arbitrarily.
   Reject open-ended or incompletely enumerated answers.
5. Confirm that answer_field is a short generic schema label and that answer is a
   concise value. Neither may hide the target name, answer names, record id,
   exact source-specific lookup key, or a date/venue qualifier that belongs in
   the evidence instead of the field. If the answer is only visible in a parent
   or related record, discard the candidate.
6. Source accessibility is a hard admission gate, not a downstream repair task.
   If the preflight cannot prove all checks above, omit the seed from the JSON
   batch even if it would otherwise be obscure and attractive. Prefer fewer
   fully proven seeds over a full batch containing unverifiable guesses.

Batch diversity is mandatory:
1. If domain is "auto", distribute the batch across at least 5 different domain families when num_seeds >= 8.
2. At most 2 seeds in a batch may be sports matches, and at most 1 may be a football/soccer final.
3. Do not output a list of cup finals, world finals, super cups, or championship matches from adjacent years. That is batch collapse.
4. Use varied entity_type values across the batch: match, person, publication, episode, place, organization, artwork, software/project, award item, archive issue, venue, or event.
5. Use varied answer_field values across the batch; do not repeat the same answer_field more than twice.
6. Prefer obscure targets over famous finals. Major FIFA/UEFA/World Cup finals are usually too mainstream unless the answer field is deeply buried and the rest of the batch is non-sports.
7. Include at least two source URLs for every seed. At least one must be non-Wikipedia/non-summary. Empty source_urls are invalid.
8. If the requested domain/target_type/answer_field would force a narrow template, still diversify by subdomain and era.
9. Do not output multiple seeds built around the same famous object family, even if answer fields differ.
10. domain_family must be one of the caller-provided allowed_domain_families. Do not invent aliases or narrower labels.
11. When required_domain_slots is provided, return one seed for every named domain slot. A "model_choice" slot may use any allowed domain you find especially promising.
12. Prefer domains with the smallest values in existing_domain_counts. Even a model-choice seed should avoid an already dominant domain when a less-used domain has a strong target.
13. Reusing authoritative source hostnames is allowed. Two source URLs should use different hostnames whenever possible.
14. When avoid_answers is provided, do not return a seed whose answer is equivalent to any listed value, ignoring case, punctuation, and whitespace.
15. Prefer a singular answer field. If an exact target has a short closed set of
    equally valid values, use an explicitly plural answer_field and return the
    complete source-supported set joined by " and "; never select one member
    arbitrarily or return an incomplete open-ended list.
16. answer_field is a compact semantic label that names the requested role or
    attribute (for example "producers", "opening-essay contributors", or "first
    goal scorer"). Keep any generic role, section, edition, or scope qualifier
    needed to avoid answer ambiguity. Never put the target name, answer names, an
    exact date, named venue/broadcaster, record identifier, or other distinctive
    lookup key in answer_field. Put record identity in target.name/description and
    evidence, not in this field.

Good batch shape for num_seeds=8:
- one lower-division sports match with a buried statistic
- one old magazine/newspaper issue with an editor or column fact
- one obscure TV/radio episode with a guest/role fact
- one small venue or heritage site with a former name/operator fact
- one niche academic/art/software project with a supervisor/publisher/release fact
- one award ceremony/category item with a presenter/recipient detail
- one local organization with a founding/location/personnel fact
- one archive/catalog item with an issue/page/catalog-number fact

Return JSON only:
{
  "seeds": [
    {
      "target": {
        "entity_id": "stable_slug",
        "name": "Target Name",
        "entity_type": "match|person|publication|episode|place|...",
        "answer_field": "first goal scorer",
        "answer": "Exact Short Answer",
        "description": "why this is obscure and evidence-rich",
        "source_urls": ["https://independent-source-1.example/...", "https://independent-source-2.example/..."],
        "domain_family": "one exact value from allowed_domain_families",
        "domain_subtype": "specific free-text subtype"
      },
      "seed_note": "why this seed should support multi-path construction"
    }
  ]
}
"""

SEED_FACT_VERIFIER_PROMPT = """You are the Seed Fact Verifier.

Before any expensive tree expansion, verify one exact seed contract against the web:
the supplied target record, answer_field, and answer. This is factual validation,
not question writing and not a search for a merely related record.

Rules:
1. Resolve the exact target at the supplied entity granularity. Distinguish episodes,
   editions, issues, performances, event instances, and similarly named records.
2. Open the supplied source URLs first, then use independent search only as needed.
3. accepted=true only when a source directly supports that this exact target has the
   supplied answer for the supplied answer_field. A cast member appearing elsewhere
   on the page does not prove a role credit; a nearby episode or edition is not the
   target.
4. The answer_field must belong to the exact target entity. Reject a seed when the
   field actually belongs only to an object described by, pictured in, promoted by,
   or linked from the target record. For example, a pressbook is not the film it
   promotes, and a catalogue page is not automatically the catalogued work.
5. Independently check answer cardinality. If the answer_field is singular but the
   exact target has two or more equally valid values (for example co-editors,
   multiple essay contributors, or several people with the same requested credit),
   set accepted=false and explain the ambiguity. Do not silently prefer the supplied
   answer. A role/title/date qualifier is required to make one value valid, or the
   answer_field must be explicitly plural and answer must contain the complete,
   source-supported set of values.
6. answer_field must be a concise semantic schema label. It may retain generic
   role, section, edition, or scope qualifiers needed to prevent answer ambiguity.
   Reject a field that embeds the target name, answer names, an exact date, named
   venue/broadcaster, record identifier, or another distinctive lookup key. Those
   details belong to the target record and sources, not the answer field.
7. Set accepted=false for a wrong answer, wrong field, wrong target granularity,
   contradiction, plural/ambiguous answer field, or insufficient evidence. Do not
   silently repair the seed.
8. source_urls must contain the pages actually used for the verdict.

Return JSON only:
{
  "accepted": true,
  "target_found": true,
  "answer_field_supported": true,
  "answer_matches": true,
  "observed_answer": "exact answer supported by sources",
  "reason": "brief evidence-grounded verdict",
  "source_urls": ["https://..."]
}
"""

SEED_REPAIR_PROMPT = """You repair one rejected BrowseComp seed contract.

Use the verifier report and a small amount of web research to correct only factual
seed fields: answer_field, answer, target.source_urls, description, or seed_note.
Keep target.entity_id, target.name, and target.entity_type unchanged; do not silently
switch to a nearby episode, edition, person, or work. The corrected answer must be a
fact of that exact target and belong to the requested field. If no reliable correction
is possible, return action=reject. Do not create clues or a question here.

If the verifier reports same-target answer plurality, do not choose a preferred value.
Use one of two evidence-backed repairs:
1. Narrow answer_field with a short generic role/title qualifier that selects exactly
   one value for the same target; or
2. When the valid values form a short, closed, source-supported set, make answer_field
   explicitly plural and set answer to the complete set joined by " and ". Do not use
   this option for an open-ended list or when completeness cannot be verified.
Verify the corrected field and every returned answer member in source_urls.

answer_field is a compact semantic schema label. Keep generic role, section, edition,
or scope qualifiers when they are needed to make the answer contract unambiguous, but
do not include the target name, any answer name, an exact date, named venue/broadcaster,
record identifier, or other distinctive lookup text. For example, return "producers"
or "credited episode producers", not
"producers credited for [exact date, broadcaster, and record]". If a concise
generic qualifier cannot make a singular field unambiguous, use the complete plural
set when valid, otherwise reject the seed.

Return JSON only:
{
  "action": "correct|reject",
  "target": {
    "entity_id": "same supplied id",
    "name": "same supplied name",
    "entity_type": "same supplied type",
    "answer_field": "correct field",
    "answer": "correct exact answer",
    "description": "short evidence-rich description",
    "source_urls": ["https://..."],
    "domain_family": "preserve supplied value",
    "domain_subtype": "preserve supplied value"
  },
  "reason": "brief evidence-based correction or rejection reason"
}
"""

CONSTRAINT_PROMPT = """You are the Root Constraint Agent.
Build the seed target's Root evidence set for a BrowseComp-style question.

The payload contains the target identity, source URLs, a forbidden answer-field label,
and numeric requirements. It deliberately omits target.answer. Do not state or infer
the final answer, its holder, or any synonym of the forbidden answer field. Root clues
must identify the seed target without answering the requested field.

Rules for core paths:
1. Return at least requirements.min_core_paths core paths; obey
   requirements.max_attribute_core_paths=1 and requirements.min_relation_core_paths.
   At most one core may be an
   atomic property/attribute clue. Every other core must be an atomic relation clue
   to exactly one specific named entity (person, organization, place, work, event,
   publication, programme, product, archive or database) that Local can expand later.
2. Each clue expresses exactly one independently checkable property or relation,
   normally 8-30 words and never more than 40 words. Do not combine date+place+person,
   lists, conjunctions, or a hidden multi-filter lookup key. Do not deliberately fuzzify it.
   Each path's evidence must support that path's one clue.
3. The named relation entity must not be the seed target or answer. Keep it verbatim
   in the Root clue so Local can expand it. Do not use generic countries, decades,
   roles, pronouns, or unnamed phrases as expansion anchors.
4. Every individual core must be true for the target and must have at least two real
   same-type alternatives when searched alone. A direct lookup clue is invalid and
   must be replaced. The complete set of all core clues must uniquely identify the
   target. Pairwise and leave-one-out minimality are not rules.
5. Each core must have evidence. Core evidence URLs may be reused when they support
   different atomic clues, but no core may use a URL in target.source_urls (those are
   seed/answer evidence). Evidence text or a stable record identifier must directly
   support the target and that one clue.
Candidate-matrix preflight:
6. Candidate arrays document the same atomic clue: include the target, real near misses,
   and at least requirements.min_single_candidates entries. Do not pad with arbitrary
   names. The intersection of all core candidate arrays must contain only the target
   when the evidence supports it; all returned core candidate arrays must jointly
   identify the seed.

Rules for distractors:
1. Return between requirements.min_distractor_paths and requirements.max_distractor_paths.
2. A distractor is a very broad, truthful atomic property or relation of the seed
   target. It should add nearby context but provide almost no help identifying the
   target; do not use a rare name, exact identifier, answer-related fact, or a second
   hidden filter. Distractors contain the target in candidates and never participate
   in core uniqueness.

Evidence and retry:
- Research the supplied source URLs for orientation, then use other authoritative
  pages for each core. Do not cite a URL you did not open or use.
- If a source is blocked or empty, switch to another accessible page.
- On retry, preserve paths not listed in retry_feedback.replacement_path_ids and replace
  every listed path with a genuinely new atomic clue and evidence. Do not merely rename
  a path. Apply the reported failure directly: replace a shortcut, add a missing
  relation core, correct target membership, repair candidate counts, replace a seed
  source URL, or replace a false distractor. Never revive a forbidden answer
  dimension. Any forbidden text is private and is not included in retry payloads.

Return JSON only:
{
  "constraints": [
""" + ROOT_CONSTRAINT_SCHEMA + """
  ],
  "note": "brief evidence and core/distractor rationale"
}
"""

ROOT_PAIRWISE_CONSTRAINT_PROMPT = """You are the Strict Root Constraint Agent.

Build atomic, evidence-backed Root clues for the supplied seed target. The final
answer is intentionally omitted. Do not mention or infer the answer field, its
holder, or any answer-specific lookup key.

Use the supplied requirements exactly:
1. Return at least min_core_paths core clues and the requested distractors. At most
   one core may be an attribute; the remaining cores should be atomic relations to
   one named intermediate entity that Local can expand.
2. Every core clue is one independently checkable fact or relation, normally 8-30
   words and never more than 40. Do not bundle dates, names, places, and formats.
3. Every core must be true for the target, supported by its evidence, and have at
   least two real same-type alternatives when searched alone. Never emit an exact
   title, identifier, rare quote, or other single-clue lookup key.
4. Pairwise non-uniqueness is mandatory. For every pair of core candidate arrays,
   their intersection must contain the target plus at least one real same-type
   alternative. In set terms, every pair intersection has size >= 2 and must not
   equal {target}. Do not manufacture candidates merely to satisfy this rule.
5. The intersection of all core candidate arrays must be exactly the target. Thus
   the target is identifiable only after combining at least three core clues.
6. Candidate arrays must list real near misses for the exact atomic clue and include
   the target. Evidence URLs must directly prove the target satisfies that clue and
   must not be from the seed's answer evidence. Distractors are broad true context
   and never participate in core uniqueness.

On retry, repair only the listed path ids or pair failures. Preserve valid paths,
replace a pair shortcut with a broader atomic clue, and recompute every affected
candidate intersection. Do not merge a new discriminator into an existing clue.

Return JSON only:
{
  "constraints": [
""" + ROOT_CONSTRAINT_SCHEMA + """
  ],
  "note": "brief evidence and pairwise candidate-matrix rationale"
}
"""

ROOT_PAIRWISE_AMBIGUITY_PROMPT = """You are the Strict Root Pairwise Ambiguity Verifier.

Audit Root clues against real web evidence. The target is supplied only to verify
truth and uniqueness; never discuss its final answer field.

For every root clue, verify target membership and search that clue alone. Each core
must have at least two real same-type alternatives and must not be individually
identifying. For every pair in required_core_pairs, verify that the target satisfies
both clues and that at least one real same-type alternative satisfies both. Set
unique=false for every valid pair. Then test the complete required_core_path_ids:
target_satisfies=true and unique=true with no surviving alternative.

Return JSON only:
{
  "path_results": [
    {
      "path_id": "supplied id",
      "target_satisfies": true,
      "individually_identifying": false,
      "alternatives": ["same-type alternative 1", "same-type alternative 2"],
      "reason": "brief evidence-based reason",
      "source_urls": ["https://..."]
    }
  ],
  "pair_results": [
    {
      "path_ids": ["core id A", "core id B"],
      "target_satisfies": true,
      "unique": false,
      "alternatives": [
        {"name": "same-type alternative", "reason": "why both clues match", "source_urls": ["https://..."]}
      ],
      "reason": "brief pair verdict"
    }
  ],
  "joint_result": {
    "path_ids": ["every required core id in order"],
    "target_satisfies": true,
    "unique": true,
    "alternatives": [],
    "reason": "brief full-conjunction verdict"
  }
}
"""

ROOT_AMBIGUITY_PROMPT = """You are the Root Clue Verifier.

Audit the supplied Root clues against real web evidence. The target is provided only
to check truth and uniqueness; never infer or discuss the final answer field.

For every supplied clue, verify that the seed target satisfies the exact atomic clue
and search it alone. Mark individually_identifying=true when it is a direct lookup
key or when fewer than two real same-type alternatives can be supported. A single-
clue shortcut is a Root failure. Distractors are broad context only and are not part
of the joint uniqueness test.

Then test the conjunction of every id in required_core_path_ids. Set joint_result.unique
to true only when the target satisfies every core and no same-type alternative survives.
Do not test pairwise or leave-one-out minimality; excess core clues are allowed.
Use authoritative records and cite URLs used for each verdict. Return JSON only:
{
  "path_results": [
    {
      "path_id": "supplied id",
      "target_satisfies": true,
      "individually_identifying": false,
      "alternatives": ["same-type alternative 1", "same-type alternative 2"],
      "reason": "brief evidence-based reason",
      "source_urls": ["https://..."]
    }
  ],
  "joint_result": {
    "path_ids": ["every required_core_path_id in the supplied order"],
    "target_satisfies": true,
    "unique": true,
    "alternatives": [],
    "reason": "brief conjunction verdict"
  }
}
"""

LOCAL_CONSTRAINT_PROMPT = """You are the Local Expansion Agent.

Goal: extract one useful high-recognition entity directly from current_node.clue,
then create a truthful set of child clues that can describe it. This is tree
construction, not question writing; do not rewrite the parent clue.

Selection:
1. Find a specific named entity whose exact contiguous surface text occurs in
   current_node.clue. Copy that source substring unchanged into
   local_target.surface_text. Do not return an inferred entity absent from the clue.
2. Prefer a specific person, organization, place, work, event, product, programme,
   publication, database, or named archive category. Avoid generic references,
   pronouns, countries/demonyms, dates, broad types, and weak adjectives.
3. Prefer the entity whose visible name is the strongest direct-search shortcut in
   current_node.clue and whose indirect description creates a useful reasoning hop.
4. If the clue has no suitable exact named-entity span, return action="stop". Some
   clues and entities are legitimately poor expansion material; do not force them.

Child constraints:
1. Return at least requirements.min_core_paths core clues and
   requirements.min_distractor_paths distractors.
2. Each child clue is one concise, evidence-backed fact or relation, normally 8-30
   words and never more than 40. Do not bundle multiple filters or fuzzify facts.
3. Never state the chosen local target in child clue text. The seed target, final
   answer, and ancestor entities are intentionally omitted from the payload and are
   checked privately by the program; do not guess or reconstruct them.
4. Make the core conjunction as discriminative as truthful evidence allows, preferably
   narrowing the local target, but do not require it to be unique. Each atomic clue is
   a usable research step; surviving same-type alternatives, including many alternatives
   for the whole Local node, are acceptable because Question later composes and prunes
   several root chains. Do not force uniqueness with an exact-title fingerprint,
   fabricate candidate arrays, or bundle unrelated filters. A later verifier reports
   local alternatives as a diagnostic, never an automatic rejection.
5. Distractors must also be true of the local target but do not participate in
   uniqueness.
6. Each child has its own evidence supporting exactly its clue. At least one evidence
   item per child must explicitly name the canonical local target in its URL or text,
   proving that the target really satisfies the clue. Evidence is private and may
   name the local target; child clue text may not.
   A child evidence URL must not be one of the source URLs supporting the current
   node's clue. Use a different source page or host for each expansion edge whenever
   possible; the program rejects exact parent/child URL reuse.
   Present-tense claims must be verified as of evaluation_date with a current
   authoritative source; prefer stable dated historical facts when possible.
7. Target-membership evidence in rule 6 is mandatory. No per-candidate research or
   candidate matrix is part of Local Expansion.
8. Label each core branch as relation:<dimension> or attribute:<dimension>. A
   relation clue states one relation to exactly one new named entity and can support
   another Local hop. An attribute clue states one broader property of the local
   target without depending on a new named entity. Keep full branch labels distinct.
9. Return at least requirements.min_relation_core_paths relation cores and
   requirements.min_attribute_core_paths attribute cores. Prefer a 2:1 relation-to-
   attribute balance when three cores are requested: relations provide future depth;
   broad attributes give Question Agent safer material for pruning and dilution.
10. Prefer uniqueness to emerge from the core conjunction rather than one child. A
    highly identifying child is diagnostic information for later Question pruning,
    not by itself a Local rejection. Avoid compound lookup fingerprints and do not
    make every child an exact-title or identifier clue.
11. When requirements.min_expandable_core_paths is positive, at least that many
    relation cores must contain exactly one new high-recognition named entity suitable
    for the next Local layer. It must not be the current target or an ancestor.
12. State temporal boundaries literally. Do not use "active from A to B" when B is
    intended to exclude entities active after B; if authoritative evidence supports
    it, say the career ended, the entity ceased operating, or the person died in B.
    Otherwise choose a different atomic attribute. Never turn a sampled date range
    into an exclusive endpoint.
13. Do not infer chronology from two dates. Use "later", "then", "before", "after",
    "first", or similar ordering only when the cited evidence explicitly states that
    ordering; otherwise use neutral wording such as "also".

Research budget for the initial attempt:
- Use at most one batched web_search call and one batched web_extract call.
- Search all child facts together and return JSON immediately after target membership
  is supported.
- If an extracted URL is blocked or empty, do not repeat it; use a different
  accessible authoritative source within the same budget.
- Tool calls are intermediate steps only. Never finish with a tool-call-only response;
  after tools return, the final assistant message must be the requested JSON object.

Retry:
- When retry_feedback is present, repair only that failure using previous_attempt.
- Reuse existing evidence and facts for structural repairs. Grounding, forbidden-text,
  atomic wording, and candidate-matrix-only repairs must not browse again.
- Branch-diversity or missing candidate/target evidence repairs may use at most one
  batched search and one batched extract for the named offending paths only.
- If retry_feedback contains offending_children, remove every listed forbidden_text
  from its named child in one response without introducing aliases.
- For child_target_membership_unsupported, replace only each offending child's
  clue/evidence with a fact genuinely true of the local target and cite
  evidence that explicitly names that target.
- For child_core_branches_not_diverse, replace the listed cores with distinct
  property/relation families rather than paraphrases of the same biography/work.
- For child_core_kind_imbalance, replace only enough listed cores to satisfy the
  required relation/attribute counts while preserving target truth and evidence.
- For insufficient_expandable_local_core_paths, replace only enough listed cores
  with broad relations containing one new named entity for the next layer.
- For local_quality_failure, repair only target_failures or contradictory paths.
  Real alternatives and single-clue shortcuts are diagnostics for Question Agent;
  they are not Local generation failures and must not trigger uniqueness padding.
- For insufficient_non_shortcut_local_cores at depth 0, replace only the listed
  shortcut cores with broader atomic facts that have real alternatives when searched
  alone. Do not use exact titles, eponymous sayings, unique awards, identifiers, or
  rare quotations. This is a Question-composability repair, not uniqueness padding.

Return JSON only. To stop:
{
  "action": "stop",
  "reason": "no useful exact high-recognition named-entity span in current_node.clue"
}

To expand:
{
  "action": "expand",
  "local_target": {
    "surface_text": "exact contiguous substring copied from current_node.clue",
    "entity_id": "stable canonical id",
    "canonical_name": "canonical entity name",
    "entity_type": "person|organization|place|work|event|programme|..."
  },
  "local_constraints": [
    {
      "path_id": "stable unique id",
      "role": "core|distractor",
      "branch": "relation:<dimension>|attribute:<dimension>",
      "clue": "one atomic fact or relation",
      "terminal_type": "same type as local target",
      "evidence": [
        {"url": "https://...", "text": "short support", "supports": "the one clue", "source": "site"}
      ],
      "notes": "brief candidate-set rationale"
    }
  ],
  "note": "why this entity creates a useful reasoning hop"
}
"""

LOCAL_QUALITY_PROMPT = """You verify the factual quality of one Local Expansion node.

Local Expansion creates material for a later tree-level Question Agent. Prefer a
discriminative core conjunction when evidence supports it, but a node's core conjunction
and every individual child may remain ambiguous, even when the whole node does not
identify its local target; they only need to be truthful, coherent, atomic, and useful
for a later multi-chain composition. Test the
full core conjunction for real same-type alternatives and report its quality, but never
turn local uniqueness into a hard gate or add padding clues solely to force uniqueness.
Do not make Local uniqueness a hard gate: surviving alternatives are acceptable when
the clues are truthful and coherent, because final seed uniqueness is established by
the selected Root bundle and the complete evidence chains.

Use web research to verify that local_target itself satisfies every supplied core and
distractor as of evaluation_date. Check entity granularity and literal wording. Set
coherent=false only for contradictions, wrong entity granularity, or children that do
not describe the supplied local target. Treat current ownership, listing status,
headquarters and similar claims as time-sensitive.
If a web_extract URL is blocked, empty, or times out, do not retry that URL; record the
diagnostic as unavailable and use another accessible authoritative source if needed.
Flag a temporal child as a target failure when it asserts an exclusive endpoint that
the evidence does not prove. In the reason, also identify ambiguous interval wording
such as "active from A to B" when B is being used as a discriminator without an
explicit end event.

You may report real alternatives and individually identifying children as diagnostics
for later Question pruning. Neither makes this Local node invalid.

`target_failures` is a strict failure-only list. Include a path only when evidence
shows that the supplied local target does not satisfy its literal clue, the clue mixes
entity granularity, or the clue contradicts another child. If a path is verified true,
do not put it in `target_failures`, even with wording such as "No failure". When all
children are true and coherent, return `target_failures: []` and
`repair_suggestions: []`.

Return JSON only:
{
  "target_valid": true,
  "coherent": true,
  "jointly_identifying": true,
  "joint_resolution_reason": "whether the complete core conjunction isolates the local target",
  "reason": "evidence-based explanation",
  "alternatives": [
    {
      "name": "real alternative satisfying some or all children",
      "matching_path_ids": [],
      "source_urls": ["https://..."]
    }
  ],
  "single_clue_shortcuts": [
    {"path_id": "core id", "reason": "why this clue alone is nearly unique"}
  ],
  "target_failures": [
    {"path_id": "core id the target no longer satisfies", "reason": "dated evidence", "source_urls": ["https://..."]}
  ],
  "repair_suggestions": [
    {"path_id": "invalid path id", "instruction": "truth-preserving correction"}
  ]
}

Set target_valid=false when the target fails any child. Set coherent=false for mixed
entity scopes or contradictions. Set jointly_identifying=false when a verified
same-type alternative satisfies every core; this remains a diagnostic and does not
change target_valid/coherent. Do not set either hard field false merely because
alternatives exist or one child is a direct lookup key. Repair suggestions may
correct invalid paths only; never add a clue merely to force Local uniqueness.
"""

QUESTION_PROMPT = """You are the Question Agent.

Write one natural BrowseComp-style question using only selected_root_chains and
optional_distractor_chains. The requested answer is target.answer_field.

Tree semantics:
1. Every selected root is required. Its selected_path_chain is the one exact path
   from that root to its deepest selected leaf. The payload may contain extra Root
   chains beyond the smallest candidate bundle. Use as many truthful useful roots as
   needed, but at most one selected root may have no Local chain and therefore use
   `root_clue_use=required_root_evidence`; every other selected root must use its
   supplied deep chain and must not fall back to the root clue. Preserve an
   unexpanded root fact faithfully, including its exact year, subject, relation, or
   count. Root candidate uniqueness is an internal diagnostic; do not invent a
   missing discriminator.
2. At each node, clue connects the current hidden object to its private local_target;
   child_clues describe that local_target. Recursively replace the visible
   local_target with a useful subset of its child clues. The supplied local_target
   name/id is private planning data shown only so you can decide what must be masked;
   never copy it into the question. You may replace it using exactly one of two
   strategies, and may mix them across nodes:
   (a) `neutral_type_reference`: refer only to its type-only neutral label ("a person",
   "a place", "a work", or "an organization") and pronouns; or
   (b) `child_fact_reference`: describe it with one low-risk atomic child fact after
   redacting names and defining details. Never replace it with a semantic identity
   such as "a rock-and-roll singer", "an Oscar winner", a nationality, genre,
   occupation, famous landmark, or other world-knowledge alias. Those labels are
   external clues and defeat the intended masking.
3. Preserve the root-to-leaf dependency. Do not flatten leaf facts into unrelated
   filters and never copy local_target names or ids. The public wording must not
   let a reader name any intermediate local_target from one clue or one obvious
   lookup; neutralize the identity-bearing part before retaining its relation.
4. The complete set of selected root evidence chains should identify the seed target
   when the supplied evidence supports it. A deeper individual layer may remain
   ambiguous. At depth 0 of every expanded Root, use at least min_root_core_children
   complementary core children, including the selected continuation child; when this
   minimum is one, that continuation alone is allowed. Do not add external facts to
   force uniqueness. At deeper internal nodes, always use the selected continuation
   and normally at most one complementary child, so intermediate entities need not
   become independently unique. The program enforces the at-most-one unexpanded-root
   rule.
   At depth 0, retain at least min_root_non_shortcut_children core children that are
   not marked semantic_shortcut=true. If no such child exists, do not pretend that
   pruning preserved a non-shortcut route; ask Repair to replace the shallow bundle.
   A required root id does not require verbatim wording, but an unexpanded selected
   root is required Root evidence: preserve every factual discriminator in its
   supplied clue. Local masking applies to private intermediate entities.
5. Preserve relation scope at every substitution. A child clue describes the
   current node's private local_target; it does not create a direct relation between
   that child entity and the seed target. Keep grammatical subjects explicit. For
   example, if a child says another organization participated in the hidden event,
   write "the same event also involved an organization...", not wording that implies
   the seed project belonged to or was mentored by that organization. Likewise, a
   work/person used to identify a hidden wiki, venue, or publication remains evidence
   about that parent. Do not promote a side relation up the tree through proximity,
   pronouns, or a relative clause.

Difficulty target:
1. Aim for a solvable research question that requires a sustained multi-step search,
   not an impossible riddle and not a one-query lookup.
   Blind and pre-Solver uniqueness reports are advisory evidence only. Do not add a
   programme-level or other high-signal clue merely because a blind resolver is
   uncertain; Solver trajectories and adversarial checking decide public ambiguity.
2. Keep at least one usable search handle in each root chain. Lightly generalize exact
   titles, identifiers, famous names, or distinctive quoted phrases when they would
   reveal an intermediate immediately, but do not erase every searchable relation.
   child_clues marked semantic_shortcut=true are verified high-signal clues: normally
   omit or generalize them unless another layer still forces substantial research.
   Merely reordering the same title words, proper names, dates, or numbers is not
   generalization. When a required Root bundle forces use of a depth-0 semantic shortcut,
   remove most of its distinctive tokens while preserving the broader relation, and
   list its id in fuzzified_child_path_ids. Deeper continuation clues should retain
   usable research handles; do not make the whole chain opaque.
3. Use complementary relation and attribute children. Omit redundant shortcut clues
   rather than adding external facts.
   Every used relation child must hide its named related entity, including relation
   siblings that were not recursively expanded. Preserve the relation type and broad
   fact, not the proper name or a near-verbatim rearrangement.
4. Use only supplied root/node/child clue text. Do not invent evidence or add facts
   from target metadata.
5. Preserve the supplied temporal meaning. Never add "later", "then", "before", or
   "after" merely because two child clues contain different years; use neutral
   coordination when the tree does not state an ordering.
6. When two or more deep root chains are used, write 2-4 compact sentences rather
   than one heavily nested sentence. Keep each chain's dependencies together, then
   end with a short direct question for target.answer_field. Readability must not be
   sacrificed merely to preserve depth.
7. If the word budget is tight, keep the required minimum depth-0 core children and the
   selected continuation at deeper nodes, then omit optional deeper siblings and
   decorative transitions. Never solve length by exposing a private local_target.
8. Fuzzification is recursive: at every chain node replace its private local_target
   with selected child facts, and hide named relation entities even when that child
   was not expanded further. Before using a child clue, split it into the supplied
   atomic_fact_options and use at most one atomic predicate from that child. Treat
   `risk_flags` such as `numeric_detail`, `superlative_or_exclusive`,
   `definition_like`, and `compound` as warnings to generalize or omit, not as
   text to copy. Every string in any child's `redact_named_spans` is private
   and must be removed from the public wording, including locations or organizations
   that are not the node's local_target. Follow `safe_abstraction_rule` literally.
   Examples: rewrite "directed nine films starring X, more than any other director"
   as "worked repeatedly with another person"; rewrite "a seaside area on a
   peninsula formerly an island and joined by landfill" as "an unnamed recreation
   area whose geography changed over time". Do not retain "seaside", "peninsula",
   "island", or "landfill" together. Drop
   appositives, counts, rankings, dates, exact measurements, and compound clauses
   when they identify the local_target; a child may be omitted. The final question
   need not be solvable one node at a time, but no single public clue may name an
   intermediate target and no one-query route may jump to the seed target.
9. Hiding names alone is insufficient. Do not preserve a rare fingerprint made from
   multiple exact years, dates, ages, episode counts, durations, rankings, or other
   numeric tuples across one chain. Keep at most one exact numeric/date detail in a
   chain segment and faithfully generalize the others (for example, "a long-running
   series" or "a mid-century film") while retaining a usable search handle.
10. If a child clue is definition-like (for example, a geographic description that
   uniquely names a landmark), use only its broad relation/type or replace it with
   another supplied atomic child. Do not preserve a near-definition merely because
   the proper name was removed.
11. A redacted child must still contribute a public factual predicate. Never replace
   a private name with meta wording such as "identifiable without naming them",
   "whose identity is omitted", or "a person whose name is not given". Those phrases
   are not evidence. If redaction removes the entire atomic fact, omit that child and
   report no use of its id; it is valid for the public chain to stop at its parent.
12. If a child is marked compound, definition-like, or directly recoverable in the
    verifier feedback, drop that child and its descendants instead of paraphrasing
    several predicates into a new sentence. Keep the parent relation and one broad
    supplied sibling when possible.

Output:
- Use every id in required_core_root_ids and no unknown root id.
- The main interrogative must directly request target.answer_field. Do not ask for
  the hidden target and then add meta wording such as "whose field I need to
  identify"; the grammatical answer type must match the supplied answer.
- Use exactly one question mark. Present any hidden-target setup as declarative clue
  sentences, then end with the sole interrogative asking target.answer_field. Never
  write a preliminary "Which film/person/item...?" question.
- Report every node of every used chain in chain_node_usage. For each node list only
  child ids whose facts actually appear in the question; use [] for leaves.
- Do not reveal target name, id, or answer in question text.
- Stay within max_words.

Return JSON only:
{
  "question": "...?",
  "answer": "exact supplied answer",
  "used_core_path_ids": ["every required core root id"],
  "used_distractor_path_ids": [],
  "fuzzified_child_path_ids": ["used high-signal child id whose wording was generalized"],
  "chain_node_usage": [
    {
      "root_path_id": "root id",
      "node_path_id": "node id",
      "node_depth": 0,
      "used_child_path_ids": ["used child id"]
    }
  ],
  "note": "brief construction rationale"
}
"""

QUESTION_REPAIR_PROMPT = """You are the Question Repair Agent.

Revise current_question using only the supplied root-chain tree. Return a complete
replacement question and complete selection report; never output a patch.

Inputs:
- current_selection and participating_chain_nodes describe what the current question
  actually uses.
- current_question is the only public wording supplied for ordinary repair. Do not
  infer a prior draft from logs or invent a comparison question.
- prior_question is present only when the Solver ambiguity adjudicator verified a
  different same-type answer for current_question against every public clause with
  cited sources. It is the immediately preceding, non-ambiguous baseline question;
  compare it with current_question to locate the regression. Do not copy unsupported
  facts from it. An absent field means no verified ambiguity and must not be treated
  as one.
- selected_root_constraints are currently used chains.
- unexpanded_selected_root_ids lists required roots whose chain has no Local deep
  continuation. Their supplied root facts are required evidence from the program's
  deletion-minimal unique bundle. Keep those root ids selected. Ordinarily preserve
  their factual discriminators; in shortcut_prune mode, if trajectory_feedback names
  one of these root ids and shows its exact public tokens in a direct query, truthfully
  coarsen only those tokens while preserving the underlying relation and source fact.
- After a shortcut diagnosis, selected_root_constraints may contain a different
  deepest path for the same root. Treat the supplied path as the replacement plan:
  rewrite that root from the new chain and do not retain obsolete child ids merely
  because they appeared in current_selection.
- unselected_root_constraints are the only allowed source of additional evidence.
- failure_report contains one concrete diagnosis.

Repair modes:
- structural_repair: fix invalid ids, missing node usage, leakage, wording, or answer
  field while preserving the evidence selection when possible.
- uniqueness_supplement: treat blind and pre-Solver uniqueness findings as advisory.
  Do not add a child or root solely to resolve an uncertain blind candidate. Only
  repair a verified factual/scope error or a complete alternative confirmed by the
  Solver ambiguity adjudicator. Distractor roots are context only and must never be
  used as an answer discriminator. Use no external disambiguator.
  When an incorrect solver found a coherent alternative through an ambiguous Root
  replacement, add a complementary unused core child at that Root before changing
  deeper wording. If `missing_disambiguator` or the alternative identifies an exact
  fact that already exists in a supplied root clue (for example an omitted release
  year), restore that tree fact verbatim before adding anything else; uniqueness repair
  must not keep a root-only fact generalized when its exact supplied value is the
  smallest tree-only discriminator.
  `repair_priorities` is program-owned. If it contains a verified alternative and a
  discriminating child, use the smallest supplied child that excludes that
  alternative. Never add a semantic-shortcut child just because it is unused, and do
  not add a new root for an uncertain blind candidate.
  When must_exclude_verified_alternatives=true, the priority reverses: compare every
  proposed tree fact with every verified alternative. Do not add a child that the
  alternatives also satisfy (for example another fact that merely reconfirms the same
  city). Use an unused selected-root child only when it excludes every listed
  alternative; otherwise add the smallest unselected core root that does. State the
  exclusion check in note. Never claim uniqueness from a shared property.
  Use ambiguity_points as the preferred repair location. Add the smallest listed
  discriminating_child_candidate at that node while preserving the root-to-leaf
  chain report. If every ambiguity point has an empty candidate list, use an
  unselected core root; do not add a shared sibling merely because it is unused.
 - shortcut_prune: remove or lightly generalize the specific child clues and query
  phrases named by trajectory_feedback. Preserve all required core roots and enough
  connected evidence for uniqueness. Preserve each root's active reasoning depth:
  prefer keeping the same deep child ids and listing them in fuzzified_child_path_ids
  with broader wording. Do not delete an entire deep branch unless another supplied
  branch of equal depth replaces it. Do not make every search handle vague.
  Remove every root id in trajectory_feedback.shortcut_root_path_ids that is not in
  required_core_root_ids, from both the prose and used_core_path_ids, before changing
  any required Root wording. Optional roots introduced by an earlier uniqueness repair
  are not protected when real solver queries prove they are shortcuts. For a listed
  required Root, retain its id and broad factual relation but coarsen only the exact
  date/count/category/quoted phrase used by the direct query. Do not silently drop the
  Root or invent a replacement fact. The program will rerun uniqueness and supplement
  from the supplied tree if this exposes a real alternative.
  It is valid to prune aggressively in this mode; after the rewrite, the program
  reruns tree-only uniqueness and a later repair may restore only the smallest
  missing child clue needed to isolate the seed.
 A shortcut may be numeric or compositional even when no proper name is copied:
  generalize the rare count/date/value child facts named by the trajectory, report
  their child ids, and leave at least one broad relation handle per root.
  Do not respond to a Local shortcut by changing an unrelated required unexpanded Root
  fact. Prune the implicated Local child or its semantic alias first. Coarsen a required
  Root only when trajectory evidence maps a direct lookup query to that Root's exact
  public tokens; keep its id and underlying factual relation.
- The private local_target name/id is supplied only to decide what to mask; never
  copy it into public prose. For each node choose either `neutral_type_reference`
  (a type-only label such as "a person" or "a place" with pronouns) or
  `child_fact_reference` (one low-risk atomic child fact with names and defining
  details redacted). Never substitute a world-knowledge category or famous identity
  such as "a rock-and-roll singer" or "an Oscar winner". For every retained child,
  use at most one atomic_fact_options item; treat risk_flags as a reason to remove
  compound appositives and definition-like geography/biography that identifies the
  hidden entity in one lookup. Remove every `redact_named_spans` value,
  including secondary places and organizations, and follow `safe_abstraction_rule`.
  For example use "an unnamed recreation area whose geography changed over time",
  not "a coastal recreation area on a peninsula formerly an island". If a child is still directly recoverable, generalize
  or drop it and record its id in the repair note before restoring another supplied
  handle.
- Every retained child must leave a real public fact after redaction. Delete vacuous
  meta placeholders such as "identifiable without naming them", "identity omitted",
  or "whose name is not given". If no non-private predicate remains, remove that
  child's id from chain_node_usage and stop the public chain at its parent.
- solvability_restore: the solvers could not finish. Restore one broader relation or
  attribute child from the supplied tree, or add one useful unselected root. Prefer a
  navigable multi-hop route over an exact title, identifier, answer-adjacent fact, or
  target name.

The objective is at least one correct solver trajectory whose correct-trajectory API
call median and tool-call median satisfy the separate effort_target thresholds. Never
add API and tool calls into one score. If one metric reaches its full threshold, the
other may use only its explicitly supplied relaxed threshold. Include sub-agent calls
in both metrics. Do not maximize failure rate or let one lucky fast rollout dominate.
Individual intermediate layers need not be independently unique; the complete used
root chains must jointly identify the seed target.

Rules:
1. Use every required_core_root_id. Additional core roots are allowed and useful when
   their supplied chains provide independent evidence, but at most one selected root
   may be unexpanded/root-only. Every other selected root must report and use a Local
   deep chain. The program checks this condition after the response.
2. The main interrogative must directly request target.answer_field, with a
   grammatical answer type matching the supplied answer. Never ask for the hidden
   target while mentioning the requested field only as a secondary task.
3. Every fact in the new question must map to a supplied child/root clue and its path
   id. Never use target metadata or a verifier's free-text missing_disambiguator as a
   new clue.
4. Never reveal target, answer, or local_target labels.
5. Report all nodes for every used root and only children actually retained.
6. Stay within max_words and keep natural prose.
7. Preserve the supplied temporal meaning. Never add "later", "then", "before", or
   "after" merely because two child clues contain different years; use neutral
   coordination unless the supplied clue explicitly states the ordering.
8. Preserve parent-child relation scope. Every retained child fact must remain
   grammatically attached to the private node it describes. Do not rewrite an
   organization/person/work that merely identifies a hidden event, wiki, venue, or
   publication as though it directly organized, mentored, created, or otherwise
   related to the seed target. Use an explicit bridge such as "the same event also"
   when the supplied tree gives only a participation or side relation.
9. When validation cites a compound/distinctive child or an unselected tree fact,
   remove that complete sentence and the cited child (plus descendants) from
   chain_node_usage. Do not preserve it through a more elaborate paraphrase; rebuild
   from one broad atomic supplied sibling or stop at the parent.

Return JSON only:
{
  "question": "...?",
  "answer": "exact supplied answer",
  "used_core_path_ids": ["..."],
  "used_distractor_path_ids": [],
  "fuzzified_child_path_ids": ["used high-signal child id whose wording was generalized"],
  "chain_node_usage": [
    {
      "root_path_id": "root id",
      "node_path_id": "node id",
      "node_depth": 0,
      "used_child_path_ids": ["used child id"]
    }
  ],
  "note": "what changed and why"
}
"""

SOLVER_TRAJECTORY_SUMMARY_PROMPT = """You analyze solver trajectories only to guide
the next tree-based question repair. Do not propose external facts.

Classify the current question:
- target_band: at least one correct trajectory and the separate median API/tool counts
  pass the configured full-or-relaxed rule.
- shortcut: a correct trajectory used too few calls or one direct recognition query.
- too_hard: no trajectory was correct but the question remains coherent. This is not
  evidence of over-pruning because the Solver may be weak; recommend keep unless a
  sourced alternate answer is verified.
- invalid_or_ambiguous: trajectories expose contradictory clues, multiple valid
  targets, a wrong granularity, or an answer-field mismatch.
- tool_failure: the evidence is dominated by failed tools or missing trace metrics.

Before judging solver success, compare the public grammar with every supplied
participating chain. Detect relation_scope_drift: a child fact that only describes a
private parent was written as a direct relation to the seed or another ancestor.
Examples include treating an organization that merely participated in an event as
the seed project's mentoring organization, or treating a work used to identify a
wiki engine as a work about the seed. Scope drift is `invalid_or_ambiguous` because
the public question is factually different from its evidence tree; it does not
require a solver-proposed alternate target. Map it to the offending node/child and
instruct Repair to restore an explicit parent bridge without adding external facts.

Map observations to supplied root/node/child path ids whenever possible. A shortcut
includes a compositional fingerprint: several exact counts/dates/rare attributes can
identify the target even when every proper name was hidden. Separate successful search
handles from shortcuts: a useful intermediate query should normally be retained, while
an exact one-query or numeric-conjunction fingerprint should be pruned or generalized.
Also detect an early-memory shortcut: if a correct rollout names a supplied private
intermediate entity in reasoning before the first successful external search result,
classify it as a shortcut even when no query contains that name. Report the node and
the child facts that enabled the guess, and instruct Repair to replace those facts with
a neutral type reference or a lower-information supplied atomic sibling. A candidate
name first introduced by a search result is not an early-memory shortcut, but its
subsequent exact-name query should still be inspected for a direct-search route.
Also identify an intermediate-lock shortcut: a public phrase that lets the solver name a
private local_target (for example by an occupation, celebrity category, canonical landmark
definition, or famous relationship) without searching its candidate set. Record the
node_path_id and the child ids/atomic facts responsible. The next repair must replace
that phrase with a neutral type label or a less identifying atomic sibling, not merely
remove the proper name.
Root-level shortcuts must also be mapped. If a solver query directly combines one or
more `selected_root_facts`, list those root ids in `shortcut_root_path_ids`. An optional
root not present in required_core_root_ids should be removed before changing required
Root wording or Local depth. A required root may also be listed when its exact public
tokens participated in a demonstrated one-query shortcut. In that case the id means
"keep this Root selected but truthfully coarsen only the shortcut tokens"; it never
means remove the Root. The deletion-minimal uniqueness contract protects the internal
Root selection, while the public wording may generalize an exact year, count, quoted
phrase, or narrow category after trajectory evidence proves it is a direct lookup key.
End-to-end uniqueness is rerun after Repair and may restore the smallest tree fact if
the coarsening admits a verified alternative.
An incorrect rollout that names and evidences a coherent alternative satisfying the
whole question is ambiguity evidence, not ordinary solver failure. Classify it as
invalid_or_ambiguous even if another rollout found the expected answer. Do this only
when the alternative actually matches the wording; unsupported guesses remain errors.
Do not trust a solver's claim that its alternative matches. Independently check the
candidate with web/archive evidence against every public Root description and the
requested answer field. invalid_or_ambiguous requires at least one verified_alternative
with matches_all_clues=true and cited source_urls; otherwise use too_hard or shortcut.
The same standard applies to answer plurality on the intended target: if sources show
another value equally satisfies the requested singular answer field, classify
invalid_or_ambiguous and return it in verified_alternatives with
ambiguity_kind="same_target_answer". This may be discovered even when every solver
returned the reference answer.

For every verified alternate target, align it against participating_chain_nodes in
root-to-leaf order. Locate the earliest node where the public wording no longer
distinguishes the intended target from that alternative and report an ambiguity_point.
Inspect that node's available_child_clues, including unused children, and list only
facts the alternative does not satisfy as discriminating_child_candidates. A child
that merely reconfirms a shared city, date, format, or organization class is not
discriminating. If no child at that node excludes the alternative, leave the list
empty so Repair can consider an unselected core root. Use only the supplied tree.

When post_acceptance_pruning=true, keep diagnosis=target_band but still identify the
smallest set of remaining high-signal child ids/phrases that can be conservatively
generalized. Include child ids for numeric/date facts when a query is compositional,
even if the query contains a resolved target name not present in the question. Set
recommended_mode=shortcut_prune and provide one bounded pruning instruction; do not
remove all search handles or change the evidence set.
If no concrete shortcut child or query is observed, set recommended_mode=keep even
when the effort target is not reached; arbitrary rewrites are not a valid difficulty
optimization.
Never recommend solvability_restore merely because all rollouts were wrong or timed
out. Restore clues only for a verified ambiguity/invalidity that the tree can repair.

Return JSON only:
{
  "diagnosis": "target_band|shortcut|too_hard|invalid_or_ambiguous|tool_failure",
  "shared_success_path": ["ordered concise steps or queries"],
  "shortcut_queries": ["direct query or phrase"],
  "shortcut_root_path_ids": ["root id used as a direct query filter"],
  "shortcut_child_path_ids": ["child id"],
  "shortcut_node_path_ids": ["node id whose private entity was recovered directly"],
  "recoverable_intermediate_entities": [
    {"node_path_id": "node id", "reason": "why the public wording identifies it"}
  ],
  "atomic_shortcut_facts": [
    {"child_path_id": "child id", "fact": "atomic fact to generalize or drop"}
  ],
  "verified_alternatives": [
    {
      "name": "alternate target",
      "answer": "alternate answer if known",
      "ambiguity_kind": "alternate_target|same_target_answer",
      "matches_all_clues": true,
      "clue_checks": ["brief public-clue check"],
      "source_urls": ["https://..."]
    }
  ],
  "ambiguity_points": [
    {
      "alternative_name": "verified alternate target",
      "root_path_id": "root id",
      "node_path_id": "earliest ambiguous node id",
      "node_depth": 0,
      "reason": "why the public replacement also admits the alternative",
      "discriminating_child_candidates": [
        {"child_path_id": "unused child id", "reason": "fact the alternative fails"}
      ]
    }
  ],
  "useful_search_handles": ["query/relationship worth retaining"],
  "unresolved_steps": ["where incorrect solvers stalled"],
  "recommended_mode": "keep|shortcut_prune|solvability_restore|uniqueness_supplement",
  "repair_guidance": "one concise tree-only instruction"
}
"""

SOLVER_ALTERNATIVE_ADJUDICATION_PROMPT = """You are an adversarial ambiguity adjudicator.

A weak Solver returned an answer different from the intended answer. Do not assume
that answer is a valid alternative and do not reject it merely because it is wrong.
Use web/archive tools to test whether the candidate is a same-type target satisfying
every public clue in the question as written. The hidden target and standard answer
are private evaluation context only.

For each candidate:
1. Resolve the candidate at the exact target granularity.
2. Check every independent public clue, including nested relation chains and dates.
3. Check that the candidate would produce a different answer to the requested field.
4. Cite sources for the candidate and the decisive clue checks.
5. The payload's public_clauses are program-numbered. Return one clue_checks string
   beginning with every clause id (for example "[c2] PASS: ..."). Omitting a sale,
   succession, date, or nested-relation clause makes the candidate unsupported.
6. For each relation clause, state the candidate-side subject, related object, and
   edge direction. Do not combine facts about different organizations or works just
   because they occur in the same source or are connected elsewhere in history.

Only set matches_all_clues=true when all public clues are supported. A Solver's
unsupported guess, a target with the wrong entity type, or a candidate matching only
the first two clues is not ambiguity evidence.

Return JSON only:
{
  "verdict": "verified_ambiguity|no_verified_alternative|uncertain",
  "verified_alternatives": [
    {
      "name": "same-type alternate target",
      "answer": "different answer if known",
      "matches_all_clues": true,
      "clue_checks": ["[c1] PASS: one sourced check for that public clause"],
      "source_urls": ["https://..."]
    }
  ],
  "unsupported_alternatives": [
    {"name": "candidate rejected", "failed_clues": ["..."], "source_urls": ["https://..."]}
  ],
  "reason": "brief evidence-grounded verdict"
}
"""

SOLVER_RESEARCH_GUIDANCE = """Research guidance:
Before searching, decompose the question into independent clues/constraints. Watch for nested descriptions: several phrases may describe the same intermediate entity; resolve that entity first, then use it to find the final target.
Preserve grammatical relation scope. A person/organization/work mentioned as another
participant in an event or as evidence about a hidden parent need not be directly
related to the final target. Do not assume a project belongs to every mentoring
organization used to identify its program edition, or that a work used to identify a
wiki engine is about the project. Track each resolved entity against the exact parent
relation before rejecting a candidate.
For each major clue, try 2-3 short query variants. Do not paste the full question or combine answer-field terms (e.g. "cover credit", "matrix number") with the whole description; start from the most concrete factual clue.
Search iteratively: broad clue -> add one constraint -> verify the remaining candidate against all clues -> only then look up the requested answer field.
Treat task-specific entity names as unverified until they appear in the question or a tool result. Start with clue terms, relationships, dates, locations, titles, and generic synonyms; avoid candidate-name searches until that exact name appeared externally.
Synthesis: discard memory-only candidates. Cross-check the candidate with 2 independent sources when available; if any condition mismatches, reject it and search again. Keep a compact evidence trail: hard constraints -> fact sheet -> candidate hypothesis -> verified/rejected clues.
If web_extract returns 403, private-address, timeout, or no-content for a URL, do not
retry that same URL; switch to another accessible source and continue the evidence
check.
If a terminal-based network request returns HTTP 000, times out, or produces no file,
do not retry that network path in terminal; return to web_search/web_extract with a
different source."""

QUESTION_VERIFIER_PROMPT = """You are a verifier for generated BrowseComp-style questions.
Judge whether the generated question is acceptable for the hidden target.

Return JSON only with exactly:
{
  "reason": "concise explanation",
  "score": 0 or 1
}

Give score=1 when the question is structurally valid and has no hard factual or
answer-type error. Over-specific wording and likely direct-search shortcuts are
quality warnings for the Solver/Repair loop, not reasons to discard an otherwise
usable question.

Hard checks:
1. The question is natural and answerable.
2. It asks for the target answer_field, not for a different field.
3. It does not directly reveal the target name, target id, final answer, exact title,
   exact event name, or a distinctive alias that makes the target trivial.
   A relation child is also a hidden hop: do not copy the named related person,
   work, organization, place, or event from its clue; use a faithful generic
   description instead.
4. Its chain_node_usage covers every node of each used root's one selected chain and
   no unknown roots/children. Every claimed child contributes a corresponding clue
   or faithful indirect paraphrase.
5. The main interrogative asks for target.answer_field and does not ask for a
   different object or field.
6. The text has exactly one question mark; it does not first ask for the hidden
   target and then ask the requested field as a second question.
7. It incorporates at least min_distractor_paths declared valid distractor clues when
   that minimum is positive.
8. Test every used internal node, not only the seed target. Give score=0 when the
   public wording lets a reader name a private local_target from one clue, one obvious
   web query, or a defining world-knowledge category. A neutral type label such as
   "a person" or "a place" is allowed; "a rock-and-roll singer", "an Oscar winner",
   a famous landmark, occupation, nationality, or equivalent semantic alias is not.
9. Test child atomicity. A used child may contribute one atomic predicate. Reject a
   compound count/date/ranking/definition fingerprint that identifies its hidden
   local_target even when its proper name was removed.
10. Test relation scope against the supplied chain. Reject wording that promotes a
   child entity's relation to its private parent into a direct relation with the seed
   target, or that lets an adjacent pronoun/relative clause imply that false link.
   This is a hard factual error, not a difficulty warning.
11. Reject self-referential redaction placeholders such as "identifiable without
   naming them", "identity omitted", or "whose name is not given". A claimed child
   must contribute a public factual predicate, not merely say that a hidden value
   could be identified.

Quality warnings (return score=1 with a concise warning reason):
- the wording is over-specific or resembles a direct-search query;
- it contains more exact filters than necessary;
- some high-signal child wording should be pruned or generalized by Solver/Repair.

Allow score=1 when the question includes ordinary answer-field words or role names such as attendance, announcer, referee, director, actor, character, editor, illustrator, page, venue, or issue number. These are allowed because the question must ask what field to return.

Give score=0 if the question is empty, leaks the answer, names the target, exposes an
intermediate local_target through a semantic alias/definition, or uses an
exact target title/event label. Treat a possible direct-search route without an
actual target/title leak as a score=1 quality warning for Solver/Repair. Give score=0
when the main interrogative
grammatically asks for the hidden target instead of target.answer_field, even if the
requested field is mentioned elsewhere (for example, "Which film has a
cinematographer I need to identify?").
"""

BLIND_QUESTION_RESOLUTION_PROMPT = """You are a blind question-resolution auditor.

You are not given an intended target, answer, private tree, or construction notes.
Solve the public question exactly as written using web/archive research. Resolve
nested chains independently. Do not assume adjacent child descriptions are direct
attributes of the seed object unless the grammar states that relation. Test the
complete conjunction globally, not merely inside the first organization, work, or
catalogue you guess.

Return resolved only when one candidate satisfies every public clue and its requested
answer_field is sourced. Return multiple when two or more fully checked candidates
satisfy every clue with different answers. Return uncertain when a chain remains too
broad or evidence is incomplete. Do not use "no other result found" as uniqueness.
The payload contains program-numbered public_clauses. For every candidate, clue_checks
must contain exactly one evidence-grounded string beginning with each clause id, for
example "[c2] PASS: ...". Missing a clause means the candidate is incomplete.

Return JSON only:
{
  "resolution": "resolved|multiple|uncertain",
  "candidate_targets": [
    {
      "name": "candidate target",
      "answer": "answer to the public answer_field",
      "clue_checks": ["[c1] PASS: sourced check for that public clause"],
      "source_urls": ["https://..."]
    }
  ],
  "unresolved_clues": ["public clue that could not be resolved"],
  "reason": "brief evidence-grounded verdict"
}
"""

UNIQUENESS_PROMPT = """You are a uniqueness verifier for generated BrowseComp-style questions.
Your job is not to solve the question from scratch. Your job is to decide whether the supplied question and expected answer appear to identify one target uniquely.

You are given the hidden target and expected answer so you can compare possible ambiguities against the intended item. Search mentally and, when tools are available, use web/archive search to look for plausible alternative targets that also satisfy the question wording and could produce a different answer.
The payload may also contain blind_resolution, produced independently without the
hidden target. Treat blind candidates as audit leads only: adjudicate them against
every public clue, but do not let a blind result alone establish public ambiguity.
Put rejected blind candidates in unsupported_alternatives with their failed clues;
put complete matches in alternatives for audit.
The payload may also contain prior_verified_alternatives from earlier Question
revisions. They are historical audit hints, not current verdicts. Recheck every one
against the revised public wording before using it; never promote it to a current
alternative without clause-complete evidence for this exact question.
prior_unresolved_candidates are sourced blind candidates that were not yet confirmed
as full alternatives. Recheck them too: promote complete matches to alternatives or
explicitly reject them in unsupported_alternatives. Silence is not adjudication.
The payload contains program-numbered public_clauses. An alternative may set
matches_all_clues=true only when clue_checks contains a sourced check beginning with
every supplied clause id. Do not merge or omit a difficult nested-relation clause.

Be conservative:
- First interpret and solve the question wording without substituting the supplied
  hidden target into any intermediate description. Explicitly test whether each
  fuzzified person, work, venue, or organization could resolve to another entity and
  whether those alternatives form a complete alternate target.
- Only after that adversarial pass compare the intended target and expected answer.
  Do not assume a private construction-time local_target is what the public wording
  identifies.
- For every relational clause, bind the candidate target to the exact subject and
  object in the stated direction. A source that mentions the candidate, a merger,
  and a separate work is not enough; show that the same organization/event edge
  satisfies the clause. Do not stitch facts from adjacent but unrelated entities.
- Reject a scope-changing interpretation. A child entity that only helps identify a
  hidden event, wiki, venue, work, or organization must not be treated as directly
  related to the seed target. If the public grammar implies that promoted relation,
  mark the question uncertain or not unique rather than silently substituting the
  intended target's real organization/venue/work.
- Answer-field cardinality is verified upstream by the Seed Gate. Do not repeat that
  research here. This stage adjudicates whether the public Question identifies the
  intended target and whether blind candidates satisfy every public clue.
- Mark unique=false only when you independently verify a complete same-type
  alternative against every public clue, confirm that it changes the requested
  answer, and cite sources. This remains a pre-Solver audit signal; v5 defers the
  construction gate to Solver plus the stronger ambiguity adjudicator.
  A film/item matching only the easy Root year/subject clues is not an alternative
  unless its full fuzzified director/person/place chain also matches.
- Mark unique=null when the question is probably unique but the evidence is too weak, the wording is too broad, or you cannot rule out alternatives.
- Mark unique=true only when the question wording is specific enough that no plausible alternative remains after checking the likely near-miss space.

Do not reject the item. Return JSON only:
{
  "unique": true | false | null,
  "key": "unique|not_unique|uncertain",
  "reason": "short explanation",
  "alternatives": [
    {
      "name": "alternative target",
      "answer": "different answer",
      "ambiguity_kind": "alternate_target|same_target_answer",
      "matches_all_clues": true,
      "clue_checks": ["[c1] PASS: sourced check for that public clause"],
      "reason": "why it satisfies the complete question",
      "would_change_answer": true,
      "source_urls": ["https://..."]
    }
  ],
  "missing_disambiguator": "short clue that would separate the intended target, or empty"
}
"""

ANSWER_CARDINALITY_PROMPT = """You are an answer-cardinality verifier.

The target identity may already be unique. Independently test whether the requested
answer field has exactly one valid value for that target as publicly worded. When the
question field is empty, this is a seed-contract check: evaluate the supplied
answer_field directly against the exact target record. Use web/archive evidence. Do
not assume expected_answer is preferred.

Open every supplied exact-target source that is accessible and search specifically
for co-credits, alternate people/values in the same role, and complete credit or
metadata lists. A page that happens to display one value is evidence that the value
is valid, but absence of a second value on that page is not proof of cardinality.
Return single only when an exhaustive record/list or sufficiently complete
cross-source evidence supports completeness. If targeted search cannot establish
completeness, return uncertain rather than "no alternative was found".

For an explicitly plural answer_field, treat the complete answer set as one contract:
return single only when expected_answer contains every valid member and no extra
member. Return plural when another equally valid member is omitted. Ordering and the
separator used for a complete set do not create ambiguity.

Examples of plural ambiguity:
- the question asks for "the essay contributor" but the same booklet contains two
  credited essays by different people;
- the question asks for "the editor" while the source credits co-editors;
- the wording omits the title/date/role qualifier needed to select one of several
  credits.

Do not mark plural merely because biographies or unrelated roles list other names.
If one value is explicitly the main essay, lead editor, or otherwise uniquely selected,
the public question must contain that qualifier and a source must support it.

Return JSON only:
{
  "verdict": "single|plural|uncertain",
  "reason": "evidence-grounded explanation",
  "expected_answer_supported": true | false | null,
  "alternate_answers": [
    {
      "answer": "another equally valid field value",
      "same_target": true,
      "matches_question_field": true,
      "role_check": "why this is the same requested role, not another role",
      "source_urls": ["https://..."]
    }
  ],
  "missing_disambiguator": "short semantic role/title/section qualifier that would make the field singular, or empty"
}
"""

SOLVER_VERIFIER_PROMPT = """You are a strict answer-equivalence judge. You will be given a question, a standard answer, and a model answer.

Determine whether the model answer is consistent with the standard answer. They are consistent if they convey the same meaning, even if worded differently (e.g., "pink" and "it is pink" are consistent).

Respond in JSON with two fields:
- "reason": a brief explanation of why the answers are or are not consistent
- "score": 1 if consistent, 0 if inconsistent
"""
