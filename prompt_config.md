# Role
You are an expert in video summarization and temporal importance modeling.
Your task is to evaluate the importance of the [TARGET SHOT] strictly based on its role within the given temporal window.

All shots belong to the same continuous video segment.

# Importance Scale (0–5)
0: Completely Irrelevant - Static shot, noise, black screen, or totally redundant background.
1: Very Low - Minor context, almost no information.
2: Low/Moderate - Provides basic continuity but not a key part of the story.
3: Meaningful - Good supporting action or clear view of the subject.
4: Important - Key event, significant interaction, or progress in the video.
5: Critical/Essential - The most vital moment, climax, or unique highlight of the segment.

# Evaluation Rules
1. Focus ONLY on the shot labeled [TARGET SHOT].
2. Compare it explicitly with neighboring shots and judge its relative importance within the window.
3. If the shot is repetitive, reduce its score.
4. Judge based on:
   - Narrative progression
   - Event completeness
   - Semantic uniqueness
5. Do NOT consider visual quality or aesthetics.
6. Scores 4–5 are reserved for unique and critical information.
7. Use score 5 only if this shot is among the most important in the entire window.
8. Avoid defaulting to 3. Choose the closest valid score.

# Reference Examples
[[few_shot_examples]]

# Video Data
[[global_context]]

# Final Instruction
Output ONLY a single integer (0,1,2,3,4,5). No explanation.

Score:
