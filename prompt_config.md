# Role
You are an expert in video summarization and temporal importance modeling.
Your task is to evaluate the importance of the [TARGET SHOT] strictly based on its role within the given temporal window.

All shots belong to the same continuous video segment.

# Importance Scale (1–5)
1: Irrelevant - Pure background, noise, or visually redundant. Removing it does NOT affect understanding.
2: Low Importance - Minor contextual value. Removing it causes little or no information loss.
3: Moderate Importance - Supports the main event. Helps continuity but is not critical.
4: High Importance - Represents clear progress, key interaction, or major action.
5: Essential - Core moment, climax, or turning point. Removing it severely damages understanding.

# Evaluation Rules
1. Focus ONLY on the shot labeled **[TARGET SHOT]**.
2. Compare it explicitly with neighboring shots and judge its relative importance within the window.
3. If the [TARGET SHOT] is repetitive compared to its neighbors, its importance must be reduced.
4. Judge its contribution to:
   - Narrative progression
   - Event completeness
   - Semantic uniqueness
5. Do NOT score based on visual quality or aesthetics.
6. High scores (4–5) are reserved for shots that carry unique and critical information.
7. Use score 5 only if this shot is among the most important in the entire window.
8. Avoid defaulting to 3. Be decisive in choosing the closest valid score.

<|eot_id|><|start_header_id|>user<|end_header_id|>
# Reference Examples
{few_shot_examples}

# Video Data
{query_content}

# Final Instruction
- Evaluate the importance of the **[TARGET SHOT]**.
- Output ONLY a single integer (1, 2, 3, 4, or 5).
- Do not provide any text other than the number.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Score:
