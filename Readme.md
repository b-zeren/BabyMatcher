## Baby Matcher

Baby Matcher is a computer vision project designed to analyze facial similarities within families. The project aims to provide a quantitative answer to the age-old question: "Does the child look more like the mother or the father?" and identifies which specific facial features are inherited from each parent.

You can try the live demo here: https://babymatcher.streamlit.app/

## How does it work

The app calculates resemblance through:

Face Detection: The system uses YuNet to automatically find and crop faces from your uploaded photos.

Feature Analysis: It uses the SFace model to convert facial features into numerical data (embeddings). This allows the AI to "measure" the unique structure of each face and individual traits.

Comparison: The AI compares the child’s data against both parents using Cosine Similarity. By measuring the mathematical distance between these points, it determines which parent the child aligns with more closely.

Results: Scores are normalized into easy-to-read percentages to show the final breakdown of resemblance.